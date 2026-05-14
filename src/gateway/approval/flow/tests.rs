    use super::*;
    use crate::gateway::auth::AuthScope;
    use serde_json::json;

    fn make_flow() -> ApprovalFlow {
        let store = Arc::new(ApprovalStore::new(Duration::from_secs(900)));
        let audit = Arc::new(AuditPublisher::new(None));
        ApprovalFlow::new(store, audit, Duration::from_secs(900))
    }

    fn make_draft(user: &str) -> PendingActionDraft {
        PendingActionDraft {
            correlation_id: "cid".into(),
            user_id: user.into(),
            scope: AuthScope::ReadAndAct,
            tool_name: "create_company".into(),
            tool_args: json!({"name": "Acme"}),
            server_label: "crm".into(),
        }
    }

    #[tokio::test]
    async fn propose_inserts_proposed_action_in_store() {
        let flow = make_flow();
        let id = flow.propose(make_draft("alice")).await;
        let stored = flow.store.get(id).expect("inserted");
        assert_eq!(stored.status, ApprovalStatus::Proposed);
        assert_eq!(stored.user_id, "alice");
    }

    #[tokio::test]
    async fn confirm_happy_path_transitions_to_approved() {
        let flow = make_flow();
        let id = flow.propose(make_draft("alice")).await;
        let updated = flow.confirm(id, "alice").await.expect("confirm ok");
        assert_eq!(updated.status, ApprovalStatus::Approved);
    }

    #[tokio::test]
    async fn double_confirm_returns_already_decided() {
        let flow = make_flow();
        let id = flow.propose(make_draft("alice")).await;
        flow.confirm(id, "alice").await.expect("first confirm ok");
        let err = flow.confirm(id, "alice").await.expect_err("second fails");
        assert!(matches!(err, ApprovalError::AlreadyDecided(_)));
    }

    #[tokio::test]
    async fn confirm_with_wrong_user_returns_wrong_user() {
        let flow = make_flow();
        let id = flow.propose(make_draft("alice")).await;
        let err = flow
            .confirm(id, "mallory")
            .await
            .expect_err("hijack must be rejected");
        assert!(matches!(err, ApprovalError::WrongUser { .. }));
        // Status unchanged in store.
        assert_eq!(flow.store.get(id).unwrap().status, ApprovalStatus::Proposed);
    }

    #[tokio::test]
    async fn confirm_after_expiry_returns_expired() {
        let store = Arc::new(ApprovalStore::new(Duration::from_secs(900)));
        let audit = Arc::new(AuditPublisher::new(None));
        let flow = ApprovalFlow::new(store.clone(), audit, Duration::from_secs(900));
        let id = flow.propose(make_draft("alice")).await;
        // Manually age the action by overwriting expires_at in the store.
        store
            .try_transition(id, |a| {
                // Returning current status keeps this pure-introspection;
                // the side-effect we want is mutating expires_at, but
                // try_transition's pred only sees &PendingAction. Fall back:
                // remove + re-insert with aged timestamp.
                Ok(a.status.clone())
            })
            .expect("noop ok");
        if let Some(mut action) = store.remove(id) {
            action.expires_at = Utc::now() - chrono::Duration::seconds(1);
            store.insert(action);
        }
        let err = flow.confirm(id, "alice").await.expect_err("expired");
        assert!(matches!(err, ApprovalError::Expired(_)));
    }

    #[tokio::test]
    async fn reject_happy_path_transitions_to_rejected() {
        let flow = make_flow();
        let id = flow.propose(make_draft("alice")).await;
        let updated = flow
            .reject(id, "alice", Some("nope".into()))
            .await
            .expect("reject ok");
        assert_eq!(updated.status, ApprovalStatus::Rejected);
    }

    #[tokio::test]
    async fn mark_executed_only_after_approved() {
        let flow = make_flow();
        let id = flow.propose(make_draft("alice")).await;
        // From Proposed: must fail.
        let err = flow
            .mark_executed(id, "ok", 100)
            .await
            .expect_err("not yet approved");
        assert!(matches!(err, ApprovalError::AlreadyDecided(_)));
        flow.confirm(id, "alice").await.expect("confirm");
        let updated = flow
            .mark_executed(id, "1 record affected", 412)
            .await
            .expect("execute ok");
        assert_eq!(updated.status, ApprovalStatus::Executed);
        let stored = flow.store.get(id).unwrap();
        assert_eq!(stored.executed_result.as_deref(), Some("1 record affected"),);
        assert_eq!(stored.executed_duration_ms, Some(412));
    }

    #[tokio::test]
    async fn concurrent_confirm_exactly_one_succeeds() {
        let flow = Arc::new(make_flow());
        let id = flow.propose(make_draft("alice")).await;

        let mut handles = Vec::new();
        for _ in 0..5 {
            let f = flow.clone();
            handles.push(tokio::spawn(async move { f.confirm(id, "alice").await }));
        }
        let results: Vec<_> = futures_util::future::join_all(handles).await;
        let oks = results
            .iter()
            .filter(|r| r.as_ref().unwrap().is_ok())
            .count();
        let errs = results
            .iter()
            .filter(|r| r.as_ref().unwrap().is_err())
            .count();
        assert_eq!(oks, 1, "exactly one confirm wins the race");
        assert_eq!(errs, 4, "the other four see AlreadyDecided");
    }
