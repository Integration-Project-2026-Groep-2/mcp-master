use super::*;
use crate::gateway::approval::types::PendingAction;
use crate::gateway::auth::AuthScope;
use chrono::Duration as ChronoDuration;
use serde_json::json;

fn make_action(status: ApprovalStatus, expires_in: ChronoDuration) -> PendingAction {
    let now = Utc::now();
    PendingAction {
        action_id: Uuid::new_v4(),
        correlation_id: "cid".into(),
        user_id: "alice".into(),
        scope: AuthScope::ReadAndAct,
        tool_name: "create_company".into(),
        tool_args: json!({"name": "Acme"}),
        server_label: "crm".into(),
        proposed_at: now,
        expires_at: now + expires_in,
        status,
        executed_result: None,
        executed_duration_ms: None,
    }
}

#[test]
fn insert_then_get_returns_action() {
    let store = ApprovalStore::new(Duration::from_secs(900));
    let action = make_action(ApprovalStatus::Proposed, ChronoDuration::minutes(15));
    let id = action.action_id;
    store.insert(action.clone());
    let got = store.get(id).expect("inserted action retrievable");
    assert_eq!(got.action_id, action.action_id);
    assert_eq!(got.tool_name, action.tool_name);
}

#[test]
fn cleanup_removes_only_expired_proposed() {
    let store = ApprovalStore::new(Duration::from_secs(900));
    let fresh = make_action(ApprovalStatus::Proposed, ChronoDuration::minutes(15));
    let expired_proposed = make_action(ApprovalStatus::Proposed, ChronoDuration::minutes(-1));
    let expired_approved = make_action(ApprovalStatus::Approved, ChronoDuration::minutes(-1));
    let fresh_id = fresh.action_id;
    let expired_proposed_id = expired_proposed.action_id;
    let expired_approved_id = expired_approved.action_id;
    store.insert(fresh);
    store.insert(expired_proposed);
    store.insert(expired_approved);

    let swept = store.cleanup_expired(Utc::now());
    assert_eq!(swept.len(), 1, "only expired+proposed swept");
    assert_eq!(swept[0].action_id, expired_proposed_id);
    assert_eq!(swept[0].status, ApprovalStatus::Expired);
    assert!(store.get(fresh_id).is_some(), "fresh proposed retained");
    assert!(
        store.get(expired_approved_id).is_some(),
        "expired but Approved retained — must wait for execute",
    );
    assert!(
        store.get(expired_proposed_id).is_none(),
        "expired+proposed removed",
    );
}

#[tokio::test]
async fn try_transition_concurrent_only_first_wins() {
    let store = Arc::new(ApprovalStore::new(Duration::from_secs(900)));
    let action = make_action(ApprovalStatus::Proposed, ChronoDuration::minutes(15));
    let id = action.action_id;
    store.insert(action);

    // Two parallel callers race to transition Proposed → Approved.
    // The first wins; the second sees status != Proposed and returns
    // AlreadyDecided.
    let s1 = store.clone();
    let s2 = store.clone();
    let pred = |a: &PendingAction| -> Result<ApprovalStatus, ApprovalError> {
        if a.status != ApprovalStatus::Proposed {
            return Err(ApprovalError::AlreadyDecided(a.status.clone()));
        }
        Ok(ApprovalStatus::Approved)
    };
    let (r1, r2) = tokio::join!(
        tokio::task::spawn_blocking(move || s1.try_transition(id, pred)),
        tokio::task::spawn_blocking(move || s2.try_transition(id, pred)),
    );
    let r1 = r1.unwrap();
    let r2 = r2.unwrap();

    let ok_count = [&r1, &r2].iter().filter(|r| r.is_ok()).count();
    let err_count = [&r1, &r2].iter().filter(|r| r.is_err()).count();
    assert_eq!(ok_count, 1, "exactly one transition succeeds");
    assert_eq!(err_count, 1, "the other returns AlreadyDecided");
}

#[test]
fn try_transition_propagates_predicate_error() {
    let store = ApprovalStore::new(Duration::from_secs(900));
    let action = make_action(ApprovalStatus::Approved, ChronoDuration::minutes(15));
    let id = action.action_id;
    store.insert(action);

    let result = store.try_transition(id, |a| {
        Err::<ApprovalStatus, _>(ApprovalError::AlreadyDecided(a.status.clone()))
    });
    assert!(matches!(result, Err(ApprovalError::AlreadyDecided(_))));
    // status unchanged in store
    assert_eq!(store.get(id).unwrap().status, ApprovalStatus::Approved);
}

#[test]
fn try_transition_unknown_id_returns_not_found() {
    let store = ApprovalStore::new(Duration::from_secs(900));
    let result = store.try_transition(Uuid::new_v4(), |_| Ok(ApprovalStatus::Approved));
    assert!(matches!(result, Err(ApprovalError::NotFound(_))));
}

#[tokio::test]
async fn cleanup_task_exits_on_shutdown_signal() {
    let store = Arc::new(ApprovalStore::new(Duration::from_secs(900)));
    let audit = Arc::new(AuditPublisher::new(None));
    let (tx, rx) = watch::channel(false);

    let handle = tokio::spawn(run_cleanup_task(store, audit, rx));

    // Give the task a moment to enter its select loop, then shut down.
    tokio::time::sleep(Duration::from_millis(50)).await;
    tx.send(true).expect("shutdown channel still open");
    let result = tokio::time::timeout(Duration::from_millis(500), handle)
        .await
        .expect("cleanup task exits within 500ms of shutdown signal");
    result.expect("task panic-free");
}
