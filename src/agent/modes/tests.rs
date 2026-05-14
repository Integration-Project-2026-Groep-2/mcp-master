    use super::*;
    use crate::gateway::approval::state::ApprovalStore;
    use crate::gateway::approval::types::ApprovalStatus;
    use crate::gateway::audit::AuditPublisher;
    use async_trait::async_trait;
    use serde_json::json;
    use std::time::Duration;
    use tokio::sync::Mutex;

    fn make_flow() -> Arc<ApprovalFlow> {
        let store = Arc::new(ApprovalStore::new(Duration::from_secs(900)));
        let audit = Arc::new(AuditPublisher::new(None));
        Arc::new(ApprovalFlow::new(store, audit, Duration::from_secs(900)))
    }

    fn make_ctx() -> DispatchContext {
        DispatchContext {
            correlation_id: "cid-test".into(),
            user_id: "alice".into(),
            scope: AuthScope::ReadAndAct,
        }
    }

    /// Test executor that records calls and answers from a queue.
    struct StubExecutor {
        canned_response: Mutex<Option<(String, ToolCallTrace)>>,
        server_label: Option<String>,
    }

    #[async_trait]
    impl McpExecutor for StubExecutor {
        async fn call(&self, name: &str, _args: Value) -> anyhow::Result<(String, ToolCallTrace)> {
            self.canned_response
                .lock()
                .await
                .clone()
                .ok_or_else(|| anyhow::anyhow!("StubExecutor: no canned response for {name}"))
        }
        fn server_label_for(&self, _name: &str) -> Option<String> {
            self.server_label.clone()
        }
    }

    fn stub_with_response(text: &str, label: Option<&str>) -> StubExecutor {
        let trace = ToolCallTrace {
            tool: "search_contact".into(),
            server: label.unwrap_or("test").into(),
            ms: 1,
            ok: true,
            error: None,
            args: None,
            status: None,
            action_id: None,
        };
        StubExecutor {
            canned_response: Mutex::new(Some((text.into(), trace))),
            server_label: label.map(str::to_string),
        }
    }

    #[test]
    fn read_only_mode_advertises_no_write_tools() {
        let m = ReadOnlyMode;
        assert_eq!(m.label(), "read-only");
        assert!(!m.allows_write_tools());
    }

    #[test]
    fn actionable_mode_advertises_write_tools() {
        let m = ActionableMode::new(make_flow());
        assert_eq!(m.label(), "actionable");
        assert!(m.allows_write_tools());
    }

    #[test]
    fn agent_mode_dispatches_to_inner_via_trait() {
        let read = AgentMode::ReadOnly(ReadOnlyMode);
        let act = AgentMode::Actionable(ActionableMode::new(make_flow()));
        assert_eq!(read.label(), "read-only");
        assert_eq!(act.label(), "actionable");
        assert!(!read.allows_write_tools());
        assert!(act.allows_write_tools());
    }

    #[tokio::test]
    async fn read_only_dispatch_read_tool_delegates_to_executor() {
        let mode = ReadOnlyMode;
        let exec = stub_with_response("44 active contacts", None);
        let (result, trace) = mode
            .dispatch_read_tool(&exec, "count_contacts", json!({}))
            .await
            .unwrap();
        assert_eq!(result, "44 active contacts");
        assert!(trace.ok);
        assert!(trace.status.is_none());
    }

    #[tokio::test]
    async fn actionable_dispatch_read_tool_delegates_to_executor() {
        let mode = ActionableMode::new(make_flow());
        let exec = stub_with_response("44 active contacts", None);
        let (result, trace) = mode
            .dispatch_read_tool(&exec, "count_contacts", json!({}))
            .await
            .unwrap();
        assert_eq!(result, "44 active contacts");
        assert!(trace.ok);
    }

    #[tokio::test]
    async fn actionable_dispatch_write_tool_proposes_via_flow() {
        // Construct store+flow side-by-side so the test can inspect the
        // store after the dispatch — ApprovalFlow's own store is private.
        let store = Arc::new(ApprovalStore::new(Duration::from_secs(900)));
        let audit = Arc::new(AuditPublisher::new(None));
        let flow = Arc::new(ApprovalFlow::new(
            store.clone(),
            audit,
            Duration::from_secs(900),
        ));
        let mode = ActionableMode::new(flow);
        let exec = stub_with_response("unused", Some("crm"));
        let (text, trace) = mode
            .dispatch_write_tool(
                &exec,
                &make_ctx(),
                "create_company",
                json!({"name": "Acme"}),
            )
            .await
            .unwrap();

        // Pull the action_id out of the marker so we can fetch from the store.
        let action_id_str = text
            .split("action_id=")
            .nth(1)
            .and_then(|s| s.split(';').next())
            .expect("marker contains action_id=<uuid>");
        let action_id: uuid::Uuid = action_id_str.parse().expect("uuid parse");

        let stored = store.get(action_id).expect("flow inserted action");
        assert_eq!(stored.tool_name, "create_company");
        assert_eq!(stored.status, ApprovalStatus::Proposed);
        assert_eq!(stored.user_id, "alice");
        assert_eq!(trace.status.as_deref(), Some("pending"));
        assert_eq!(trace.server, "crm");
    }

    #[tokio::test]
    async fn actionable_dispatch_write_tool_returns_action_proposed_marker() {
        let mode = ActionableMode::new(make_flow());
        let exec = stub_with_response("unused", Some("crm"));
        let (text, _trace) = mode
            .dispatch_write_tool(&exec, &make_ctx(), "create_company", json!({}))
            .await
            .unwrap();
        assert!(
            text.starts_with("ACTION_PROPOSED:"),
            "result must start with the marker, got: {text}",
        );
        assert!(text.contains("action_id="));
        assert!(text.contains("create_company"));
    }

    #[tokio::test]
    async fn actionable_dispatch_write_tool_falls_back_when_executor_has_no_label() {
        let mode = ActionableMode::new(make_flow());
        let exec = stub_with_response("unused", None);
        let (_text, trace) = mode
            .dispatch_write_tool(&exec, &make_ctx(), "create_company", json!({}))
            .await
            .unwrap();
        assert_eq!(trace.server, "<unknown>");
    }

    #[test]
    fn build_blocked_read_only_marks_is_error_and_status() {
        let (text, trace) = build_blocked_read_only_result("delete_company", "crm");
        assert!(text.contains("TOOL_BLOCKED_READ_ONLY"));
        assert!(text.contains("delete_company"));
        assert!(!trace.ok);
        assert_eq!(trace.error.as_deref(), Some("blocked_read_only"));
        assert_eq!(trace.status.as_deref(), Some("blocked_read_only"));
        assert_eq!(trace.server, "crm");
    }
