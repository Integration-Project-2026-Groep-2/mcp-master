    use super::*;
    use rmcp::model::Tool;
    use serde_json::json;
    use std::sync::Arc;

    fn tool(name: &str) -> Tool {
        let mut t = Tool::default();
        // `Tool.name` is `Cow<'static, str>`; force owned (`String -> Cow::Owned`)
        // so the borrow on `name` doesn't constrain its lifetime to `'static`.
        t.name = name.to_string().into();
        t.input_schema = Arc::new(serde_json::Map::new());
        t
    }

    #[test]
    fn tool_to_spec_translates_name_description_and_schema() {
        let schema_obj = json!({
            "type": "object",
            "properties": { "limit": { "type": "number" } },
            "required": []
        })
        .as_object()
        .cloned()
        .unwrap();

        // `rmcp::model::Tool` is `#[non_exhaustive]`; build via Default + mutate.
        let mut t = Tool::default();
        t.name = "heartbeat_status".into();
        t.description = Some("Recent heartbeats".into());
        t.input_schema = Arc::new(schema_obj.clone());

        let spec = tool_to_spec(&t);

        assert_eq!(spec.name, "heartbeat_status");
        assert_eq!(spec.description, "Recent heartbeats");
        assert_eq!(spec.input_schema, Value::Object(schema_obj));
    }

    #[test]
    fn tool_to_spec_uses_empty_description_when_none() {
        let mut t = Tool::default();
        t.name = "anon".into();
        t.description = None;
        t.input_schema = Arc::new(serde_json::Map::new());
        let spec = tool_to_spec(&t);
        assert_eq!(spec.description, "");
    }

    #[test]
    fn tool_to_spec_extracts_requires_approval_from_read_only_hint_false() {
        // Write-tool: readOnlyHint=false → requires_approval=true.
        let mut t = Tool::default();
        t.name = "create_company".into();
        t.input_schema = Arc::new(serde_json::Map::new());
        let mut ann = rmcp::model::ToolAnnotations::default();
        ann.read_only_hint = Some(false);
        t.annotations = Some(ann);
        let spec = tool_to_spec(&t);
        assert!(spec.requires_approval);
    }

    #[test]
    fn tool_to_spec_extracts_requires_approval_from_read_only_hint_true() {
        // Read-tool: readOnlyHint=true → requires_approval=false.
        let mut t = Tool::default();
        t.name = "search_contact".into();
        t.input_schema = Arc::new(serde_json::Map::new());
        let mut ann = rmcp::model::ToolAnnotations::default();
        ann.read_only_hint = Some(true);
        t.annotations = Some(ann);
        let spec = tool_to_spec(&t);
        assert!(!spec.requires_approval);
    }

    #[test]
    fn tool_to_spec_defaults_requires_approval_true_when_no_annotations() {
        // Fail-closed: tools without annotations are treated as write-tools
        // and forced through the approval-flow. An absent hint must not
        // silently downgrade a write-tool to a read-tool — server authors
        // must declare intent explicitly via readOnlyHint.
        let mut t = Tool::default();
        t.name = "legacy_tool".into();
        t.input_schema = Arc::new(serde_json::Map::new());
        t.annotations = None;
        let spec = tool_to_spec(&t);
        assert!(spec.requires_approval);
    }

    #[test]
    fn parse_endpoints_handles_comma_separated_pairs() {
        let v =
            parse_endpoints("crm@http://localhost:7001/mcp,controlroom@http://localhost:7002/mcp");
        assert_eq!(
            v,
            vec![
                ("crm".to_string(), "http://localhost:7001/mcp".to_string()),
                (
                    "controlroom".to_string(),
                    "http://localhost:7002/mcp".to_string(),
                ),
            ],
        );
    }

    #[test]
    fn parse_endpoints_skips_malformed_entries() {
        // Bad entries: no `@`, empty label, empty url. Only well-formed pairs survive.
        let v =
            parse_endpoints("crm@http://x,bad-no-at,@empty-label,empty-url@,controlroom@http://y");
        assert_eq!(
            v,
            vec![
                ("crm".to_string(), "http://x".to_string()),
                ("controlroom".to_string(), "http://y".to_string()),
            ],
        );
    }

    #[test]
    fn parse_endpoints_handles_empty_string() {
        assert!(parse_endpoints("").is_empty());
    }

    #[test]
    fn build_routing_table_assigns_tools_to_sessions() {
        let server_tools = vec![
            (
                "crm".to_string(),
                vec![tool("search_contact"), tool("count_contacts")],
            ),
            (
                "controlroom".to_string(),
                vec![tool("error_analysis"), tool("heartbeat_status")],
            ),
        ];

        let (specs, idx) = build_routing_table(&server_tools).expect("should build");
        assert_eq!(specs.len(), 4, "all tools collected into specs");
        assert_eq!(idx.get("search_contact").copied(), Some(0));
        assert_eq!(idx.get("count_contacts").copied(), Some(0));
        assert_eq!(idx.get("error_analysis").copied(), Some(1));
        assert_eq!(idx.get("heartbeat_status").copied(), Some(1));
    }

    #[test]
    fn validate_args_rejects_oversized_limit() {
        let specs = vec![ToolSpec {
            name: "search_contact".into(),
            description: "Fuzzy contact search".into(),
            input_schema: json!({
                "type": "object",
                "properties": {
                    "query": {"type": "string"},
                    "limit": {"type": "integer", "minimum": 1, "maximum": 100}
                },
                "required": ["query"]
            }),
            requires_approval: false,
        }];
        let bad = json!({"query": "x", "limit": 999_999});
        let err = validate_args_against_schema(&specs, "search_contact", &bad)
            .expect_err("limit > maximum must fail");
        let msg = format!("{err:#}");
        assert!(msg.contains("input_schema"), "got: {msg}");
    }

    #[test]
    fn validate_args_accepts_valid_args() {
        let specs = vec![ToolSpec {
            name: "search_contact".into(),
            description: "..".into(),
            input_schema: json!({
                "type": "object",
                "properties": {"query": {"type": "string"}, "limit": {"type": "integer"}},
                "required": ["query"]
            }),
            requires_approval: false,
        }];
        validate_args_against_schema(
            &specs,
            "search_contact",
            &json!({"query": "Brend", "limit": 10}),
        )
        .expect("valid args pass");
    }

    #[test]
    fn validate_args_passes_unknown_tool_through() {
        let specs: Vec<ToolSpec> = vec![];
        validate_args_against_schema(&specs, "anything", &json!({}))
            .expect("unknown tool returns Ok — caller's routing-table check handles it");
    }

    #[test]
    fn validate_args_rejects_missing_required_field() {
        let specs = vec![ToolSpec {
            name: "get_contact".into(),
            description: "..".into(),
            input_schema: json!({
                "type": "object",
                "properties": {"contact_id": {"type": "string"}},
                "required": ["contact_id"]
            }),
            requires_approval: false,
        }];
        let err = validate_args_against_schema(&specs, "get_contact", &json!({}))
            .expect_err("missing required field must fail");
        let msg = format!("{err:#}");
        assert!(msg.contains("input_schema"), "got: {msg}");
    }

    #[test]
    fn build_routing_table_errors_on_collision() {
        let server_tools = vec![
            ("crm".to_string(), vec![tool("ping")]),
            ("controlroom".to_string(), vec![tool("ping")]),
        ];

        let err =
            build_routing_table(&server_tools).expect_err("should bail on duplicate tool name");
        let msg = format!("{err}");
        assert!(msg.contains("ping"), "error msg should mention tool: {msg}");
        assert!(
            msg.contains("crm"),
            "error msg should mention first owner: {msg}"
        );
        assert!(
            msg.contains("controlroom"),
            "error msg should mention second owner: {msg}",
        );
    }

    #[test]
    fn is_transport_error_classifies_known_strings() {
        // Real-world rmcp / lapin error messages observed in production logs.
        let positives = [
            "sse stream error: body error: error decoding response body",
            "Send message error Transport [...] Client error",
            "Connection reset by peer",
            "Channel closed",
            "transport channel closed",
            "broken pipe",
            "io error: connection reset",
        ];
        for msg in positives {
            let err = anyhow::anyhow!("{msg}");
            assert!(
                is_transport_error(&err),
                "should classify as transport-error: {msg}"
            );
        }
    }

    #[test]
    fn is_transport_error_rejects_deterministic_failures() {
        // These are NOT transport errors — they're bugs in tool args or
        // server logic. Reconnect won't fix them and would just delay
        // surfacing the real problem to the LLM.
        let negatives = [
            "tool 'unknown_tool' not found",
            "input_schema validation failed: missing required field 'query'",
            "ACCESS_REFUSED login",
            "tool execution failed: invalid Salesforce ID",
        ];
        for msg in negatives {
            let err = anyhow::anyhow!("{msg}");
            assert!(
                !is_transport_error(&err),
                "should NOT classify as transport-error: {msg}"
            );
        }
    }

    #[test]
    fn is_transport_error_is_case_insensitive() {
        let err = anyhow::anyhow!("CONNECTION RESET BY PEER");
        assert!(is_transport_error(&err));
    }

    #[test]
    #[serial_test::serial]
    fn trace_args_enabled_returns_false_when_unset() {
        // SAFETY: env mutation is gated by #[serial_test::serial] across the
        // crate; no other test runs concurrently in this thread group.
        unsafe {
            std::env::remove_var("CHAT_TRACE_INCLUDE_ARGS");
        }
        assert!(!trace_args_enabled());
    }

    #[test]
    #[serial_test::serial]
    fn trace_args_enabled_returns_true_when_set_to_true() {
        unsafe {
            std::env::set_var("CHAT_TRACE_INCLUDE_ARGS", "true");
        }
        assert!(trace_args_enabled());
        unsafe {
            std::env::remove_var("CHAT_TRACE_INCLUDE_ARGS");
        }
    }

    #[test]
    #[serial_test::serial]
    fn trace_args_enabled_is_case_insensitive() {
        unsafe {
            std::env::set_var("CHAT_TRACE_INCLUDE_ARGS", "TRUE");
        }
        assert!(trace_args_enabled());
        unsafe {
            std::env::set_var("CHAT_TRACE_INCLUDE_ARGS", "False");
        }
        assert!(!trace_args_enabled());
        unsafe {
            std::env::remove_var("CHAT_TRACE_INCLUDE_ARGS");
        }
    }
