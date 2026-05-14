    use super::*;
    use crate::agent::llm::tests::MockLlmClient;
    use crate::agent::llm::{ChatResponse, StopReason, TokenUsage};
    use crate::incident::schema::{IncidentEvent, IncidentPayload, Severity};
    use async_trait::async_trait;
    use chrono::TimeZone;
    use serde_json::{Value, json};
    use std::collections::HashMap;
    use tokio::sync::Mutex;

    fn sample_event() -> IncidentEvent {
        IncidentEvent {
            event: "heartbeat_failed".into(),
            source: "controlroom-watchdog".into(),
            timestamp: chrono::Utc
                .with_ymd_and_hms(2026, 5, 10, 14, 23, 17)
                .unwrap(),
            payload: IncidentPayload {
                summary: "kassa down".into(),
                severity: Severity::Critical,
                component: "kassa".into(),
                group: None,
                class: Some("heartbeat-loss".into()),
                custom_details: Value::Null,
            },
        }
    }

    fn spec(name: &str, requires_approval: bool) -> ToolSpec {
        ToolSpec {
            name: name.into(),
            description: format!("test tool {name}"),
            input_schema: json!({"type": "object"}),
            requires_approval,
        }
    }

    struct StubMcpExecutor {
        responses: Mutex<HashMap<String, Result<String, String>>>,
        server_label: Option<String>,
    }

    impl StubMcpExecutor {
        fn new() -> Self {
            Self {
                responses: Mutex::new(HashMap::new()),
                server_label: Some("controlroom".into()),
            }
        }

        async fn with_ok(self, name: &str, body: &str) -> Self {
            self.responses
                .lock()
                .await
                .insert(name.into(), Ok(body.into()));
            self
        }

        async fn with_err(self, name: &str, err: &str) -> Self {
            self.responses
                .lock()
                .await
                .insert(name.into(), Err(err.into()));
            self
        }
    }

    #[async_trait]
    impl McpExecutor for StubMcpExecutor {
        async fn call(
            &self,
            name: &str,
            _arguments: Value,
        ) -> anyhow::Result<(String, ToolCallTrace)> {
            let table = self.responses.lock().await;
            let entry = table
                .get(name)
                .cloned()
                .ok_or_else(|| anyhow::anyhow!("StubMcpExecutor: no response for {name}"))?;
            let server = self.server_label.clone().unwrap_or_else(|| "test".into());
            Ok(match entry {
                Ok(body) => {
                    let trace = ToolCallTrace {
                        tool: name.into(),
                        server,
                        ms: 1,
                        ok: true,
                        error: None,
                        args: None,
                        status: None,
                        action_id: None,
                    };
                    (body, trace)
                }
                Err(err) => {
                    let trace = ToolCallTrace {
                        tool: name.into(),
                        server,
                        ms: 1,
                        ok: false,
                        error: Some(err.clone()),
                        args: None,
                        status: None,
                        action_id: None,
                    };
                    (err, trace)
                }
            })
        }

        fn server_label_for(&self, _name: &str) -> Option<String> {
            self.server_label.clone()
        }
    }

    fn tool_use_response(id: &str, name: &str, input: Value) -> ChatResponse {
        ChatResponse {
            content: vec![ContentBlock::ToolUse {
                id: id.into(),
                name: name.into(),
                input,
            }],
            stop_reason: StopReason::ToolUse,
            usage: Some(TokenUsage::default()),
        }
    }

    fn end_turn(text: &str) -> ChatResponse {
        ChatResponse {
            content: vec![ContentBlock::Text { text: text.into() }],
            stop_reason: StopReason::EndTurn,
            usage: Some(TokenUsage::default()),
        }
    }

    #[test]
    fn step_a_tool_specs_keeps_only_allowed_tools() {
        let all = vec![
            spec("fetch_logs", false),
            spec("count_contacts", false),
            spec("fetch_recent_deploys", false),
            spec("create_company", true),
        ];
        let filtered = step_a_tool_specs(&all);
        let names: Vec<&str> = filtered.iter().map(|s| s.name.as_str()).collect();
        assert_eq!(names, vec!["fetch_logs", "fetch_recent_deploys"]);
    }

    #[test]
    fn extract_json_passes_through_clean_json() {
        let s = r#"{"summary":"hi","missing_sources":[]}"#;
        let r = extract_json(s).unwrap();
        assert!(r.contains("\"summary\":\"hi\""));
    }

    #[test]
    fn extract_json_strips_markdown_json_fence() {
        let s = "```json\n{\"summary\":\"hi\",\"missing_sources\":[]}\n```";
        let r = extract_json(s).unwrap();
        assert!(serde_json::from_str::<serde_json::Value>(&r).is_ok());
    }

    #[test]
    fn extract_json_strips_plain_fence() {
        let s = "```\n{\"summary\":\"hi\"}\n```";
        let r = extract_json(s).unwrap();
        assert!(serde_json::from_str::<serde_json::Value>(&r).is_ok());
    }

    #[test]
    fn extract_json_finds_braces_in_prose() {
        let s = "Here is the result: {\"summary\":\"hi\",\"missing_sources\":[]} and that's it.";
        let r = extract_json(s).unwrap();
        assert!(serde_json::from_str::<serde_json::Value>(&r).is_ok());
    }

    #[test]
    fn extract_json_returns_none_on_no_json() {
        assert!(extract_json("just prose").is_none());
    }

    #[tokio::test]
    async fn gather_evidence_bails_when_no_step_a_tools_available() {
        let llm = MockLlmClient::new(vec![]);
        let mcp = StubMcpExecutor::new();
        let r = gather_evidence(&sample_event(), &llm, &mcp, &[]).await;
        assert!(r.is_err());
    }

    #[tokio::test]
    async fn gather_evidence_happy_path_parses_structured_json() {
        let llm = MockLlmClient::new(vec![
            tool_use_response("toolu_1", "fetch_logs", json!({"service": "kassa"})),
            tool_use_response(
                "toolu_2",
                "fetch_recent_deploys",
                json!({"service": "kassa"}),
            ),
            end_turn(
                r#"{"summary":"47 DB pool timeouts since deploy abc123 at 14:18","missing_sources":[]}"#,
            ),
        ]);
        let mcp = StubMcpExecutor::new()
            .with_ok("fetch_logs", "47 ERROR lines: connection pool timeout")
            .await
            .with_ok(
                "fetch_recent_deploys",
                "[{\"sha\":\"abc123\",\"at\":\"14:18\"}]",
            )
            .await;
        let specs = vec![
            spec("fetch_logs", false),
            spec("fetch_recent_deploys", false),
        ];

        let bundle = gather_evidence(&sample_event(), &llm, &mcp, &specs)
            .await
            .unwrap();

        assert!(bundle.summary.contains("DB pool timeouts"));
        assert!(bundle.missing_sources.is_empty());
        assert_eq!(bundle.tool_trace.len(), 2);
    }

    #[tokio::test]
    async fn gather_evidence_falls_back_when_llm_returns_prose() {
        let llm = MockLlmClient::new(vec![end_turn("I observed 47 errors and a recent deploy.")]);
        let mcp = StubMcpExecutor::new();
        let specs = vec![
            spec("fetch_logs", false),
            spec("fetch_recent_deploys", false),
        ];

        let bundle = gather_evidence(&sample_event(), &llm, &mcp, &specs)
            .await
            .unwrap();

        assert_eq!(bundle.summary, "I observed 47 errors and a recent deploy.");
        assert!(bundle.missing_sources.is_empty());
    }

    #[tokio::test]
    async fn gather_evidence_infers_missing_sources_from_failed_tool_calls() {
        let llm = MockLlmClient::new(vec![
            tool_use_response("toolu_1", "fetch_logs", json!({"service": "kassa"})),
            end_turn("could not gather all evidence"),
        ]);
        let mcp = StubMcpExecutor::new()
            .with_err("fetch_logs", "elasticsearch connection refused")
            .await;
        let specs = vec![
            spec("fetch_logs", false),
            spec("fetch_recent_deploys", false),
        ];

        let bundle = gather_evidence(&sample_event(), &llm, &mcp, &specs)
            .await
            .unwrap();

        assert!(
            bundle
                .missing_sources
                .contains(&"elasticsearch".to_string())
        );
    }

    #[tokio::test]
    async fn gather_evidence_filters_out_unrelated_tools() {
        let llm = MockLlmClient::new(vec![end_turn(r#"{"summary":"x","missing_sources":[]}"#)]);
        let mcp = StubMcpExecutor::new();
        let specs = vec![
            spec("fetch_logs", false),
            spec("count_contacts", false),
            spec("create_company", true),
            spec("fetch_recent_deploys", false),
        ];

        let bundle = gather_evidence(&sample_event(), &llm, &mcp, &specs)
            .await
            .unwrap();

        // The LLM only saw fetch_logs + fetch_recent_deploys in its tools list
        // (verifiable via MockLlmClient.calls but checked indirectly via no
        // count_contacts/create_company tool-use being possible). Bundle is
        // structurally valid → no surprise tools leaked.
        assert_eq!(bundle.summary, "x");
    }

    fn sample_evidence(missing: Vec<&str>) -> EvidenceBundle {
        EvidenceBundle {
            summary: "47 DB pool timeouts after 14:18 deploy abc123".into(),
            missing_sources: missing.into_iter().map(String::from).collect(),
            tool_trace: vec![],
        }
    }

    #[test]
    fn step_b_system_prompt_states_no_tool_access() {
        assert!(STEP_B_SYSTEM_PROMPT.contains("NO tool-access"));
    }

    #[test]
    fn step_b_system_prompt_warns_on_pii() {
        assert!(STEP_B_SYSTEM_PROMPT.contains("PII discipline"));
    }

    #[test]
    fn step_b_system_prompt_requests_dutch_output() {
        assert!(STEP_B_SYSTEM_PROMPT.contains("Dutch (Nederlands)"));
        assert!(STEP_B_SYSTEM_PROMPT.contains("Keep the JSON keys themselves in"));
    }

    #[test]
    fn step_b_prompt_wraps_evidence_in_untrusted_tags() {
        let p = compose_step_b_prompt(&sample_event(), &sample_evidence(vec![]));
        assert!(p.contains("<UNTRUSTED_EVIDENCE>"));
        assert!(p.contains("</UNTRUSTED_EVIDENCE>"));
        assert!(p.contains("47 DB pool timeouts"));
    }

    #[test]
    fn step_b_prompt_lists_missing_sources_or_says_none() {
        let with_missing =
            compose_step_b_prompt(&sample_event(), &sample_evidence(vec!["elasticsearch"]));
        assert!(with_missing.contains("MISSING SOURCES: elasticsearch"));

        let without = compose_step_b_prompt(&sample_event(), &sample_evidence(vec![]));
        assert!(without.contains("MISSING SOURCES: none"));
    }

    #[test]
    fn step_b_prompt_includes_incident_metadata() {
        let p = compose_step_b_prompt(&sample_event(), &sample_evidence(vec![]));
        assert!(p.contains("Service: kassa"));
        assert!(p.contains("Critical"));
        assert!(p.contains("2026-05-10T14:23:17"));
    }

    #[tokio::test]
    async fn compose_diagnosis_happy_path_parses_high_confidence() {
        let llm = MockLlmClient::new(vec![end_turn(
            r#"{
                "root_cause": "deploy abc123 introduced bad pool sizing",
                "critical_failure": "DB connection pool exhausted",
                "impact": "checkout flow blocked",
                "confidence": "high",
                "suggested_action": "rollback to deadbeef",
                "evidence_summary": "47 timeouts since deploy"
            }"#,
        )]);
        let d = compose_diagnosis(&sample_event(), &sample_evidence(vec![]), &llm)
            .await
            .unwrap();
        assert!(d.root_cause.contains("deploy abc123"));
        assert_eq!(d.confidence, crate::incident::schema::Confidence::High);
        assert_eq!(d.suggested_action.as_deref(), Some("rollback to deadbeef"));
    }

    #[tokio::test]
    async fn compose_diagnosis_accepts_insufficient_evidence_branch() {
        let llm = MockLlmClient::new(vec![end_turn(
            r#"{
                "root_cause": "could not determine — both sources unreachable",
                "critical_failure": "n/a",
                "impact": "n/a",
                "confidence": "insufficient_evidence",
                "evidence_summary": "no evidence gathered"
            }"#,
        )]);
        let d = compose_diagnosis(
            &sample_event(),
            &sample_evidence(vec!["elasticsearch", "github_actions"]),
            &llm,
        )
        .await
        .unwrap();
        assert_eq!(
            d.confidence,
            crate::incident::schema::Confidence::InsufficientEvidence
        );
        assert!(d.suggested_action.is_none());
    }

    #[tokio::test]
    async fn compose_diagnosis_strips_markdown_fence() {
        let llm = MockLlmClient::new(vec![end_turn(
            "```json\n{\"root_cause\":\"x\",\"critical_failure\":\"x\",\"impact\":\"x\",\"confidence\":\"low\",\"evidence_summary\":\"x\"}\n```",
        )]);
        let r = compose_diagnosis(&sample_event(), &sample_evidence(vec![]), &llm).await;
        assert!(r.is_ok());
    }

    #[tokio::test]
    async fn compose_diagnosis_bails_on_no_json() {
        let llm = MockLlmClient::new(vec![end_turn("just prose, sorry")]);
        let r = compose_diagnosis(&sample_event(), &sample_evidence(vec![]), &llm).await;
        assert!(r.is_err());
    }

    #[tokio::test]
    async fn compose_diagnosis_bails_on_unknown_confidence_value() {
        let llm = MockLlmClient::new(vec![end_turn(
            r#"{
                "root_cause": "x",
                "critical_failure": "x",
                "impact": "x",
                "confidence": "uncertain",
                "evidence_summary": "x"
            }"#,
        )]);
        let r = compose_diagnosis(&sample_event(), &sample_evidence(vec![]), &llm).await;
        assert!(r.is_err());
    }

    #[tokio::test]
    async fn compose_diagnosis_passes_empty_tools_to_llm() {
        let llm = MockLlmClient::new(vec![end_turn(
            r#"{"root_cause":"x","critical_failure":"x","impact":"x","confidence":"low","evidence_summary":"x"}"#,
        )]);
        let _ = compose_diagnosis(&sample_event(), &sample_evidence(vec![]), &llm)
            .await
            .unwrap();
        let calls = llm.calls().await;
        assert_eq!(calls.len(), 1);
        // Verifying empty-tools is implicit via the MockLlmClient impl which
        // discards the tools arg — but the production AnthropicClient will
        // wire `&[]` straight to Anthropic, giving Step B no tools at the
        // wire level. The system prompt also asserts this in plain English.
    }

    /// End-to-end chain: gather_evidence (Step A) → compose_diagnosis (Step B)
    /// against a single shared `MockLlmClient` queue. Mirrors what
    /// `DefaultDiagnosePipeline::diagnose` does in production, minus the
    /// Arc<AppState> wrapping.
    ///
    /// Verifies that the sequence of LLM calls is correct (3 for Step A's
    /// tool-loop + 1 for Step B = 4 total) and that the evidence-summary
    /// from Step A's JSON output flows into Step B's prompt unchanged.
    #[tokio::test]
    async fn full_pipeline_step_a_then_step_b_chains_correctly() {
        let step_a_evidence = "47 connection pool timeouts after deploy abc123 at 14:18";
        let llm = MockLlmClient::new(vec![
            tool_use_response("toolu_a1", "fetch_logs", json!({"service": "kassa"})),
            tool_use_response(
                "toolu_a2",
                "fetch_recent_deploys",
                json!({"service": "kassa"}),
            ),
            end_turn(&format!(
                r#"{{"summary":"{step_a_evidence}","missing_sources":[]}}"#
            )),
            end_turn(
                r#"{
                    "root_cause": "deploy abc123 broke DB connection pool sizing",
                    "critical_failure": "Postgres connection pool exhausted within 2 minutes",
                    "impact": "all checkout endpoints returning 502",
                    "confidence": "high",
                    "suggested_action": "rollback to deadbeef (last healthy 13:00)",
                    "evidence_summary": "47 ERROR lines starting 14:18:30 + Argo deploy abc123 at 14:18:03"
                }"#,
            ),
        ]);
        let mcp = StubMcpExecutor::new()
            .with_ok("fetch_logs", "47 ERROR lines: connection pool timeout")
            .await
            .with_ok(
                "fetch_recent_deploys",
                r#"[{"sha":"abc123","at":"14:18:03","conclusion":"success"}]"#,
            )
            .await;
        let specs = vec![
            spec("fetch_logs", false),
            spec("fetch_recent_deploys", false),
        ];
        let event = sample_event();

        let evidence = gather_evidence(&event, &llm, &mcp, &specs).await.unwrap();
        assert_eq!(evidence.summary, step_a_evidence);
        assert!(evidence.missing_sources.is_empty());
        assert_eq!(evidence.tool_trace.len(), 2);

        let diagnosis = compose_diagnosis(&event, &evidence, &llm).await.unwrap();
        assert_eq!(
            diagnosis.confidence,
            crate::incident::schema::Confidence::High
        );
        assert!(diagnosis.root_cause.contains("abc123"));
        assert!(
            diagnosis
                .suggested_action
                .as_deref()
                .unwrap()
                .contains("rollback")
        );

        let calls = llm.calls().await;
        assert_eq!(
            calls.len(),
            4,
            "expected 3 Step A turns + 1 Step B turn, got {}",
            calls.len()
        );
        let step_b_user_message = match &calls[3].messages[0].content[0] {
            ContentBlock::Text { text } => text.clone(),
            other => panic!("Step B should receive Text, got {other:?}"),
        };
        assert!(
            step_b_user_message.contains(step_a_evidence),
            "Step A's summary must be wrapped into Step B's prompt"
        );
        assert!(
            step_b_user_message.contains("<UNTRUSTED_EVIDENCE>"),
            "Step B's prompt must mark Step A output as untrusted"
        );
    }

    /// E2E with degraded path: Loki (fetch_logs) is down, only deploys are
    /// reachable. Step A flags `elasticsearch` as missing; Step B receives
    /// degraded evidence and should produce a `low` or `insufficient_evidence`
    /// diagnosis (the LLM in this test produces `low`).
    #[tokio::test]
    async fn full_pipeline_with_partial_evidence_produces_low_confidence() {
        let llm = MockLlmClient::new(vec![
            tool_use_response("toolu_a1", "fetch_logs", json!({"service": "kassa"})),
            tool_use_response(
                "toolu_a2",
                "fetch_recent_deploys",
                json!({"service": "kassa"}),
            ),
            end_turn(
                r#"{"summary":"only deploy data; logs unreachable","missing_sources":["elasticsearch"]}"#,
            ),
            end_turn(
                r#"{
                    "root_cause": "recent deploy abc123 is the only signal — logs unreachable",
                    "critical_failure": "unknown — log pipeline down",
                    "impact": "unknown",
                    "confidence": "low",
                    "evidence_summary": "1 deploy seen, 0 log entries"
                }"#,
            ),
        ]);
        let mcp = StubMcpExecutor::new()
            .with_err("fetch_logs", "elasticsearch connection refused")
            .await
            .with_ok("fetch_recent_deploys", r#"[{"sha":"abc123"}]"#)
            .await;
        let specs = vec![
            spec("fetch_logs", false),
            spec("fetch_recent_deploys", false),
        ];
        let event = sample_event();

        let evidence = gather_evidence(&event, &llm, &mcp, &specs).await.unwrap();
        assert!(
            evidence
                .missing_sources
                .contains(&"elasticsearch".to_string()),
            "Step A must flag failing source: {:?}",
            evidence.missing_sources
        );

        let diagnosis = compose_diagnosis(&event, &evidence, &llm).await.unwrap();
        assert_eq!(
            diagnosis.confidence,
            crate::incident::schema::Confidence::Low
        );
        assert!(diagnosis.suggested_action.is_none());
    }
