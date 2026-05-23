use super::*;
use crate::agent::llm::ChatResponse;
use crate::agent::llm::tests::MockLlmClient;
use serde_json::json;
use std::collections::HashMap;
use tokio::sync::Mutex;

/// Test double for `McpExecutor`. `with_response` registers a happy-path
/// canned string; `with_error` registers a tool-call failure that
/// surfaces in the trace as `ok=false` (mirrors McpPool's behaviour).
pub struct TestExecutor {
    responses: Mutex<HashMap<String, String>>,
    errors: Mutex<HashMap<String, String>>,
}

impl TestExecutor {
    pub fn new() -> Self {
        Self {
            responses: Mutex::new(HashMap::new()),
            errors: Mutex::new(HashMap::new()),
        }
    }

    pub async fn with_response(self, name: &str, result: &str) -> Self {
        self.responses
            .lock()
            .await
            .insert(name.to_string(), result.to_string());
        self
    }

    pub async fn with_error(self, name: &str, error: &str) -> Self {
        self.errors
            .lock()
            .await
            .insert(name.to_string(), error.to_string());
        self
    }
}

#[async_trait]
impl McpExecutor for TestExecutor {
    async fn call(&self, name: &str, _arguments: Value) -> anyhow::Result<(String, ToolCallTrace)> {
        if let Some(err) = self.errors.lock().await.get(name).cloned() {
            let trace = ToolCallTrace {
                tool: name.to_string(),
                server: "test".to_string(),
                ms: 0,
                ok: false,
                error: Some(err.clone()),
                args: None,
                status: None,
                action_id: None,
            };
            return Ok((err, trace));
        }
        let r = self.responses.lock().await;
        let result = r
            .get(name)
            .cloned()
            .ok_or_else(|| anyhow::anyhow!("TestExecutor: no canned response for tool {name}"))?;
        let trace = ToolCallTrace {
            tool: name.to_string(),
            server: "test".to_string(),
            ms: 0,
            ok: true,
            error: None,
            args: None,
            status: None,
            action_id: None,
        };
        Ok((result, trace))
    }
}

#[test]
fn suggestions_event_serializes_to_snake_case() {
    let ev = ProgressEvent::Suggestions {
        texts: vec!["a".into(), "b".into(), "c".into()],
    };
    let json = serde_json::to_value(&ev).expect("serialize");
    assert_eq!(json["event"], "suggestions");
    assert_eq!(json["texts"], json!(["a", "b", "c"]));
}

#[tokio::test]
async fn orchestrator_dispatches_tool_then_returns_final_text() {
    let llm = MockLlmClient::new(vec![
        ChatResponse {
            content: vec![ContentBlock::ToolUse {
                id: "toolu_test_1".to_string(),
                name: "heartbeat_status".to_string(),
                input: json!({"limit": 5}),
            }],
            stop_reason: StopReason::ToolUse,
            usage: None,
        },
        ChatResponse {
            content: vec![ContentBlock::Text {
                text: "Last 5 heartbeats are green.".to_string(),
            }],
            stop_reason: StopReason::EndTurn,
            usage: None,
        },
    ]);
    let exec = TestExecutor::new()
        .with_response("heartbeat_status", "[{\"id\":1,\"status\":\"OK\"}]")
        .await;

    let outcome = run(
        "Hoeveel heartbeats?".to_string(),
        "system",
        &llm,
        &exec,
        &[],
        10,
        4096,
    )
    .await
    .expect("run should succeed");

    assert_eq!(outcome.answer, "Last 5 heartbeats are green.");

    // Pin the conversation-history bookkeeping: the second LLM call
    // must include the original user, the assistant's tool_use, and a
    // user message carrying a tool_result with matching tool_use_id.
    let calls = llm.calls().await;
    assert_eq!(calls.len(), 2, "expected exactly 2 LLM calls");
    let second = &calls[1];
    assert_eq!(
        second.messages.len(),
        3,
        "user → assistant → user(tool_result)"
    );
    assert!(
        matches!(second.messages[0].role, Role::User),
        "msg[0] is user question",
    );
    assert!(
        matches!(second.messages[1].role, Role::Assistant),
        "msg[1] is assistant tool_use",
    );
    match &second.messages[2].content[0] {
        ContentBlock::ToolResult {
            tool_use_id,
            content,
            is_error,
        } => {
            assert_eq!(tool_use_id, "toolu_test_1");
            assert_eq!(content, "[{\"id\":1,\"status\":\"OK\"}]");
            assert!(!is_error);
        }
        other => panic!("expected ToolResult, got {other:?}"),
    }
}

#[tokio::test]
async fn orchestrator_dispatches_multiple_tools_in_one_round_in_parallel() {
    // Single assistant turn with TWO tool_use blocks → orchestrator must
    // dispatch both, gather their results into a single user turn (per
    // Anthropic's N→N convention), then iterate.
    let llm = MockLlmClient::new(vec![
        ChatResponse {
            content: vec![
                ContentBlock::ToolUse {
                    id: "toolu_a".to_string(),
                    name: "heartbeat_status".to_string(),
                    input: json!({}),
                },
                ContentBlock::ToolUse {
                    id: "toolu_b".to_string(),
                    name: "error_analysis".to_string(),
                    input: json!({}),
                },
            ],
            stop_reason: StopReason::ToolUse,
            usage: None,
        },
        ChatResponse {
            content: vec![ContentBlock::Text {
                text: "Status green; zero errors.".to_string(),
            }],
            stop_reason: StopReason::EndTurn,
            usage: None,
        },
    ]);
    let exec = TestExecutor::new()
        .with_response("heartbeat_status", "[]")
        .await
        .with_response("error_analysis", "{}")
        .await;

    let outcome = run("status?".to_string(), "system", &llm, &exec, &[], 10, 4096)
        .await
        .expect("run should succeed");
    assert_eq!(outcome.answer, "Status green; zero errors.");

    // The second LLM call's last message must be a single user turn
    // containing BOTH tool_result blocks in input order (toolu_a, toolu_b).
    let calls = llm.calls().await;
    assert_eq!(calls.len(), 2);
    let user_turn = calls[1].messages.last().expect("messages non-empty");
    assert!(matches!(user_turn.role, Role::User));
    assert_eq!(
        user_turn.content.len(),
        2,
        "both tool_results bundled in one user turn"
    );
    match &user_turn.content[0] {
        ContentBlock::ToolResult { tool_use_id, .. } => assert_eq!(tool_use_id, "toolu_a"),
        other => panic!("expected ToolResult[0], got {other:?}"),
    }
    match &user_turn.content[1] {
        ContentBlock::ToolResult { tool_use_id, .. } => assert_eq!(tool_use_id, "toolu_b"),
        other => panic!("expected ToolResult[1], got {other:?}"),
    }
}

#[tokio::test]
async fn run_with_messages_passes_seeded_history_to_llm() {
    // Three-turn seed: user → assistant → user. LLM mock just answers.
    // Verify that on the very first LLM call, all three messages reach
    // the provider — i.e. context isn't silently dropped at the seam.
    let seed = vec![
        Message {
            role: Role::User,
            content: vec![ContentBlock::Text {
                text: "Wie is Brend?".to_string(),
            }],
        },
        Message {
            role: Role::Assistant,
            content: vec![ContentBlock::Text {
                text: "Brend is een bezoeker.".to_string(),
            }],
        },
        Message {
            role: Role::User,
            content: vec![ContentBlock::Text {
                text: "Geef me het volledige user object.".to_string(),
            }],
        },
    ];
    let llm = MockLlmClient::new(vec![ChatResponse {
        content: vec![ContentBlock::Text {
            text: "Hier is Brends volledige object: ...".to_string(),
        }],
        stop_reason: StopReason::EndTurn,
        usage: None,
    }]);
    let exec = TestExecutor::new();

    let outcome = run_with_messages(seed.clone(), "system", &llm, &exec, &[], 10, 4096)
        .await
        .expect("run_with_messages should succeed");

    assert!(outcome.answer.contains("Brend"));

    let calls = llm.calls().await;
    assert_eq!(calls.len(), 1);
    assert_eq!(calls[0].messages.len(), 3, "seed must be passed verbatim");
    match &calls[0].messages[0].content[0] {
        ContentBlock::Text { text } => assert_eq!(text, "Wie is Brend?"),
        other => panic!("expected Text in seed[0], got {other:?}"),
    }
    match &calls[0].messages[2].content[0] {
        ContentBlock::Text { text } => {
            assert_eq!(text, "Geef me het volledige user object.")
        }
        other => panic!("expected Text in seed[2], got {other:?}"),
    }
}

#[tokio::test]
async fn run_with_messages_continues_tool_loop_with_seeded_history() {
    // Seed = user + assistant + user. LLM responds with tool_use → end_turn.
    // Verify: tool gets dispatched, second LLM call has seed + assistant tool_use
    // + user tool_result = 5 messages.
    let seed = vec![
        Message {
            role: Role::User,
            content: vec![ContentBlock::Text {
                text: "Vorige vraag".to_string(),
            }],
        },
        Message {
            role: Role::Assistant,
            content: vec![ContentBlock::Text {
                text: "Vorig antwoord".to_string(),
            }],
        },
        Message {
            role: Role::User,
            content: vec![ContentBlock::Text {
                text: "Hoeveel heartbeats?".to_string(),
            }],
        },
    ];
    let llm = MockLlmClient::new(vec![
        ChatResponse {
            content: vec![ContentBlock::ToolUse {
                id: "toolu_x".to_string(),
                name: "heartbeat_status".to_string(),
                input: json!({}),
            }],
            stop_reason: StopReason::ToolUse,
            usage: None,
        },
        ChatResponse {
            content: vec![ContentBlock::Text {
                text: "Vijf heartbeats, allemaal groen.".to_string(),
            }],
            stop_reason: StopReason::EndTurn,
            usage: None,
        },
    ]);
    let exec = TestExecutor::new()
        .with_response("heartbeat_status", "[]")
        .await;

    let outcome = run_with_messages(seed, "system", &llm, &exec, &[], 10, 4096)
        .await
        .expect("run_with_messages should succeed");

    assert_eq!(outcome.answer, "Vijf heartbeats, allemaal groen.");

    let calls = llm.calls().await;
    assert_eq!(calls.len(), 2);
    assert_eq!(
        calls[1].messages.len(),
        5,
        "seed (3) + assistant tool_use + user tool_result"
    );
    assert!(matches!(calls[1].messages[3].role, Role::Assistant));
    assert!(matches!(calls[1].messages[4].role, Role::User));
    match &calls[1].messages[4].content[0] {
        ContentBlock::ToolResult {
            tool_use_id,
            content,
            is_error,
        } => {
            assert_eq!(tool_use_id, "toolu_x");
            assert_eq!(content, "[]");
            assert!(!is_error);
        }
        other => panic!("expected ToolResult, got {other:?}"),
    }
}

#[tokio::test]
async fn orchestrator_aborts_after_max_iterations() {
    // Queue 11 ToolUse responses so the loop can never reach EndTurn.
    let mut q = Vec::new();
    for i in 0..11 {
        q.push(ChatResponse {
            content: vec![ContentBlock::ToolUse {
                id: format!("toolu_{i}"),
                name: "heartbeat_status".to_string(),
                input: json!({}),
            }],
            stop_reason: StopReason::ToolUse,
            usage: None,
        });
    }
    let llm = MockLlmClient::new(q);
    let exec = TestExecutor::new()
        .with_response("heartbeat_status", "{}")
        .await;

    let err = run("ping".to_string(), "system", &llm, &exec, &[], 10, 4096)
        .await
        .expect_err("run must fail when loop exceeds bound");

    let msg = format!("{err}");
    assert!(
        msg.contains("exceeded") && msg.contains("10"),
        "error message should mention exceeded and 10, got: {msg}"
    );
}

#[tokio::test]
async fn run_outcome_collects_trace_in_dispatch_order() {
    // Two parallel tool calls in one round. Trace order must match
    // tool_calls input order — the contract that try_join_all preserves.
    let llm = MockLlmClient::new(vec![
        ChatResponse {
            content: vec![
                ContentBlock::ToolUse {
                    id: "toolu_a".to_string(),
                    name: "tool_a".to_string(),
                    input: json!({}),
                },
                ContentBlock::ToolUse {
                    id: "toolu_b".to_string(),
                    name: "tool_b".to_string(),
                    input: json!({}),
                },
            ],
            stop_reason: StopReason::ToolUse,
            usage: None,
        },
        ChatResponse {
            content: vec![ContentBlock::Text {
                text: "done".to_string(),
            }],
            stop_reason: StopReason::EndTurn,
            usage: None,
        },
    ]);
    let exec = TestExecutor::new()
        .with_response("tool_a", "ra")
        .await
        .with_response("tool_b", "rb")
        .await;

    let outcome = run("q".to_string(), "system", &llm, &exec, &[], 10, 4096)
        .await
        .expect("run should succeed");

    assert_eq!(outcome.tool_trace.len(), 2);
    assert_eq!(outcome.tool_trace[0].tool, "tool_a");
    assert_eq!(outcome.tool_trace[1].tool, "tool_b");
    assert!(outcome.tool_trace.iter().all(|t| t.ok));
}

#[tokio::test]
async fn run_outcome_sums_tokens_across_iterations() {
    let llm = MockLlmClient::new(vec![
        ChatResponse {
            content: vec![ContentBlock::ToolUse {
                id: "toolu_1".to_string(),
                name: "tool_a".to_string(),
                input: json!({}),
            }],
            stop_reason: StopReason::ToolUse,
            usage: Some(TokenUsage {
                input: 100,
                output: 50,
                cache_creation_input: None,
                cache_read_input: None,
            }),
        },
        ChatResponse {
            content: vec![ContentBlock::Text {
                text: "done".to_string(),
            }],
            stop_reason: StopReason::EndTurn,
            usage: Some(TokenUsage {
                input: 200,
                output: 30,
                cache_creation_input: None,
                cache_read_input: None,
            }),
        },
    ]);
    let exec = TestExecutor::new().with_response("tool_a", "result").await;

    let outcome = run("q".to_string(), "system", &llm, &exec, &[], 10, 4096)
        .await
        .expect("run should succeed");

    assert_eq!(outcome.tokens.input, 300);
    assert_eq!(outcome.tokens.output, 80);
}

#[tokio::test]
async fn run_outcome_records_failed_tool_with_is_error_block() {
    // TestExecutor::with_error mirrors McpPool: tool-call failure
    // surfaces as Ok((err_text, trace{ok=false})), orchestrator builds
    // is_error=true ToolResult so Anthropic can plan recovery.
    let llm = MockLlmClient::new(vec![
        ChatResponse {
            content: vec![ContentBlock::ToolUse {
                id: "toolu_fail".to_string(),
                name: "tool_fails".to_string(),
                input: json!({}),
            }],
            stop_reason: StopReason::ToolUse,
            usage: None,
        },
        ChatResponse {
            content: vec![ContentBlock::Text {
                text: "I tried but the tool failed.".to_string(),
            }],
            stop_reason: StopReason::EndTurn,
            usage: None,
        },
    ]);
    let exec = TestExecutor::new()
        .with_error("tool_fails", "salesforce timeout")
        .await;

    let outcome = run("q".to_string(), "system", &llm, &exec, &[], 10, 4096)
        .await
        .expect("run should still succeed — tool errors don't bail");

    assert_eq!(outcome.tool_trace.len(), 1);
    let trace = &outcome.tool_trace[0];
    assert!(!trace.ok);
    assert_eq!(trace.error.as_deref(), Some("salesforce timeout"));

    // Anthropic must have seen is_error=true so it could recover.
    let calls = llm.calls().await;
    let user_turn = calls[1].messages.last().expect("non-empty");
    match &user_turn.content[0] {
        ContentBlock::ToolResult { is_error, .. } => assert!(*is_error),
        other => panic!("expected ToolResult, got {other:?}"),
    }
}

#[test]
fn tool_call_trace_skips_none_error_and_args_in_json() {
    let trace = ToolCallTrace {
        tool: "x".into(),
        server: "y".into(),
        ms: 42,
        ok: true,
        error: None,
        args: None,
        status: None,
        action_id: None,
    };
    let v = serde_json::to_value(&trace).unwrap();
    // skip_serializing_if keeps None fields out of the wire JSON so
    // clients don't see noisy `error: null` / `args: null` keys.
    assert!(v.get("error").is_none(), "error: null should be omitted");
    assert!(v.get("args").is_none(), "args: null should be omitted");
    assert!(v.get("status").is_none(), "status: null should be omitted");
    assert!(
        v.get("action_id").is_none(),
        "action_id: null should be omitted",
    );
    assert_eq!(v["ok"], true);
    assert_eq!(v["ms"], 42);
}

#[tokio::test]
async fn run_outcome_iteration_count_matches_loop_passes() {
    // tool_use → tool_use → end_turn = 3 iterations of the outer loop.
    let llm = MockLlmClient::new(vec![
        ChatResponse {
            content: vec![ContentBlock::ToolUse {
                id: "t1".to_string(),
                name: "tool_a".to_string(),
                input: json!({}),
            }],
            stop_reason: StopReason::ToolUse,
            usage: None,
        },
        ChatResponse {
            content: vec![ContentBlock::ToolUse {
                id: "t2".to_string(),
                name: "tool_a".to_string(),
                input: json!({}),
            }],
            stop_reason: StopReason::ToolUse,
            usage: None,
        },
        ChatResponse {
            content: vec![ContentBlock::Text {
                text: "done".to_string(),
            }],
            stop_reason: StopReason::EndTurn,
            usage: None,
        },
    ]);
    let exec = TestExecutor::new().with_response("tool_a", "{}").await;

    let outcome = run("q".to_string(), "system", &llm, &exec, &[], 10, 4096)
        .await
        .expect("run should succeed");

    assert_eq!(outcome.iterations, 3);
}

// ---- PR-3: mode-aware dispatch tests ----

use crate::agent::modes::{ActionableMode, AgentMode, DispatchContext, ReadOnlyMode};
use crate::gateway::approval::flow::ApprovalFlow;
use crate::gateway::approval::state::ApprovalStore;
use crate::gateway::audit::AuditPublisher;
use std::sync::Arc;

fn write_tool_spec(name: &str) -> ToolSpec {
    ToolSpec {
        name: name.into(),
        description: "writes things".into(),
        input_schema: json!({"type": "object"}),
        requires_approval: true,
    }
}

fn read_tool_spec(name: &str) -> ToolSpec {
    ToolSpec {
        name: name.into(),
        description: "reads things".into(),
        input_schema: json!({"type": "object"}),
        requires_approval: false,
    }
}

fn make_actionable_with_store() -> (AgentMode, Arc<ApprovalStore>) {
    let store = Arc::new(ApprovalStore::new(std::time::Duration::from_secs(900)));
    let audit = Arc::new(AuditPublisher::new(None));
    let flow = Arc::new(ApprovalFlow::new(
        store.clone(),
        audit,
        std::time::Duration::from_secs(900),
    ));
    (AgentMode::Actionable(ActionableMode::new(flow)), store)
}

fn dispatch_ctx() -> DispatchContext {
    DispatchContext {
        correlation_id: "cid-test".into(),
        user_id: "alice".into(),
        scope: crate::gateway::auth::AuthScope::ReadAndAct,
    }
}

#[tokio::test]
async fn orchestrator_read_only_mode_blocks_write_tool() {
    let llm = MockLlmClient::new(vec![
        ChatResponse {
            content: vec![ContentBlock::ToolUse {
                id: "toolu_w1".into(),
                name: "create_company".into(),
                input: json!({"name": "Acme"}),
            }],
            stop_reason: StopReason::ToolUse,
            usage: None,
        },
        ChatResponse {
            content: vec![ContentBlock::Text {
                text: "I cannot perform that action with your current scope.".into(),
            }],
            stop_reason: StopReason::EndTurn,
            usage: None,
        },
    ]);
    let exec = TestExecutor::new();
    let mode = AgentMode::ReadOnly(ReadOnlyMode);
    let ctx = DispatchContext::default();
    let specs = vec![write_tool_spec("create_company")];

    let outcome = run_with_messages_in_mode(
        vec![Message {
            role: Role::User,
            content: vec![ContentBlock::Text {
                text: "create Acme please".into(),
            }],
        }],
        "system",
        &llm,
        &exec,
        &specs,
        10,
        4096,
        &mode,
        &ctx,
    )
    .await
    .expect("run ok");

    // The tool_result block sent back to the LLM must be is_error=true
    // with the TOOL_BLOCKED_READ_ONLY marker.
    let calls = llm.calls().await;
    let second = &calls[1];
    match &second.messages[2].content[0] {
        ContentBlock::ToolResult {
            content, is_error, ..
        } => {
            assert!(*is_error);
            assert!(content.contains("TOOL_BLOCKED_READ_ONLY"));
            assert!(content.contains("create_company"));
        }
        other => panic!("expected ToolResult, got {other:?}"),
    }
    // Trace surfaces the discriminator so the v1.4 audit feed can
    // filter blocked attempts.
    assert_eq!(outcome.tool_trace.len(), 1);
    assert_eq!(
        outcome.tool_trace[0].status.as_deref(),
        Some("blocked_read_only"),
    );
}

#[tokio::test]
async fn orchestrator_actionable_mode_proposes_write_tool() {
    let llm = MockLlmClient::new(vec![
        ChatResponse {
            content: vec![ContentBlock::ToolUse {
                id: "toolu_w1".into(),
                name: "create_company".into(),
                input: json!({"name": "Acme"}),
            }],
            stop_reason: StopReason::ToolUse,
            usage: None,
        },
        ChatResponse {
            content: vec![ContentBlock::Text {
                text: "I have proposed creating Acme; please approve.".into(),
            }],
            stop_reason: StopReason::EndTurn,
            usage: None,
        },
    ]);
    let exec = TestExecutor::new(); // never called — proposal short-circuits
    let (mode, store) = make_actionable_with_store();
    let ctx = dispatch_ctx();
    let specs = vec![write_tool_spec("create_company")];

    let outcome = run_with_messages_in_mode(
        vec![Message {
            role: Role::User,
            content: vec![ContentBlock::Text {
                text: "create Acme please".into(),
            }],
        }],
        "system",
        &llm,
        &exec,
        &specs,
        10,
        4096,
        &mode,
        &ctx,
    )
    .await
    .expect("run ok");

    // tool_result sent to the LLM contains the ACTION_PROPOSED marker.
    let calls = llm.calls().await;
    let second = &calls[1];
    let action_id = match &second.messages[2].content[0] {
        ContentBlock::ToolResult { content, .. } => {
            assert!(content.starts_with("ACTION_PROPOSED:"));
            let action_id_str = content
                .split("action_id=")
                .nth(1)
                .and_then(|s| s.split(';').next())
                .expect("marker contains action_id");
            action_id_str.parse::<uuid::Uuid>().expect("uuid parses")
        }
        other => panic!("expected ToolResult, got {other:?}"),
    };

    // Store has the proposed action with matching identity.
    let stored = store.get(action_id).expect("action stored");
    assert_eq!(stored.tool_name, "create_company");
    assert_eq!(stored.user_id, "alice");
    assert_eq!(stored.correlation_id, "cid-test");
    assert_eq!(outcome.tool_trace[0].status.as_deref(), Some("pending"),);
}

#[tokio::test]
async fn orchestrator_dispatches_read_tools_through_either_mode() {
    // Sanity — read-tools shouldn't change behaviour just because we
    // run under Actionable mode. Both modes pass through to executor.
    let exec = TestExecutor::new()
        .with_response("count_contacts", "44")
        .await;

    for mode in [
        AgentMode::ReadOnly(ReadOnlyMode),
        make_actionable_with_store().0,
    ] {
        let llm = MockLlmClient::new(vec![
            ChatResponse {
                content: vec![ContentBlock::ToolUse {
                    id: "toolu_r1".into(),
                    name: "count_contacts".into(),
                    input: json!({}),
                }],
                stop_reason: StopReason::ToolUse,
                usage: None,
            },
            ChatResponse {
                content: vec![ContentBlock::Text {
                    text: "There are 44.".into(),
                }],
                stop_reason: StopReason::EndTurn,
                usage: None,
            },
        ]);
        let specs = vec![read_tool_spec("count_contacts")];
        let outcome = run_with_messages_in_mode(
            vec![Message {
                role: Role::User,
                content: vec![ContentBlock::Text {
                    text: "how many?".into(),
                }],
            }],
            "system",
            &llm,
            &exec,
            &specs,
            10,
            4096,
            &mode,
            &dispatch_ctx(),
        )
        .await
        .expect("run ok");
        assert_eq!(outcome.answer, "There are 44.");
        // No status discriminator on read-tool dispatches.
        assert!(outcome.tool_trace[0].status.is_none());
    }
}

#[tokio::test]
async fn run_with_messages_default_shim_uses_read_only() {
    // The legacy run_with_messages shim defaults to ReadOnlyMode, which
    // means a write-tool request from the LLM gets blocked even when
    // the executor would otherwise succeed.
    let llm = MockLlmClient::new(vec![
        ChatResponse {
            content: vec![ContentBlock::ToolUse {
                id: "toolu_w1".into(),
                name: "delete_company".into(),
                input: json!({"crm_id": "abc"}),
            }],
            stop_reason: StopReason::ToolUse,
            usage: None,
        },
        ChatResponse {
            content: vec![ContentBlock::Text {
                text: "ok blocked".into(),
            }],
            stop_reason: StopReason::EndTurn,
            usage: None,
        },
    ]);
    let exec = TestExecutor::new();
    let specs = vec![write_tool_spec("delete_company")];
    let outcome = run_with_messages(
        vec![Message {
            role: Role::User,
            content: vec![ContentBlock::Text {
                text: "delete it".into(),
            }],
        }],
        "system",
        &llm,
        &exec,
        &specs,
        10,
        4096,
    )
    .await
    .expect("run ok");
    assert_eq!(
        outcome.tool_trace[0].status.as_deref(),
        Some("blocked_read_only"),
        "legacy shim must default to ReadOnly behaviour",
    );
}

// ---- Streaming-variant tests ----
//
// Pattern: run `run_with_messages_in_mode_streaming` and a drain-rx
// future concurrently via `tokio::join!`. The orchestrator owns `tx`;
// dropping it on return closes the channel, ending the drain loop.

use crate::agent::llm::StreamEvent;

fn user_seed(prompt: &str) -> Vec<Message> {
    vec![Message {
        role: Role::User,
        content: vec![ContentBlock::Text {
            text: prompt.into(),
        }],
    }]
}

#[tokio::test]
async fn streaming_emits_text_chunks_and_terminal_done() {
    let llm = MockLlmClient::new(vec![]);
    llm.queue_stream(vec![
        StreamEvent::TextDelta("Hello".into()),
        StreamEvent::TextDelta(", ".into()),
        StreamEvent::TextDelta("world".into()),
        StreamEvent::Done {
            stop_reason: StopReason::EndTurn,
            usage: Some(TokenUsage {
                input: 10,
                output: 3,
                cache_creation_input: None,
                cache_read_input: None,
            }),
            full_content: vec![ContentBlock::Text {
                text: "Hello, world".into(),
            }],
        },
    ])
    .await;
    let exec = TestExecutor::new();
    let mode = AgentMode::ReadOnly(ReadOnlyMode);
    let ctx = dispatch_ctx();

    let (tx, mut rx) = tokio::sync::mpsc::channel::<ProgressEvent>(64);
    let drain = async {
        let mut events = Vec::new();
        while let Some(ev) = rx.recv().await {
            events.push(ev);
        }
        events
    };
    let run = run_with_messages_in_mode_streaming(
        user_seed("say hi"),
        "system",
        &llm,
        &exec,
        &[],
        10,
        4096,
        &mode,
        &ctx,
        tx,
    );
    let (result, events) = tokio::join!(run, drain);

    result.expect("run ok");
    assert_eq!(events.len(), 4);
    assert_eq!(
        events[0],
        ProgressEvent::TextChunk {
            text: "Hello".into()
        }
    );
    assert_eq!(events[1], ProgressEvent::TextChunk { text: ", ".into() });
    assert_eq!(
        events[2],
        ProgressEvent::TextChunk {
            text: "world".into()
        }
    );
    match &events[3] {
        ProgressEvent::Done {
            tokens,
            iterations,
            correlation_id,
        } => {
            assert_eq!(tokens.input, 10);
            assert_eq!(tokens.output, 3);
            assert_eq!(*iterations, 1);
            assert_eq!(correlation_id, &ctx.correlation_id);
        }
        other => panic!("expected Done, got {other:?}"),
    }
}

#[tokio::test]
async fn streaming_tool_loop_emits_started_and_completed() {
    let llm = MockLlmClient::new(vec![]);
    llm.queue_stream(vec![StreamEvent::Done {
        stop_reason: StopReason::ToolUse,
        usage: None,
        full_content: vec![ContentBlock::ToolUse {
            id: "toolu_1".into(),
            name: "heartbeat_status".into(),
            input: json!({"limit": 5}),
        }],
    }])
    .await;
    llm.queue_stream(vec![
        StreamEvent::TextDelta("Last 5 heartbeats green.".into()),
        StreamEvent::Done {
            stop_reason: StopReason::EndTurn,
            usage: None,
            full_content: vec![ContentBlock::Text {
                text: "Last 5 heartbeats green.".into(),
            }],
        },
    ])
    .await;
    let exec = TestExecutor::new()
        .with_response("heartbeat_status", "[]")
        .await;
    let mode = AgentMode::ReadOnly(ReadOnlyMode);
    let ctx = dispatch_ctx();

    let (tx, mut rx) = tokio::sync::mpsc::channel::<ProgressEvent>(64);
    let drain = async {
        let mut events = Vec::new();
        while let Some(ev) = rx.recv().await {
            events.push(ev);
        }
        events
    };
    let run = run_with_messages_in_mode_streaming(
        user_seed("status?"),
        "system",
        &llm,
        &exec,
        &[],
        10,
        4096,
        &mode,
        &ctx,
        tx,
    );
    let (result, events) = tokio::join!(run, drain);
    result.expect("run ok");

    let started = events
            .iter()
            .position(|e| matches!(e, ProgressEvent::ToolCallStarted { name, .. } if name == "heartbeat_status"))
            .expect("got ToolCallStarted");
    let completed = events
            .iter()
            .position(|e| matches!(e, ProgressEvent::ToolCallCompleted { name, .. } if name == "heartbeat_status"))
            .expect("got ToolCallCompleted");
    assert!(started < completed, "started must come before completed");

    match &events[completed] {
        ProgressEvent::ToolCallCompleted { ok, status, .. } => {
            assert!(*ok);
            assert!(status.is_none(), "no status discriminator for happy read");
        }
        _ => unreachable!(),
    }

    let text_idx = events
        .iter()
        .position(|e| matches!(e, ProgressEvent::TextChunk { .. }))
        .expect("got TextChunk after tool");
    assert!(text_idx > completed, "tool completion precedes final text");
    assert!(matches!(events.last(), Some(ProgressEvent::Done { .. })));
}

#[tokio::test]
async fn streaming_parallel_tool_dispatch_preserves_order() {
    let llm = MockLlmClient::new(vec![]);
    llm.queue_stream(vec![StreamEvent::Done {
        stop_reason: StopReason::ToolUse,
        usage: None,
        full_content: vec![
            ContentBlock::ToolUse {
                id: "toolu_a".into(),
                name: "tool_a".into(),
                input: json!({}),
            },
            ContentBlock::ToolUse {
                id: "toolu_b".into(),
                name: "tool_b".into(),
                input: json!({}),
            },
        ],
    }])
    .await;
    llm.queue_stream(vec![StreamEvent::Done {
        stop_reason: StopReason::EndTurn,
        usage: None,
        full_content: vec![ContentBlock::Text {
            text: "done".into(),
        }],
    }])
    .await;
    let exec = TestExecutor::new()
        .with_response("tool_a", "ra")
        .await
        .with_response("tool_b", "rb")
        .await;
    let mode = AgentMode::ReadOnly(ReadOnlyMode);
    let ctx = dispatch_ctx();

    let (tx, mut rx) = tokio::sync::mpsc::channel::<ProgressEvent>(64);
    let drain = async {
        let mut events = Vec::new();
        while let Some(ev) = rx.recv().await {
            events.push(ev);
        }
        events
    };
    let run = run_with_messages_in_mode_streaming(
        user_seed("q"),
        "system",
        &llm,
        &exec,
        &[],
        10,
        4096,
        &mode,
        &ctx,
        tx,
    );
    let (result, events) = tokio::join!(run, drain);
    result.expect("run ok");

    let starts: Vec<&str> = events
        .iter()
        .filter_map(|e| match e {
            ProgressEvent::ToolCallStarted { name, .. } => Some(name.as_str()),
            _ => None,
        })
        .collect();
    assert_eq!(starts, vec!["tool_a", "tool_b"]);

    let completes: Vec<&str> = events
        .iter()
        .filter_map(|e| match e {
            ProgressEvent::ToolCallCompleted { name, .. } => Some(name.as_str()),
            _ => None,
        })
        .collect();
    assert_eq!(completes, vec!["tool_a", "tool_b"]);
}

#[tokio::test]
async fn streaming_approval_pending_terminates_stream() {
    let llm = MockLlmClient::new(vec![]);
    llm.queue_stream(vec![StreamEvent::Done {
        stop_reason: StopReason::ToolUse,
        usage: None,
        full_content: vec![ContentBlock::ToolUse {
            id: "toolu_w".into(),
            name: "create_company".into(),
            input: json!({"name": "Acme"}),
        }],
    }])
    .await;
    let exec = TestExecutor::new();
    let (mode, _store) = make_actionable_with_store();
    let ctx = dispatch_ctx();
    let specs = vec![write_tool_spec("create_company")];

    let (tx, mut rx) = tokio::sync::mpsc::channel::<ProgressEvent>(64);
    let drain = async {
        let mut events = Vec::new();
        while let Some(ev) = rx.recv().await {
            events.push(ev);
        }
        events
    };
    let run = run_with_messages_in_mode_streaming(
        user_seed("create Acme"),
        "system",
        &llm,
        &exec,
        &specs,
        10,
        4096,
        &mode,
        &ctx,
        tx,
    );
    let (result, events) = tokio::join!(run, drain);
    result.expect("run ok");

    // Must have ApprovalPending followed by terminal Done.
    let approval_idx = events
        .iter()
        .position(|e| matches!(e, ProgressEvent::ApprovalPending { .. }))
        .expect("got ApprovalPending");
    let done_idx = events
        .iter()
        .position(|e| matches!(e, ProgressEvent::Done { .. }))
        .expect("got terminal Done");
    assert!(
        approval_idx < done_idx,
        "ApprovalPending precedes terminal Done",
    );
    match &events[approval_idx] {
        ProgressEvent::ApprovalPending {
            tool, action_id, ..
        } => {
            assert_eq!(tool, "create_company");
            assert!(uuid::Uuid::parse_str(action_id).is_ok());
        }
        _ => unreachable!(),
    }

    // Mock got exactly one stream_chat call — no second LLM round.
    let calls = llm.calls().await;
    assert_eq!(calls.len(), 1, "orchestrator must terminate after pending");
}

#[tokio::test]
async fn streaming_thinking_delta_emits_progress_event() {
    let llm = MockLlmClient::new(vec![]);
    llm.queue_stream(vec![
        StreamEvent::ThinkingDelta("Let me check...".into()),
        StreamEvent::TextDelta("Done.".into()),
        StreamEvent::Done {
            stop_reason: StopReason::EndTurn,
            usage: None,
            full_content: vec![ContentBlock::Text {
                text: "Done.".into(),
            }],
        },
    ])
    .await;
    let exec = TestExecutor::new();
    let mode = AgentMode::ReadOnly(ReadOnlyMode);
    let ctx = dispatch_ctx();

    let (tx, mut rx) = tokio::sync::mpsc::channel::<ProgressEvent>(64);
    let drain = async {
        let mut events = Vec::new();
        while let Some(ev) = rx.recv().await {
            events.push(ev);
        }
        events
    };
    let run = run_with_messages_in_mode_streaming(
        user_seed("q"),
        "system",
        &llm,
        &exec,
        &[],
        10,
        4096,
        &mode,
        &ctx,
        tx,
    );
    let (result, events) = tokio::join!(run, drain);
    result.expect("run ok");

    assert!(matches!(events[0], ProgressEvent::Thinking { .. }));
    assert!(matches!(events[1], ProgressEvent::TextChunk { .. }));
    assert!(matches!(events.last(), Some(ProgressEvent::Done { .. })));
}

#[tokio::test]
async fn streaming_llm_stream_error_emits_error_event_and_bails() {
    let llm = MockLlmClient::new(vec![]);
    llm.queue_stream_results(vec![
        Ok(StreamEvent::TextDelta("partial".into())),
        Err(anyhow::anyhow!("anthropic stream transport error")),
    ])
    .await;
    let exec = TestExecutor::new();
    let mode = AgentMode::ReadOnly(ReadOnlyMode);
    let ctx = dispatch_ctx();

    let (tx, mut rx) = tokio::sync::mpsc::channel::<ProgressEvent>(64);
    let drain = async {
        let mut events = Vec::new();
        while let Some(ev) = rx.recv().await {
            events.push(ev);
        }
        events
    };
    let run = run_with_messages_in_mode_streaming(
        user_seed("q"),
        "system",
        &llm,
        &exec,
        &[],
        10,
        4096,
        &mode,
        &ctx,
        tx,
    );
    let (result, events) = tokio::join!(run, drain);
    result.expect_err("stream error must propagate as Err");

    assert!(matches!(events[0], ProgressEvent::TextChunk { .. }));
    match &events[1] {
        ProgressEvent::Error {
            message,
            correlation_id,
        } => {
            // Wire body must be opaque per v1.1 hardening — the
            // "transport error" substring is the *anyhow* context
            // chain we want to keep server-side-only.
            assert_eq!(message, "internal error");
            assert_eq!(correlation_id, &ctx.correlation_id);
        }
        other => panic!("expected Error, got {other:?}"),
    }
    // No terminal Done after Error — the error itself is terminal.
    assert!(
        !events
            .iter()
            .any(|e| matches!(e, ProgressEvent::Done { .. }))
    );
}

fn text_response(payload: &str) -> ChatResponse {
    ChatResponse {
        content: vec![ContentBlock::Text {
            text: payload.into(),
        }],
        stop_reason: StopReason::EndTurn,
        usage: None,
    }
}

fn sample_tool_specs() -> Vec<ToolSpec> {
    vec![
        read_tool_spec("count_registrations"),
        read_tool_spec("search_company"),
        read_tool_spec("error_analysis"),
    ]
}

#[tokio::test]
async fn generate_suggestions_happy_path_returns_three_strings() {
    let llm = MockLlmClient::new(vec![text_response(
        r#"{"items":[{"text":"Toon recente registraties","tool":"count_registrations"},{"text":"Welke bedrijven zijn er?","tool":"search_company"},{"text":"Status systemen?","tool":"error_analysis"}]}"#,
    )]);
    let got = generate_suggestions(
        &llm,
        "Er zijn 42 actieve contacten.",
        &sample_tool_specs(),
        "cid-1",
    )
    .await;
    assert_eq!(got.len(), 3);
    assert_eq!(got[0], "Toon recente registraties");
    assert_eq!(got[2], "Status systemen?");
}

#[tokio::test]
async fn generate_suggestions_malformed_json_returns_empty() {
    let llm = MockLlmClient::new(vec![text_response("dit is geen json")]);
    let got = generate_suggestions(&llm, "antwoord", &sample_tool_specs(), "cid-2").await;
    assert!(got.is_empty());
}

#[tokio::test]
async fn generate_suggestions_fewer_valid_items_returns_fewer() {
    let llm = MockLlmClient::new(vec![text_response(
        r#"{"items":[{"text":"Eerste vraag?","tool":"count_registrations"},{"text":"Tweede vraag?","tool":"search_company"}]}"#,
    )]);
    let got = generate_suggestions(&llm, "antwoord", &sample_tool_specs(), "cid-3").await;
    assert_eq!(got.len(), 2);
}

#[tokio::test]
async fn generate_suggestions_oversized_string_dropped() {
    let big = "x".repeat(101);
    let payload = format!(
        r#"{{"items":[{{"text":"Eerste vraag?","tool":"count_registrations"}},{{"text":"{big}","tool":"search_company"}}]}}"#,
    );
    let llm = MockLlmClient::new(vec![text_response(&payload)]);
    let got = generate_suggestions(&llm, "antwoord", &sample_tool_specs(), "cid-4").await;
    assert_eq!(got, vec!["Eerste vraag?".to_string()]);
}

#[tokio::test]
async fn generate_suggestions_whitespace_only_dropped() {
    let llm = MockLlmClient::new(vec![text_response(
        r#"{"items":[{"text":"     ","tool":"count_registrations"},{"text":"Geldige vraag een?","tool":"search_company"},{"text":"Geldige vraag twee?","tool":"error_analysis"}]}"#,
    )]);
    let got = generate_suggestions(&llm, "antwoord", &sample_tool_specs(), "cid-ws").await;
    assert_eq!(got.len(), 2);
    assert_eq!(got[0], "Geldige vraag een?");
}

#[tokio::test]
async fn generate_suggestions_control_chars_dropped() {
    let llm = MockLlmClient::new(vec![text_response(
        "{\"items\":[{\"text\":\"\u{202E}Pas op vraag?\",\"tool\":\"count_registrations\"},{\"text\":\"Geldige vraag een?\",\"tool\":\"search_company\"},{\"text\":\"Geldige vraag twee?\",\"tool\":\"error_analysis\"}]}",
    )]);
    let got = generate_suggestions(&llm, "antwoord", &sample_tool_specs(), "cid-ctrl").await;
    assert_eq!(got.len(), 2);
}

#[tokio::test]
async fn generate_suggestions_returns_trimmed_strings() {
    let llm = MockLlmClient::new(vec![text_response(
        r#"{"items":[{"text":"  Vraag een?  ","tool":"count_registrations"},{"text":"Vraag twee?","tool":"search_company"},{"text":"Vraag drie?","tool":"error_analysis"}]}"#,
    )]);
    let got = generate_suggestions(&llm, "antwoord", &sample_tool_specs(), "cid-trim").await;
    assert_eq!(got.len(), 3);
    assert_eq!(got[0], "Vraag een?");
}

#[tokio::test]
async fn generate_suggestions_skips_on_empty_answer() {
    let llm = MockLlmClient::new(vec![]);
    let got = generate_suggestions(&llm, "   ", &sample_tool_specs(), "cid-empty").await;
    assert!(got.is_empty());
    assert_eq!(
        llm.calls().await.len(),
        0,
        "empty-answer guard must skip the LLM call entirely",
    );
}

#[tokio::test]
async fn generate_suggestions_skips_when_no_tools_connected() {
    let llm = MockLlmClient::new(vec![]);
    let got = generate_suggestions(&llm, "antwoord", &[], "cid-notools").await;
    assert!(got.is_empty());
    assert_eq!(
        llm.calls().await.len(),
        0,
        "no connected tools means nothing is answerable — skip the LLM call",
    );
}

#[tokio::test]
async fn generate_suggestions_injects_tool_catalog_into_system_prompt() {
    let llm = MockLlmClient::new(vec![text_response(
        r#"{"items":[{"text":"Toon recente registraties","tool":"count_registrations"},{"text":"Welke bedrijven zijn er?","tool":"search_company"},{"text":"Status systemen?","tool":"error_analysis"}]}"#,
    )]);
    let _ = generate_suggestions(&llm, "antwoord", &sample_tool_specs(), "cid-cat").await;
    let calls = llm.calls().await;
    assert_eq!(calls.len(), 1);
    assert!(calls[0].system.contains("BESCHIKBARE TOOLS:"));
    assert!(calls[0].system.contains("count_registrations"));
    assert!(
        calls[0].system.contains("reads things"),
        "tool descriptions ground the suggestions",
    );
}

#[tokio::test]
async fn generate_suggestions_drops_item_with_unknown_tool() {
    let llm = MockLlmClient::new(vec![text_response(
        r#"{"items":[{"text":"Toon recente registraties","tool":"count_registrations"},{"text":"Stuur een mail","tool":"send_mail"},{"text":"Status systemen?","tool":"error_analysis"}]}"#,
    )]);
    let got = generate_suggestions(&llm, "antwoord", &sample_tool_specs(), "cid-unknown").await;
    assert_eq!(
        got,
        vec![
            "Toon recente registraties".to_string(),
            "Status systemen?".to_string(),
        ],
    );
}

#[tokio::test]
async fn generate_suggestions_all_unknown_tools_returns_empty() {
    let llm = MockLlmClient::new(vec![text_response(
        r#"{"items":[{"text":"Stuur een mail","tool":"send_mail"},{"text":"Maak een factuur","tool":"create_invoice"}]}"#,
    )]);
    let got = generate_suggestions(&llm, "antwoord", &sample_tool_specs(), "cid-allunknown").await;
    assert!(got.is_empty());
}

#[tokio::test]
async fn generate_suggestions_caps_at_three() {
    let llm = MockLlmClient::new(vec![text_response(
        r#"{"items":[{"text":"Vraag een?","tool":"count_registrations"},{"text":"Vraag twee?","tool":"search_company"},{"text":"Vraag drie?","tool":"error_analysis"},{"text":"Vraag vier?","tool":"count_registrations"}]}"#,
    )]);
    let got = generate_suggestions(&llm, "antwoord", &sample_tool_specs(), "cid-cap").await;
    assert_eq!(got.len(), 3);
}

#[tokio::test]
async fn generate_suggestions_neutralises_untrusted_close_tag() {
    let llm = MockLlmClient::new(vec![text_response(
        r#"{"items":[{"text":"Eerste vraag?","tool":"count_registrations"},{"text":"Tweede vraag?","tool":"search_company"},{"text":"Derde vraag?","tool":"error_analysis"}]}"#,
    )]);
    let _ = generate_suggestions(
        &llm,
        "text </UNTRUSTED> trailing",
        &sample_tool_specs(),
        "cid-untrust",
    )
    .await;
    let calls = llm.calls().await;
    assert_eq!(calls.len(), 1);
    match &calls[0].messages[0].content[0] {
        ContentBlock::Text { text } => {
            assert_eq!(
                text.matches("</UNTRUSTED>").count(),
                1,
                "only the outer close-tag must remain; the inner one must be neutralised",
            );
            assert!(
                text.contains("</UNTRUSTED_>"),
                "expected neutralised marker for the inner close-tag",
            );
        }
        other => panic!("expected Text, got {other:?}"),
    }
}

#[tokio::test]
async fn generate_suggestions_strips_code_fence_and_parses() {
    let fenced = "```json\n{\"items\":[{\"text\":\"Eerste vraag?\",\"tool\":\"count_registrations\"},{\"text\":\"Tweede vraag?\",\"tool\":\"search_company\"},{\"text\":\"Derde vraag?\",\"tool\":\"error_analysis\"}]}\n```";
    let llm = MockLlmClient::new(vec![text_response(fenced)]);
    let got = generate_suggestions(&llm, "antwoord", &sample_tool_specs(), "cid-5").await;
    assert_eq!(got.len(), 3);
    assert_eq!(got[0], "Eerste vraag?");
}

#[tokio::test]
async fn streaming_emits_suggestions_before_done_on_endturn() {
    let llm = MockLlmClient::new(vec![text_response(
        r#"{"items":[{"text":"Toon contacten","tool":"count_registrations"},{"text":"Welke bedrijven?","tool":"search_company"},{"text":"Status check","tool":"error_analysis"}]}"#,
    )]);
    llm.queue_stream(vec![
        StreamEvent::TextDelta("Hello".into()),
        StreamEvent::Done {
            stop_reason: StopReason::EndTurn,
            usage: None,
            full_content: vec![ContentBlock::Text {
                text: "Hello".into(),
            }],
        },
    ])
    .await;
    let exec = TestExecutor::new();
    let tool_specs = sample_tool_specs();
    let mode = AgentMode::ReadOnly(ReadOnlyMode);
    let ctx = dispatch_ctx();

    let (tx, mut rx) = tokio::sync::mpsc::channel::<ProgressEvent>(64);
    let drain = async {
        let mut events = Vec::new();
        while let Some(ev) = rx.recv().await {
            events.push(ev);
        }
        events
    };
    let run = run_with_messages_in_mode_streaming(
        user_seed("q"),
        "system",
        &llm,
        &exec,
        &tool_specs,
        10,
        4096,
        &mode,
        &ctx,
        tx,
    );
    let (result, events) = tokio::join!(run, drain);
    result.expect("run ok");

    let suggestions_idx = events
        .iter()
        .position(|e| matches!(e, ProgressEvent::Suggestions { .. }))
        .expect("got Suggestions");
    let done_idx = events
        .iter()
        .position(|e| matches!(e, ProgressEvent::Done { .. }))
        .expect("got terminal Done");
    assert!(
        suggestions_idx < done_idx,
        "Suggestions precedes terminal Done",
    );
    match &events[suggestions_idx] {
        ProgressEvent::Suggestions { texts } => {
            assert_eq!(texts.len(), 3);
            assert_eq!(texts[0], "Toon contacten");
        }
        _ => unreachable!(),
    }
}

#[tokio::test]
async fn streaming_omits_suggestions_when_llm_call_fails() {
    let llm = MockLlmClient::new(vec![]);
    llm.queue_stream(vec![
        StreamEvent::TextDelta("Hello".into()),
        StreamEvent::Done {
            stop_reason: StopReason::EndTurn,
            usage: None,
            full_content: vec![ContentBlock::Text {
                text: "Hello".into(),
            }],
        },
    ])
    .await;
    let exec = TestExecutor::new();
    let tool_specs = sample_tool_specs();
    let mode = AgentMode::ReadOnly(ReadOnlyMode);
    let ctx = dispatch_ctx();

    let (tx, mut rx) = tokio::sync::mpsc::channel::<ProgressEvent>(64);
    let drain = async {
        let mut events = Vec::new();
        while let Some(ev) = rx.recv().await {
            events.push(ev);
        }
        events
    };
    let run = run_with_messages_in_mode_streaming(
        user_seed("q"),
        "system",
        &llm,
        &exec,
        &tool_specs,
        10,
        4096,
        &mode,
        &ctx,
        tx,
    );
    let (result, events) = tokio::join!(run, drain);
    result.expect("run ok");

    assert!(
        !events
            .iter()
            .any(|e| matches!(e, ProgressEvent::Suggestions { .. })),
        "no Suggestions event when LLM call fails",
    );
    assert!(matches!(events.last(), Some(ProgressEvent::Done { .. })));
}

#[tokio::test]
async fn streaming_omits_suggestions_on_max_tokens() {
    let llm = MockLlmClient::new(vec![text_response(
        r#"{"texts":["Eerste vraag?","Tweede vraag?","Derde vraag?"]}"#,
    )]);
    llm.queue_stream(vec![
        StreamEvent::TextDelta("Partial".into()),
        StreamEvent::Done {
            stop_reason: StopReason::MaxTokens,
            usage: None,
            full_content: vec![ContentBlock::Text {
                text: "Partial".into(),
            }],
        },
    ])
    .await;
    let exec = TestExecutor::new();
    let mode = AgentMode::ReadOnly(ReadOnlyMode);
    let ctx = dispatch_ctx();

    let (tx, mut rx) = tokio::sync::mpsc::channel::<ProgressEvent>(64);
    let drain = async {
        let mut events = Vec::new();
        while let Some(ev) = rx.recv().await {
            events.push(ev);
        }
        events
    };
    let run = run_with_messages_in_mode_streaming(
        user_seed("q"),
        "system",
        &llm,
        &exec,
        &[],
        10,
        4096,
        &mode,
        &ctx,
        tx,
    );
    let (result, events) = tokio::join!(run, drain);
    result.expect("run ok");

    assert!(
        !events
            .iter()
            .any(|e| matches!(e, ProgressEvent::Suggestions { .. })),
        "no Suggestions event on max_tokens-truncated answer",
    );
    assert!(matches!(events.last(), Some(ProgressEvent::Done { .. })));
}

#[tokio::test]
async fn streaming_omits_suggestions_on_premature_close() {
    let llm = MockLlmClient::new(vec![text_response(
        r#"{"texts":["Eerste vraag?","Tweede vraag?","Derde vraag?"]}"#,
    )]);
    llm.queue_stream(vec![
        StreamEvent::TextDelta("Half".into()),
        StreamEvent::Done {
            stop_reason: StopReason::Other("premature_close".into()),
            usage: None,
            full_content: vec![ContentBlock::Text {
                text: "Half".into(),
            }],
        },
    ])
    .await;
    let exec = TestExecutor::new();
    let mode = AgentMode::ReadOnly(ReadOnlyMode);
    let ctx = dispatch_ctx();

    let (tx, mut rx) = tokio::sync::mpsc::channel::<ProgressEvent>(64);
    let drain = async {
        let mut events = Vec::new();
        while let Some(ev) = rx.recv().await {
            events.push(ev);
        }
        events
    };
    let run = run_with_messages_in_mode_streaming(
        user_seed("q"),
        "system",
        &llm,
        &exec,
        &[],
        10,
        4096,
        &mode,
        &ctx,
        tx,
    );
    let (result, events) = tokio::join!(run, drain);
    result.expect("run ok");

    assert!(
        !events
            .iter()
            .any(|e| matches!(e, ProgressEvent::Suggestions { .. })),
        "no Suggestions event on premature_close",
    );
    assert!(matches!(events.last(), Some(ProgressEvent::Done { .. })));
}

#[tokio::test]
async fn streaming_omits_suggestions_on_approval_pending() {
    let llm = MockLlmClient::new(vec![]);
    llm.queue_stream(vec![StreamEvent::Done {
        stop_reason: StopReason::ToolUse,
        usage: None,
        full_content: vec![ContentBlock::ToolUse {
            id: "toolu_w".into(),
            name: "create_company".into(),
            input: json!({"name": "Acme"}),
        }],
    }])
    .await;
    let exec = TestExecutor::new();
    let (mode, _store) = make_actionable_with_store();
    let ctx = dispatch_ctx();
    let specs = vec![write_tool_spec("create_company")];

    let (tx, mut rx) = tokio::sync::mpsc::channel::<ProgressEvent>(64);
    let drain = async {
        let mut events = Vec::new();
        while let Some(ev) = rx.recv().await {
            events.push(ev);
        }
        events
    };
    let run = run_with_messages_in_mode_streaming(
        user_seed("create Acme"),
        "system",
        &llm,
        &exec,
        &specs,
        10,
        4096,
        &mode,
        &ctx,
        tx,
    );
    let (result, events) = tokio::join!(run, drain);
    result.expect("run ok");

    assert!(
        !events
            .iter()
            .any(|e| matches!(e, ProgressEvent::Suggestions { .. })),
        "no Suggestions event on approval-pending path",
    );
}
