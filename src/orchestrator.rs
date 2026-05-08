//! Tool-calling loop.
//!
//! Drives a single conversation: ask the LLM, dispatch any tool-use blocks
//! to the MCP layer, feed results back, repeat until end_turn or the
//! iteration cap. The cap is the hard runaway-prevention boundary.

use anyhow::bail;
use async_trait::async_trait;
use futures_util::future::try_join_all;
use serde::Serialize;
use serde_json::Value;

use crate::llm::{ContentBlock, LlmClient, Message, Role, StopReason, TokenUsage, ToolSpec};

/// Per-call trace built by `McpExecutor::call` impls. `ok=false` carries the
/// error message; `args` is `None` unless the executor opts in to recording
/// them (production gates this on `CHAT_TRACE_INCLUDE_ARGS=true`).
#[derive(Debug, Clone, Serialize)]
pub struct ToolCallTrace {
    pub tool: String,
    pub server: String,
    pub ms: u64,
    pub ok: bool,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub error: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub args: Option<Value>,
}

/// Outcome of one full tool-loop run.
///
/// `tool_trace` is in dispatch order across all iterations. A failed tool
/// call surfaces as `ok=false` in its entry — execution continues so the
/// LLM can plan recovery (the matching `ContentBlock::ToolResult` carries
/// `is_error=true` to Anthropic). Fundamental errors (no-such-tool,
/// schema-mismatch) still bail the whole run with `Err`.
#[derive(Debug, Clone)]
pub struct RunOutcome {
    pub answer: String,
    pub tool_trace: Vec<ToolCallTrace>,
    pub tokens: TokenUsage,
    pub iterations: u32,
}

/// MCP tool dispatcher. Trait so the orchestrator is testable without
/// spinning up an rmcp session; the production impl in `crate::mcp` wraps
/// `RunningService`. `Send + Sync` so a `&dyn McpExecutor` can cross
/// `.await` points and be shared across tasks.
///
/// Returns `(result_text, trace)` on success **or** on tool-call failure —
/// the trace's `ok` flag distinguishes the two. Only fundamental errors
/// (validation, routing) propagate as `Err`.
#[async_trait]
pub trait McpExecutor: Send + Sync {
    async fn call(&self, name: &str, arguments: Value) -> anyhow::Result<(String, ToolCallTrace)>;
}

/// Run one conversation turn against the LLM, dispatching tools via `mcp`.
///
/// Backwards-compat single-prompt entry point: builds a one-message seed and
/// delegates to `run_with_messages`. Used by Teams scheduled trigger and
/// `--terminal-mode` paths that have no client-side history.
///
/// See `run_with_messages` for the full loop semantics and contract.
pub async fn run(
    question: String,
    system_prompt: &str,
    llm: &dyn LlmClient,
    mcp: &dyn McpExecutor,
    tool_specs: &[ToolSpec],
    max_iterations: usize,
    max_tokens: u32,
) -> anyhow::Result<RunOutcome> {
    let messages = vec![Message {
        role: Role::User,
        content: vec![ContentBlock::Text { text: question }],
    }];
    run_with_messages(
        messages,
        system_prompt,
        llm,
        mcp,
        tool_specs,
        max_iterations,
        max_tokens,
    )
    .await
}

/// Run the tool-call loop on top of a pre-seeded conversation history.
///
/// Caller MUST ensure `messages` ends with a `Role::User` turn — that's the
/// question we're answering. Validation lives one layer up (HTTP handler);
/// this fn trusts the input.
///
/// Loop semantics:
/// - `EndTurn` — fold response Text blocks into a single string, return.
/// - `ToolUse` — record the assistant turn, dispatch every tool-use,
///   append all tool-results into one user turn, iterate. Tool-call
///   failures surface as `is_error=true` ToolResult so Anthropic can plan
///   recovery; only fundamental MCP errors `bail!` the run.
/// - `MaxTokens` — log warn, return whatever text we have. Partial answer
///   beats `bail!` from a UX perspective.
/// - `Other(s)` — bail with context — unknown stop_reason for this rev.
///
/// Caller-provided `max_iterations` (typically 10) caps runaway tool loops.
pub async fn run_with_messages(
    mut messages: Vec<Message>,
    system_prompt: &str,
    llm: &dyn LlmClient,
    mcp: &dyn McpExecutor,
    tool_specs: &[ToolSpec],
    max_iterations: usize,
    max_tokens: u32,
) -> anyhow::Result<RunOutcome> {
    let mut tool_trace: Vec<ToolCallTrace> = Vec::new();
    let mut tokens = TokenUsage::default();

    for iteration in 0..max_iterations {
        // Every `.await` here must be `Send` for the future to live behind
        // `&dyn LlmClient` + `&dyn McpExecutor`. The `Send + Sync` supertraits
        // plus `async-trait`'s `Box<dyn Future + Send>` desugaring make that
        // work without manual `Pin` juggling.
        let response = llm
            .chat(system_prompt, &messages, tool_specs, max_tokens)
            .await?;

        if let Some(u) = &response.usage {
            tokens.add(u);
        }

        match response.stop_reason {
            StopReason::EndTurn => {
                return Ok(RunOutcome {
                    answer: collect_text(&response.content),
                    tool_trace,
                    tokens,
                    iterations: iteration as u32 + 1,
                });
            }
            StopReason::MaxTokens => {
                tracing::warn!(
                    iteration,
                    "anthropic max_tokens hit; returning partial response"
                );
                return Ok(RunOutcome {
                    answer: collect_text(&response.content),
                    tool_trace,
                    tokens,
                    iterations: iteration as u32 + 1,
                });
            }
            StopReason::ToolUse => {
                // Record the assistant's turn (full content, including text +
                // tool_use blocks) so the LLM sees its own request next round.
                messages.push(Message {
                    role: Role::Assistant,
                    content: response.content.clone(),
                });

                // Collect tool_use blocks first, then dispatch ALL in parallel
                // via `try_join_all`. For multi-team queries (CRM + Controlroom
                // in one round), latency drops from sum(t_i) to max(t_i) —
                // material on Salesforce cold-paths that take seconds each.
                // Order is preserved (try_join_all maintains input order) so
                // tool_use_id ↔ tool_result pairing stays correct.
                let tool_calls: Vec<(String, String, Value)> = response
                    .content
                    .into_iter()
                    .filter_map(|b| match b {
                        ContentBlock::ToolUse { id, name, input } => Some((id, name, input)),
                        _ => None,
                    })
                    .collect();
                if tool_calls.is_empty() {
                    bail!("stop_reason=tool_use but no tool_use blocks found");
                }

                let tool_futs = tool_calls.into_iter().map(|(id, name, input)| async move {
                    let (result, trace) = mcp.call(&name, input).await?;
                    let block = ContentBlock::ToolResult {
                        tool_use_id: id,
                        content: result,
                        is_error: !trace.ok,
                    };
                    Ok::<(ContentBlock, ToolCallTrace), anyhow::Error>((block, trace))
                });
                let outputs: Vec<(ContentBlock, ToolCallTrace)> = try_join_all(tool_futs).await?;
                let (results, traces): (Vec<_>, Vec<_>) = outputs.into_iter().unzip();
                tool_trace.extend(traces);

                messages.push(Message {
                    role: Role::User,
                    content: results,
                });
            }
            StopReason::Other(s) => bail!("unexpected stop_reason: {s}"),
        }
    }

    bail!("tool-call loop exceeded {max_iterations} iterations");
}

/// Concatenate the `Text` blocks of a content vector with `\n\n`.
fn collect_text(content: &[ContentBlock]) -> String {
    content
        .iter()
        .filter_map(|b| match b {
            ContentBlock::Text { text } => Some(text.as_str()),
            _ => None,
        })
        .collect::<Vec<_>>()
        .join("\n\n")
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::llm::ChatResponse;
    use crate::llm::tests::MockLlmClient;
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
        async fn call(
            &self,
            name: &str,
            _arguments: Value,
        ) -> anyhow::Result<(String, ToolCallTrace)> {
            if let Some(err) = self.errors.lock().await.get(name).cloned() {
                let trace = ToolCallTrace {
                    tool: name.to_string(),
                    server: "test".to_string(),
                    ms: 0,
                    ok: false,
                    error: Some(err.clone()),
                    args: None,
                };
                return Ok((err, trace));
            }
            let r = self.responses.lock().await;
            let result = r.get(name).cloned().ok_or_else(|| {
                anyhow::anyhow!("TestExecutor: no canned response for tool {name}")
            })?;
            let trace = ToolCallTrace {
                tool: name.to_string(),
                server: "test".to_string(),
                ms: 0,
                ok: true,
                error: None,
                args: None,
            };
            Ok((result, trace))
        }
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
        };
        let v = serde_json::to_value(&trace).unwrap();
        // skip_serializing_if keeps None fields out of the wire JSON so
        // clients don't see noisy `error: null` / `args: null` keys.
        assert!(v.get("error").is_none(), "error: null should be omitted");
        assert!(v.get("args").is_none(), "args: null should be omitted");
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
}
