//! Tool-calling loop.
//!
//! Drives a single conversation: ask the LLM, dispatch any tool-use blocks
//! to the MCP layer, feed results back, repeat until end_turn or the
//! iteration cap. The cap is the hard runaway-prevention boundary.

use anyhow::bail;
use async_trait::async_trait;
use serde_json::Value;

use crate::llm::{ContentBlock, LlmClient, Message, Role, StopReason, ToolSpec};

/// MCP tool dispatcher. Trait so the orchestrator is testable without
/// spinning up an rmcp session; the production impl in `crate::mcp` wraps
/// `RunningService`. `Send + Sync` so a `&dyn McpExecutor` can cross
/// `.await` points and be shared across tasks.
#[async_trait]
pub trait McpExecutor: Send + Sync {
    async fn call(&self, name: &str, arguments: Value) -> anyhow::Result<String>;
}

/// Run one conversation turn against the LLM, dispatching tools via `mcp`.
///
/// Loop semantics:
/// - `EndTurn` — fold response Text blocks into a single string, return.
/// - `ToolUse` — record the assistant turn, dispatch every tool-use,
///   append all tool-results into one user turn, iterate.
/// - `MaxTokens` — log warn, return whatever text we have. Partial answer
///   beats `bail!` from a UX perspective.
/// - `Other(s)` — bail with context — unknown stop_reason for this rev.
///
/// Caller-provided `max_iterations` (typically 10) caps runaway tool loops.
pub async fn run(
    question: String,
    system_prompt: &str,
    llm: &dyn LlmClient,
    mcp: &dyn McpExecutor,
    tool_specs: &[ToolSpec],
    max_iterations: usize,
    max_tokens: u32,
) -> anyhow::Result<String> {
    let mut messages: Vec<Message> = vec![Message {
        role: Role::User,
        content: vec![ContentBlock::Text { text: question }],
    }];

    for iteration in 0..max_iterations {
        // Every `.await` here must be `Send` for the future to live behind
        // `&dyn LlmClient` + `&dyn McpExecutor`. The `Send + Sync` supertraits
        // plus `async-trait`'s `Box<dyn Future + Send>` desugaring make that
        // work without manual `Pin` juggling.
        let response = llm
            .chat(system_prompt, &messages, tool_specs, max_tokens)
            .await?;

        match response.stop_reason {
            StopReason::EndTurn => return Ok(collect_text(&response.content)),
            StopReason::MaxTokens => {
                tracing::warn!(
                    iteration,
                    "anthropic max_tokens hit; returning partial response"
                );
                return Ok(collect_text(&response.content));
            }
            StopReason::ToolUse => {
                // Record the assistant's turn (full content, including text +
                // tool_use blocks) so the LLM sees its own request next round.
                messages.push(Message {
                    role: Role::Assistant,
                    content: response.content.clone(),
                });

                // Dispatch every tool_use block, gather all tool_results into
                // a single user turn (Anthropic convention: N tool_use in one
                // assistant turn → N tool_result in the next user turn).
                let mut results: Vec<ContentBlock> = Vec::new();
                for block in response.content {
                    if let ContentBlock::ToolUse { id, name, input } = block {
                        let result = mcp.call(&name, input).await?;
                        results.push(ContentBlock::ToolResult {
                            tool_use_id: id,
                            content: result,
                            is_error: false,
                        });
                    }
                }
                if results.is_empty() {
                    bail!("stop_reason=tool_use but no tool_use blocks found");
                }
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

    /// Test double for `McpExecutor`: returns canned strings keyed by tool
    /// name. `bail!`s on unknown tool names.
    pub struct TestExecutor {
        responses: Mutex<HashMap<String, String>>,
    }

    impl TestExecutor {
        pub fn new() -> Self {
            Self {
                responses: Mutex::new(HashMap::new()),
            }
        }

        pub async fn with_response(self, name: &str, result: &str) -> Self {
            self.responses
                .lock()
                .await
                .insert(name.to_string(), result.to_string());
            self
        }
    }

    #[async_trait]
    impl McpExecutor for TestExecutor {
        async fn call(&self, name: &str, _arguments: Value) -> anyhow::Result<String> {
            let r = self.responses.lock().await;
            r.get(name)
                .cloned()
                .ok_or_else(|| anyhow::anyhow!("TestExecutor: no canned response for tool {name}"))
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
            },
            ChatResponse {
                content: vec![ContentBlock::Text {
                    text: "Last 5 heartbeats are green.".to_string(),
                }],
                stop_reason: StopReason::EndTurn,
            },
        ]);
        let exec = TestExecutor::new()
            .with_response("heartbeat_status", "[{\"id\":1,\"status\":\"OK\"}]")
            .await;

        let answer = run(
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

        assert_eq!(answer, "Last 5 heartbeats are green.");

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
}
