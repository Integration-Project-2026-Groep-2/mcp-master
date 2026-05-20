//! Tool-calling loop.
//!
//! Drives a single conversation: ask the LLM, dispatch any tool-use blocks
//! to the MCP layer, feed results back, repeat until end_turn or the
//! iteration cap. The cap is the hard runaway-prevention boundary.

use anyhow::bail;
use async_trait::async_trait;
use futures_util::StreamExt;
use futures_util::future::try_join_all;
use serde::{Deserialize, Serialize};
use serde_json::Value;
use std::time::Duration;
use tokio::sync::mpsc;

use crate::agent::llm::{
    ContentBlock, LlmClient, Message, Role, StopReason, StreamEvent, TokenUsage, ToolSpec,
};
use crate::agent::modes::{
    AgentMode, DispatchContext, ReadOnlyMode, build_blocked_read_only_result,
};
use crate::agent::prompts::SUGGESTIONS_SYSTEM_PROMPT;

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
    /// Lifecycle marker for the v1.4 audit feed when the tool didn't
    /// actually execute. `Some("pending")` for an action awaiting approval;
    /// `Some("blocked_read_only")` for a write-tool denied by ReadOnlyMode.
    /// `None` for normal dispatched calls (skip-if-none keeps the wire shape).
    #[serde(skip_serializing_if = "Option::is_none")]
    pub status: Option<String>,
    /// PendingAction id for write-tools intercepted by ActionableMode.
    /// Drupal `jarvis_chat` reads this to render the approval-card without
    /// parsing it from the marker text. `None` for everything else.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub action_id: Option<String>,
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

/// Higher-level events emitted by [`run_with_messages_in_mode_streaming`] over
/// an `mpsc::Sender` so the HTTP handler (PR4) can forward them as SSE.
///
/// `Thinking` / `TextChunk` are forwarded straight from the provider's
/// `StreamEvent` deltas; `ToolCallStarted` / `ToolCallCompleted` /
/// `ApprovalPending` are synthesised at the orchestrator-level boundary so
/// the client sees a coherent "thought → tool call → answer" narrative
/// without having to track provider-level block indices.
///
/// `Done` is terminal — always sent exactly once at the end of a successful
/// run, carrying the totals so a single trailing event closes the stream.
/// `Error` is also terminal on the failure path.
///
/// `args_preview` is intentionally absent on `ToolCallStarted`: tool args
/// may contain GDPR data (VATs, emails, names) per `HTTP_API.md §1`. UI
/// renders the tool name; full args remain audit-feed-only.
#[derive(Debug, Clone, Serialize, PartialEq)]
#[serde(tag = "event", rename_all = "snake_case")]
#[allow(dead_code)] // reachable via /chat/stream HTTP route in PR4
pub enum ProgressEvent {
    Thinking {
        text: String,
    },
    TextChunk {
        text: String,
    },
    ToolCallStarted {
        name: String,
        #[serde(skip_serializing_if = "Option::is_none")]
        server: Option<String>,
    },
    ToolCallCompleted {
        name: String,
        ok: bool,
        ms: u64,
        #[serde(skip_serializing_if = "Option::is_none")]
        status: Option<String>,
        #[serde(skip_serializing_if = "Option::is_none")]
        action_id: Option<String>,
    },
    ApprovalPending {
        action_id: String,
        tool: String,
        server: String,
    },
    Suggestions {
        texts: Vec<String>,
    },
    Done {
        tokens: TokenUsage,
        iterations: u32,
        correlation_id: String,
    },
    /// Terminal failure event. `message` is intentionally opaque — full
    /// error context goes to `tracing::error!` server-side with the same
    /// `correlation_id`, mirroring `AppError::into_response`'s v1.1
    /// hardening so SSE doesn't regress the opaque-error invariant.
    Error {
        message: String,
        correlation_id: String,
    },
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

    /// Resolve the MCP-server label that owns `tool_name`, if any.
    ///
    /// Used by `ActionableMode::dispatch_write_tool` to populate
    /// `PendingActionDraft.server_label` so the audit envelope can name the
    /// downstream server without dereferencing executor internals.
    /// Default returns `None` so test fakes that don't care can skip.
    fn server_label_for(&self, _tool_name: &str) -> Option<String> {
        None
    }
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
    messages: Vec<Message>,
    system_prompt: &str,
    llm: &dyn LlmClient,
    mcp: &dyn McpExecutor,
    tool_specs: &[ToolSpec],
    max_iterations: usize,
    max_tokens: u32,
) -> anyhow::Result<RunOutcome> {
    // Backwards-compat shim — legacy callers (Teams scheduled trigger,
    // --terminal-mode, current /chat handler) get ReadOnly behaviour.
    // PR-4 routes /chat through `run_with_messages_in_mode` with the
    // JWT-derived AgentMode + DispatchContext.
    let mode = AgentMode::ReadOnly(ReadOnlyMode);
    let ctx = DispatchContext::default();
    run_with_messages_in_mode(
        messages,
        system_prompt,
        llm,
        mcp,
        tool_specs,
        max_iterations,
        max_tokens,
        &mode,
        &ctx,
    )
    .await
}

/// Mode-aware tool-loop. Identical to `run_with_messages` except every
/// tool dispatch routes through the [`AgentMode`]:
/// - `ReadOnly` + read-tool → executor passthrough
/// - `ReadOnly` + write-tool → synthetic blocked-read-only result
/// - `Actionable` + read-tool → executor passthrough
/// - `Actionable` + write-tool → `ApprovalFlow::propose` + synthetic
///   `ACTION_PROPOSED:` marker
///
/// Compile-time gate: the (`ReadOnly`, write) arm calls
/// `build_blocked_read_only_result` directly because `ReadOnlyMode` has
/// no `dispatch_write_tool` method. The match's exhaustiveness over
/// `(AgentMode, bool)` makes the gate visible at the call-site.
#[allow(clippy::too_many_arguments)]
pub async fn run_with_messages_in_mode(
    mut messages: Vec<Message>,
    system_prompt: &str,
    llm: &dyn LlmClient,
    mcp: &dyn McpExecutor,
    tool_specs: &[ToolSpec],
    max_iterations: usize,
    max_tokens: u32,
    mode: &AgentMode,
    ctx: &DispatchContext,
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
                    let requires_approval = tool_specs
                        .iter()
                        .find(|s| s.name == name)
                        .map(|s| s.requires_approval)
                        .unwrap_or(false);
                    let (result, trace) = match (mode, requires_approval) {
                        (AgentMode::ReadOnly(_), true) => {
                            let server_label = mcp
                                .server_label_for(&name)
                                .unwrap_or_else(|| "<unknown>".into());
                            build_blocked_read_only_result(&name, &server_label)
                        }
                        (AgentMode::ReadOnly(m), false) => {
                            m.dispatch_read_tool(mcp, &name, input).await?
                        }
                        (AgentMode::Actionable(m), false) => {
                            m.dispatch_read_tool(mcp, &name, input).await?
                        }
                        (AgentMode::Actionable(m), true) => {
                            m.dispatch_write_tool(mcp, ctx, &name, input).await?
                        }
                    };
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

/// Streaming counterpart of [`run_with_messages_in_mode`]. Same tool-loop
/// semantics, but every LLM call goes through `stream_chat` and per-delta
/// progress is forwarded on `tx` so the HTTP layer can stream SSE events
/// to the client without buffering the full answer first.
///
/// Returns `Ok(())` on natural completion (terminal `Done` event already
/// sent on `tx`); `Err` only on fundamental failures the HTTP layer needs
/// to surface as a 5xx. Send-errors on `tx` (client disconnect) are
/// logged and ignored — the orchestrator runs the iteration to completion
/// so partial state isn't left dangling.
///
/// On approval-pending: emits `ApprovalPending` + terminal `Done` and
/// returns `Ok(())`. Client restarts a new `/chat/stream` after
/// `/chat/approve`. This matches R2's request/response approval semantics
/// — orchestrator-state is not persisted across HTTP calls.
#[allow(clippy::too_many_arguments)]
#[allow(dead_code)] // reachable via /chat/stream HTTP route in PR4
pub async fn run_with_messages_in_mode_streaming(
    mut messages: Vec<Message>,
    system_prompt: &str,
    llm: &dyn LlmClient,
    mcp: &dyn McpExecutor,
    tool_specs: &[ToolSpec],
    max_iterations: usize,
    max_tokens: u32,
    mode: &AgentMode,
    ctx: &DispatchContext,
    tx: mpsc::Sender<ProgressEvent>,
) -> anyhow::Result<()> {
    let mut tokens = TokenUsage::default();

    for iteration in 0..max_iterations {
        // Drain one streaming round. The provider's translator (PR2) emits
        // exactly one terminal `Done` event per call carrying the
        // reassembled `full_content` (thinking signatures byte-identical,
        // tool_use input JSON parsed once), `stop_reason`, and `usage`.
        let mut stream = llm
            .stream_chat(system_prompt, &messages, tool_specs, max_tokens)
            .await?;

        let mut full_content: Vec<ContentBlock> = Vec::new();
        let mut stop_reason = StopReason::Other("stream_ended_without_done".into());

        while let Some(event_result) = stream.next().await {
            match event_result {
                Ok(StreamEvent::TextDelta(text)) => {
                    let _ = tx.send(ProgressEvent::TextChunk { text }).await;
                }
                Ok(StreamEvent::ThinkingDelta(text)) => {
                    let _ = tx.send(ProgressEvent::Thinking { text }).await;
                }
                Ok(StreamEvent::ToolUseStart { .. })
                | Ok(StreamEvent::ToolUseDelta { .. })
                | Ok(StreamEvent::ToolUseStop { .. }) => {
                    // Provider-level deltas — the orchestrator emits the
                    // higher-level ToolCallStarted from the aggregated
                    // tool_use blocks in `full_content` instead.
                }
                Ok(StreamEvent::Done {
                    stop_reason: sr,
                    usage,
                    full_content: fc,
                }) => {
                    if let Some(u) = usage {
                        tokens.add(&u);
                    }
                    stop_reason = sr;
                    full_content = fc;
                    break;
                }
                Err(e) => {
                    tracing::error!(
                        correlation_id = %ctx.correlation_id,
                        error = ?e,
                        "streaming error from llm.stream_chat",
                    );
                    let _ = tx
                        .send(ProgressEvent::Error {
                            message: "internal error".into(),
                            correlation_id: ctx.correlation_id.clone(),
                        })
                        .await;
                    return Err(e);
                }
            }
        }

        match stop_reason {
            StopReason::EndTurn => {
                maybe_emit_suggestions(&full_content, &tx, llm, &ctx.correlation_id).await;
                let _ = tx
                    .send(ProgressEvent::Done {
                        tokens,
                        iterations: iteration as u32 + 1,
                        correlation_id: ctx.correlation_id.clone(),
                    })
                    .await;
                return Ok(());
            }
            StopReason::MaxTokens => {
                tracing::warn!(
                    iteration,
                    "anthropic max_tokens hit; closing stream with partial response"
                );
                let _ = tx
                    .send(ProgressEvent::Done {
                        tokens,
                        iterations: iteration as u32 + 1,
                        correlation_id: ctx.correlation_id.clone(),
                    })
                    .await;
                return Ok(());
            }
            StopReason::ToolUse => {
                messages.push(Message {
                    role: Role::Assistant,
                    content: full_content.clone(),
                });

                let tool_calls: Vec<(String, String, Value)> = full_content
                    .into_iter()
                    .filter_map(|b| match b {
                        ContentBlock::ToolUse { id, name, input } => Some((id, name, input)),
                        _ => None,
                    })
                    .collect();
                if tool_calls.is_empty() {
                    let msg = "stop_reason=tool_use but no tool_use blocks found".to_string();
                    let _ = tx
                        .send(ProgressEvent::Error {
                            message: msg.clone(),
                            correlation_id: ctx.correlation_id.clone(),
                        })
                        .await;
                    bail!(msg);
                }

                // Surface each upcoming dispatch before we kick off the
                // parallel join — the UI can render "Calling tool: X" in
                // the order the assistant requested.
                for (_, name, _) in &tool_calls {
                    let _ = tx
                        .send(ProgressEvent::ToolCallStarted {
                            name: name.clone(),
                            server: mcp.server_label_for(name),
                        })
                        .await;
                }

                let tool_futs = tool_calls.into_iter().map(|(id, name, input)| async move {
                    let requires_approval = tool_specs
                        .iter()
                        .find(|s| s.name == name)
                        .map(|s| s.requires_approval)
                        .unwrap_or(false);
                    let (result, trace) = match (mode, requires_approval) {
                        (AgentMode::ReadOnly(_), true) => {
                            let server_label = mcp
                                .server_label_for(&name)
                                .unwrap_or_else(|| "<unknown>".into());
                            build_blocked_read_only_result(&name, &server_label)
                        }
                        (AgentMode::ReadOnly(m), false) => {
                            m.dispatch_read_tool(mcp, &name, input).await?
                        }
                        (AgentMode::Actionable(m), false) => {
                            m.dispatch_read_tool(mcp, &name, input).await?
                        }
                        (AgentMode::Actionable(m), true) => {
                            m.dispatch_write_tool(mcp, ctx, &name, input).await?
                        }
                    };
                    let block = ContentBlock::ToolResult {
                        tool_use_id: id,
                        content: result,
                        is_error: !trace.ok,
                    };
                    Ok::<(ContentBlock, ToolCallTrace), anyhow::Error>((block, trace))
                });
                let outputs: Vec<(ContentBlock, ToolCallTrace)> = try_join_all(tool_futs).await?;

                let mut pending: Option<(String, String, String)> = None;
                for (_, trace) in &outputs {
                    let _ = tx
                        .send(ProgressEvent::ToolCallCompleted {
                            name: trace.tool.clone(),
                            ok: trace.ok,
                            ms: trace.ms,
                            status: trace.status.clone(),
                            action_id: trace.action_id.clone(),
                        })
                        .await;
                    if trace.status.as_deref() == Some("pending")
                        && pending.is_none()
                        && let Some(action_id) = &trace.action_id
                    {
                        pending =
                            Some((action_id.clone(), trace.tool.clone(), trace.server.clone()));
                    }
                }

                if let Some((action_id, tool, server)) = pending {
                    let _ = tx
                        .send(ProgressEvent::ApprovalPending {
                            action_id,
                            tool,
                            server,
                        })
                        .await;
                    let _ = tx
                        .send(ProgressEvent::Done {
                            tokens,
                            iterations: iteration as u32 + 1,
                            correlation_id: ctx.correlation_id.clone(),
                        })
                        .await;
                    return Ok(());
                }

                let (results, _traces): (Vec<_>, Vec<_>) = outputs.into_iter().unzip();
                messages.push(Message {
                    role: Role::User,
                    content: results,
                });
            }
            StopReason::Other(ref s) if s == "premature_close" => {
                // Anthropic closed the connection without `message_stop`.
                // The provider-side translator has already finalised any
                // partially-built blocks and the user already saw the text
                // deltas. Close the stream gracefully with a terminal Done
                // — no iteration, no follow-up call (which would 400 on
                // malformed signatures anyway).
                tracing::warn!(
                    correlation_id = %ctx.correlation_id,
                    "anthropic stream closed prematurely; flushing partial response",
                );
                let _ = tx
                    .send(ProgressEvent::Done {
                        tokens,
                        iterations: iteration as u32 + 1,
                        correlation_id: ctx.correlation_id.clone(),
                    })
                    .await;
                return Ok(());
            }
            StopReason::Other(s) => {
                let msg = format!("unexpected stop_reason: {s}");
                let _ = tx
                    .send(ProgressEvent::Error {
                        message: msg.clone(),
                        correlation_id: ctx.correlation_id.clone(),
                    })
                    .await;
                bail!(msg);
            }
        }
    }

    let msg = format!("tool-call loop exceeded {max_iterations} iterations");
    let _ = tx
        .send(ProgressEvent::Error {
            message: msg.clone(),
            correlation_id: ctx.correlation_id.clone(),
        })
        .await;
    bail!(msg);
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

#[derive(Deserialize)]
struct SuggestionsPayload {
    texts: Vec<String>,
}

const SUGGESTIONS_MAX_TOKENS: u32 = 256;
const SUGGESTIONS_TIMEOUT_SECS: u64 = 15;
const SUGGESTIONS_MIN_CHARS: usize = 5;
const SUGGESTIONS_MAX_CHARS: usize = 80;

/// Returns `Vec::new()` on any failure so the terminal `Done` event is never
/// delayed by a stuck inference and the caller can skip the SSE frame.
pub(crate) async fn generate_suggestions(
    llm: &dyn LlmClient,
    final_answer: &str,
    correlation_id: &str,
) -> Vec<String> {
    if final_answer.trim().is_empty() {
        return Vec::new();
    }
    let safe_answer = final_answer.replace("</UNTRUSTED>", "</UNTRUSTED_>");
    let user_text = format!("<UNTRUSTED>{safe_answer}</UNTRUSTED>\n\nGenereer 3 vervolgvragen.");
    let messages = vec![Message {
        role: Role::User,
        content: vec![ContentBlock::Text { text: user_text }],
    }];
    let outcome = tokio::time::timeout(
        Duration::from_secs(SUGGESTIONS_TIMEOUT_SECS),
        llm.chat(
            SUGGESTIONS_SYSTEM_PROMPT,
            &messages,
            &[],
            SUGGESTIONS_MAX_TOKENS,
        ),
    )
    .await;
    let response = match outcome {
        Ok(Ok(r)) => r,
        Ok(Err(e)) => {
            tracing::warn!(%correlation_id, error = ?e, "suggestions llm.chat failed");
            return Vec::new();
        }
        Err(_) => {
            tracing::warn!(%correlation_id, "suggestions llm.chat timed out");
            return Vec::new();
        }
    };
    let raw = collect_text(&response.content);
    let payload_json = strip_outer_code_fence(&raw);
    let payload: SuggestionsPayload = match serde_json::from_str(payload_json) {
        Ok(p) => p,
        Err(e) => {
            tracing::warn!(%correlation_id, error = ?e, "suggestions JSON parse failed");
            return Vec::new();
        }
    };
    if payload.texts.len() != 3 {
        tracing::warn!(%correlation_id, len = payload.texts.len(), "suggestions wrong count");
        return Vec::new();
    }
    for t in &payload.texts {
        let trimmed = t.trim();
        let n = trimmed.chars().count();
        let has_disallowed = trimmed.chars().any(is_disallowed_suggestion_char);
        if trimmed.is_empty()
            || !(SUGGESTIONS_MIN_CHARS..=SUGGESTIONS_MAX_CHARS).contains(&n)
            || has_disallowed
        {
            tracing::warn!(
                %correlation_id,
                chars = n,
                disallowed = has_disallowed,
                "suggestion failed validation"
            );
            return Vec::new();
        }
    }
    payload
        .texts
        .into_iter()
        .map(|t| t.trim().to_string())
        .collect()
}

/// Frontend `textContent` defangs HTML but cannot stop RTL-override or ZWJ
/// from confusing the visual chip layout.
fn is_disallowed_suggestion_char(c: char) -> bool {
    c.is_control()
        || ('\u{200B}'..='\u{200F}').contains(&c)
        || ('\u{202A}'..='\u{202E}').contains(&c)
}

async fn maybe_emit_suggestions(
    full_content: &[ContentBlock],
    tx: &mpsc::Sender<ProgressEvent>,
    llm: &dyn LlmClient,
    correlation_id: &str,
) {
    let answer_text = collect_text(full_content);
    if answer_text.is_empty() {
        return;
    }
    let texts = generate_suggestions(llm, &answer_text, correlation_id).await;
    if !texts.is_empty() {
        let _ = tx.send(ProgressEvent::Suggestions { texts }).await;
    }
}

fn strip_outer_code_fence(s: &str) -> &str {
    let t = s.trim();
    if !t.starts_with("```") {
        return t;
    }
    let after_open = match t.find('\n') {
        Some(idx) => &t[idx + 1..],
        None => return t,
    };
    match after_open.rfind("```") {
        Some(idx) => after_open[..idx].trim(),
        None => t,
    }
}

#[cfg(test)]
mod tests;
