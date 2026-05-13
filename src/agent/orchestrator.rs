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
const SUGGESTIONS_MAX_CHARS: usize = 100;

/// Best-effort follow-up generator. Wraps a single `LlmClient::chat` in a
/// hard timeout so a stuck inference cannot delay the terminal `Done`
/// event. Returns an empty vector on any failure — caller treats that as
/// "skip Suggestions event" (graceful degradation).
async fn generate_suggestions(
    llm: &dyn LlmClient,
    final_answer: &str,
    correlation_id: &str,
) -> Vec<String> {
    let user_text = format!("<UNTRUSTED>{final_answer}</UNTRUSTED>\n\nGenereer 3 vervolgvragen.");
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
        if trimmed.is_empty() || !(SUGGESTIONS_MIN_CHARS..=SUGGESTIONS_MAX_CHARS).contains(&n) {
            tracing::warn!(%correlation_id, chars = n, "suggestion failed length check");
            return Vec::new();
        }
    }
    payload.texts
}

/// Generate + emit follow-up suggestions for `full_content` if the
/// assistant produced any user-visible text. No-op when answer-text is
/// empty (e.g. tool-only turns) or when the LLM call fails — keeps the
/// terminal `Done` event semantics intact under degradation.
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

/// Strip an optional triple-backtick fence around a JSON payload. Defensive
/// for the case where the model wraps its structured output despite the
/// system-prompt instruction not to.
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
mod tests {
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
                    status: None,
                    action_id: None,
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

    #[tokio::test]
    async fn generate_suggestions_happy_path_returns_three_strings() {
        let llm = MockLlmClient::new(vec![text_response(
            r#"{"texts":["Toon recente registraties","Welke bedrijven zijn er?","Status systemen?"]}"#,
        )]);
        let got = generate_suggestions(&llm, "Er zijn 42 actieve contacten.", "cid-1").await;
        assert_eq!(got.len(), 3);
        assert_eq!(got[0], "Toon recente registraties");
        assert_eq!(got[2], "Status systemen?");
    }

    #[tokio::test]
    async fn generate_suggestions_malformed_json_returns_empty() {
        let llm = MockLlmClient::new(vec![text_response("dit is geen json")]);
        let got = generate_suggestions(&llm, "antwoord", "cid-2").await;
        assert!(got.is_empty());
    }

    #[tokio::test]
    async fn generate_suggestions_wrong_count_returns_empty() {
        let llm = MockLlmClient::new(vec![text_response(
            r#"{"texts":["Eerste vraag?","Tweede vraag?"]}"#,
        )]);
        let got = generate_suggestions(&llm, "antwoord", "cid-3").await;
        assert!(got.is_empty());
    }

    #[tokio::test]
    async fn generate_suggestions_oversized_string_returns_empty() {
        let big = "x".repeat(101);
        let payload = format!(r#"{{"texts":["Eerste vraag?","Tweede vraag?","{big}"]}}"#);
        let llm = MockLlmClient::new(vec![text_response(&payload)]);
        let got = generate_suggestions(&llm, "antwoord", "cid-4").await;
        assert!(got.is_empty());
    }

    #[tokio::test]
    async fn generate_suggestions_strips_code_fence_and_parses() {
        let fenced =
            "```json\n{\"texts\":[\"Eerste vraag?\",\"Tweede vraag?\",\"Derde vraag?\"]}\n```";
        let llm = MockLlmClient::new(vec![text_response(fenced)]);
        let got = generate_suggestions(&llm, "antwoord", "cid-5").await;
        assert_eq!(got.len(), 3);
        assert_eq!(got[0], "Eerste vraag?");
    }

    #[tokio::test]
    async fn streaming_emits_suggestions_before_done_on_endturn() {
        let llm = MockLlmClient::new(vec![text_response(
            r#"{"texts":["Toon contacten","Welke bedrijven?","Status check"]}"#,
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
            "no Suggestions event when LLM call fails",
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
}
