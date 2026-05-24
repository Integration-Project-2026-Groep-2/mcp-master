use std::sync::Arc;

use anyhow::{Context, Result, bail};
use axum::{
    Json, Router,
    extract::{DefaultBodyLimit, Path, State},
    http::{HeaderValue, Method, StatusCode, header::CONTENT_TYPE},
    response::{
        IntoResponse, Response,
        sse::{Event as SseEvent, KeepAlive, Sse},
    },
    routing::{delete, get, post},
};
use chrono::{NaiveTime, Timelike, Utc};
use serde::{Deserialize, Serialize};
use tokio::sync::{mpsc, watch};
use tower_http::{cors::CorsLayer, timeout::TimeoutLayer, trace::TraceLayer};

use crate::{
    agent::llm::{ContentBlock, Message, Role, TokenUsage, ToolSpec, anthropic::AnthropicClient},
    agent::orchestrator::{self, McpExecutor, ProgressEvent, ToolCallTrace},
    agent::prompts::{ANALYZE_CONTROLROOM_PROMPT, SETUP_PROMPT},
    gateway::approval::types::ApprovalError,
    mcp::McpPool,
    memory::{MemoryInteraction, MemoryService, MemorySource, SqliteMemory},
    rabbitmq::{config::RabbitMqConfig, consumer as rabbitmq_consumer, publisher::Publisher},
    teams::{TeamsConfig, publish_to_teams},
};

const READ_ONLY_MAX_ITERATIONS: usize = 10;
const ACTIONABLE_MAX_ITERATIONS: usize = 20;
const MAX_TOKENS: u32 = 8192;

/// Tool-loop budget per mode: read-only Q&A converges in a few rounds, while the
/// Actionable investigate-and-fix flow is inherently multi-step and needs
/// headroom rather than bailing mid-investigation.
fn max_iterations_for(mode: &crate::agent::modes::AgentMode) -> usize {
    use crate::agent::modes::AgentMode;
    match mode {
        AgentMode::ReadOnly(_) => READ_ONLY_MAX_ITERATIONS,
        AgentMode::Actionable(_) => ACTIONABLE_MAX_ITERATIONS,
    }
}

// DoS guards. Total request body capped at the axum layer; in addition the
// per-turn caps in `ChatRequest::into_messages` reject pathological shapes
// (1000-turn arrays, single 64KB content blocks) that fit within the body
// budget but would still amplify Anthropic input tokens 10x via the
// orchestrator tool-loop.
const MAX_BODY_BYTES: usize = 64 * 1024;
const MAX_TURNS: usize = 40;
const MAX_CONTENT_BYTES_PER_TURN: usize = 8192;
const REQUEST_TIMEOUT_SECONDS: u64 = 240;

// Anthropic Tier 2 standaard is 50 req/min op /v1/messages. Onze gemiddelde
// /chat is ~10s end-to-end → 8 parallel × 10s = 48 req/min — net binnen budget.
// Burst boven 8 → tower's interne queue buffert; TimeoutLayer firet als de
// queue te lang blijft staan.
const MAX_CONCURRENT_CHAT: usize = 8;

// Background /fix-flow runs detach from the request concurrency slot (it frees
// at the 202), so they get their own cap to bound concurrent agent token-burn.
const MAX_CONCURRENT_FIX_FLOW: usize = 4;

// Wallclock cap for the entire `/chat/stream` run. TimeoutLayer is
// deliberately not applied to the streaming route (would kill long-but-
// legitimate tool-cascades) so this is the hard bound on a concurrency
// slot. 600s comfortably exceeds the longest observed multi-MCP cascade.
const STREAM_DEADLINE_SECS: u64 = 600;

fn cloudflare_pad_comment() -> &'static str {
    use std::sync::OnceLock;
    static PAD: OnceLock<String> = OnceLock::new();
    PAD.get_or_init(|| "cf-stream-pad ".repeat(293))
}

pub struct AppState {
    pub llm: AnthropicClient,
    pub pool: McpPool,
    pub tool_specs: Vec<ToolSpec>,
    pub publisher: Option<Arc<Publisher>>,
    pub cache: Option<Arc<SqliteMemory>>,
    pub memory: Option<Arc<MemoryService>>,
    /// Prometheus render handle served by the `/metrics` endpoint.
    pub metrics_handle: crate::metrics::Handle,
    /// Wired into chat() in commit 2 (mode dispatch) and chat_approve/reject
    /// in commits 3+4. Held as Arc so the cleanup task holds Arc<ApprovalStore>
    /// (not Arc<AppState>) — keeps `Arc::try_unwrap` clean at shutdown.
    #[allow(dead_code)]
    pub approval_flow: Arc<crate::gateway::approval::flow::ApprovalFlow>,
    pub fix_flow_limit: Arc<tokio::sync::Semaphore>,
    /// Live last-seen-per-service map fed by the heartbeat consumer, read by
    /// `GET /status`.
    pub heartbeat_state: Arc<crate::heartbeat::HeartbeatState>,
}

/// Read `CHAT_APPROVAL_TTL_SECONDS` env-var; fall back to 900s (15min) on
/// missing or unparseable values.
fn approval_ttl() -> std::time::Duration {
    const DEFAULT_SECS: u64 = 900;
    match std::env::var("CHAT_APPROVAL_TTL_SECONDS") {
        Ok(raw) => match raw.trim().parse::<u64>() {
            Ok(secs) if secs > 0 => std::time::Duration::from_secs(secs),
            _ => {
                tracing::warn!(
                    raw = %raw,
                    "CHAT_APPROVAL_TTL_SECONDS unparseable — falling back to {DEFAULT_SECS}s"
                );
                std::time::Duration::from_secs(DEFAULT_SECS)
            }
        },
        Err(_) => std::time::Duration::from_secs(DEFAULT_SECS),
    }
}

/// Read `CHAT_STREAM_KEEPALIVE_SECONDS`; default 10s, clamped to 3..=60. The
/// keep-alive comment must fire faster than any proxy idle-timeout so the SSE
/// connection is not torn down during silent tool calls.
fn keepalive_secs() -> u64 {
    const DEFAULT_SECS: u64 = 10;
    const MIN_SECS: u64 = 3;
    const MAX_SECS: u64 = 60;
    match std::env::var("CHAT_STREAM_KEEPALIVE_SECONDS") {
        Ok(raw) => match raw.trim().parse::<u64>() {
            Ok(secs) => secs.clamp(MIN_SECS, MAX_SECS),
            Err(_) => {
                tracing::warn!(
                    raw = %raw,
                    "CHAT_STREAM_KEEPALIVE_SECONDS unparseable — falling back to {DEFAULT_SECS}s"
                );
                DEFAULT_SECS
            }
        },
        Err(_) => DEFAULT_SECS,
    }
}

/// Append the canonical service->repo coordinates to the system prompt when the
/// mode can use write tools, so the agent passes explicit owner/repo/base to
/// GitHub write tools instead of guessing slugs.
fn system_prompt_with_hints(base: String, mode: &crate::agent::modes::AgentMode) -> String {
    use crate::agent::modes::Mode;
    if mode.allows_write_tools() {
        format!("{base}{}", crate::agent::repo_map::repo_hints_prompt())
    } else {
        base
    }
}

/// Wire-level role for one chat turn. Strict-lowercase to match Anthropic's
/// convention and the JS-side string literals in jarvis_chat.
#[derive(Debug, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "lowercase")]
pub enum ChatRole {
    User,
    Assistant,
}

/// One conversation turn as the client sends it. `content` is plain string —
/// thinking/tool_use blocks from previous rounds are intentionally not echoed
/// by clients; mcp-master generates fresh ones in this call.
#[derive(Debug, Deserialize)]
pub struct ChatTurn {
    pub role: ChatRole,
    pub content: String,
}

/// Body shape for `POST /chat`. Accepts two mutually-exclusive shapes:
/// - Legacy single-shot: `{"prompt": "..."}` — used by Teams scheduled trigger
/// - Multi-turn:        `{"messages": [{"role": ..., "content": ...}, ...]}`
///   — used by jarvis_chat client-side history flow
///
/// Validation lives in `into_messages()`, not in `serde` derive, so the error
/// path stays a single 400 with an explanatory body instead of serde's
/// auto-generated tagged-enum errors.
#[derive(Debug, Deserialize)]
pub struct ChatRequest {
    pub prompt: Option<String>,
    pub messages: Option<Vec<ChatTurn>>,
}

impl ChatRequest {
    /// Validate body shape and convert to internal `Vec<Message>`. Returns
    /// the seed conversation that the orchestrator runs the tool-loop on top
    /// of. Last turn must be `Role::User` — that's the question we answer.
    pub fn into_messages(self) -> Result<Vec<Message>, &'static str> {
        match (self.prompt, self.messages) {
            (Some(_), Some(_)) => Err("provide either 'prompt' or 'messages', not both"),
            (None, None) => Err("missing 'prompt' or 'messages'"),
            (Some(p), None) => {
                let trimmed = p.trim();
                if trimmed.is_empty() {
                    return Err("prompt is empty");
                }
                if trimmed.len() > MAX_CONTENT_BYTES_PER_TURN {
                    return Err("prompt exceeds maximum length");
                }
                Ok(vec![Message {
                    role: Role::User,
                    content: vec![ContentBlock::Text {
                        text: trimmed.to_string(),
                    }],
                }])
            }
            (None, Some(turns)) => {
                if turns.is_empty() {
                    return Err("messages array is empty");
                }
                if turns.len() > MAX_TURNS {
                    return Err("messages array exceeds maximum length");
                }
                if turns
                    .iter()
                    .any(|t| t.content.len() > MAX_CONTENT_BYTES_PER_TURN)
                {
                    return Err("message content exceeds maximum length per turn");
                }
                // Defence against forged tool-use blocks smuggled in via the
                // text content of an assistant turn. The wire-format only
                // accepts plain strings per turn, but a string that LOOKS like
                // a tool_use marker can confuse the model into thinking it
                // already has tool results. Reject any such content.
                if turns.iter().any(|t| contains_tool_marker(&t.content)) {
                    return Err("messages content contains forbidden tool-use markers");
                }
                if !matches!(turns.last().unwrap().role, ChatRole::User) {
                    return Err("last message must have role=user");
                }
                Ok(turns
                    .into_iter()
                    .map(|t| Message {
                        role: match t.role {
                            ChatRole::User => Role::User,
                            ChatRole::Assistant => Role::Assistant,
                        },
                        content: vec![ContentBlock::Text { text: t.content }],
                    })
                    .collect())
            }
        }
    }
}

/// Wire shape for `POST /chat` 2xx responses. Extra fields beyond `answer`
/// are additive — clients destructuring only `answer` continue working.
///
/// `tool_trace` carries one entry per MCP tool dispatched, in dispatch order.
/// `tokens` aggregates Anthropic usage across the full tool-loop. The
/// `correlation_id` ties this response to matching `tool_called` and
/// `chat_completed` events on the AMQP audit feed. Errors stay opaque
/// (see `AppError::into_response`) — no trace leaks via error responses.
#[derive(Debug, Serialize)]
pub struct ChatResponse {
    pub answer: String,
    pub cached: bool,
    pub tool_trace: Vec<ToolCallTrace>,
    pub tokens: TokenUsage,
    pub iterations: u32,
    pub correlation_id: String,
    #[serde(skip_serializing_if = "Vec::is_empty")]
    pub suggestions: Vec<String>,
}

/// Substrings that suggest the client tried to forge a structured tool-use
/// block inside plain content. Matched case-insensitively. Anthropic's
/// content-block taxonomy uses these strings; if a user echoes them, treat
/// it as adversarial.
const TOOL_MARKERS: &[&str] = &["tool_use_id", "<tool_use", "</tool_use", "<tool_result"];

fn contains_tool_marker(content: &str) -> bool {
    let lower = content.to_ascii_lowercase();
    TOOL_MARKERS.iter().any(|m| lower.contains(m))
}

pub struct AppError(pub anyhow::Error);

impl<E> From<E> for AppError
where
    E: Into<anyhow::Error>,
{
    fn from(e: E) -> Self {
        Self(e.into())
    }
}

impl IntoResponse for AppError {
    fn into_response(self) -> Response {
        // Full anyhow context-chain stays in stdout logs (which only ops see).
        // The HTTP body returns an opaque error + correlation_id so attackers
        // can't fish for RABBITMQ_URL credentials, env-var names, or internal
        // file paths via the `{:#}` formatter on the cause-chain.
        let correlation_id = uuid::Uuid::new_v4();
        tracing::error!(
            correlation_id = %correlation_id,
            "/chat handler error: {:#}",
            self.0,
        );
        let body = Json(serde_json::json!({
            "error": "internal error",
            "correlation_id": correlation_id.to_string(),
        }));
        (StatusCode::INTERNAL_SERVER_ERROR, body).into_response()
    }
}

const CACHE_SWEEP_INTERVAL_SECONDS: u64 = 60;

async fn run_cache_sweeper(cache: Arc<SqliteMemory>, mut shutdown_rx: watch::Receiver<bool>) {
    let mut ticker =
        tokio::time::interval(std::time::Duration::from_secs(CACHE_SWEEP_INTERVAL_SECONDS));
    ticker.tick().await; // skip immediate first fire

    loop {
        tokio::select! {
            biased;
            _ = shutdown_rx.changed() => {
                if *shutdown_rx.borrow() {
                    tracing::info!("response cache sweeper shutting down");
                    return;
                }
            }
            _ = ticker.tick() => {
                let cache = Arc::clone(&cache);
                match tokio::task::spawn_blocking(move || cache.purge_expired()).await {
                    Ok(Ok(n)) if n > 0 => tracing::debug!(count = n, "swept expired response cache"),
                    Ok(Err(e)) => tracing::warn!("response cache sweep failed: {e:#}"),
                    Err(e) => tracing::warn!("response cache sweep task panicked: {e:#}"),
                    _ => {}
                }
            }
        }
    }
}

async fn health() -> &'static str {
    "ok"
}

/// Live per-service status derived from the heartbeat tap. Read-only +
/// unauthenticated like `/health`/`/metrics` — non-sensitive, behind the Drupal
/// proxy + network isolation. Services never seen are absent (Frontend renders
/// them "unknown").
async fn status(State(state): State<Arc<AppState>>) -> impl IntoResponse {
    let now = chrono::Utc::now();
    Json(serde_json::json!({
        "services": crate::heartbeat::snapshot(&state.heartbeat_state, now),
        "checked_at": now.to_rfc3339(),
    }))
}

async fn metrics(State(state): State<Arc<AppState>>) -> impl IntoResponse {
    // 0.0.4 is the Prometheus text-exposition version tag scrapers expect.
    (
        [(CONTENT_TYPE, "text/plain; version=0.0.4")],
        state.metrics_handle.render(),
    )
}

/// Keys over the whole conversation, not just the last turn, so two chats
/// ending in the same message don't collide. None when nothing is cacheable.
fn conversation_cache_key(messages: &[Message]) -> Option<String> {
    use std::fmt::Write;
    let mut buf = String::new();
    let mut has_text = false;
    for message in messages {
        let role = match message.role {
            Role::User => "user",
            Role::Assistant => "assistant",
        };
        for block in &message.content {
            if let ContentBlock::Text { text } = block {
                let _ = writeln!(buf, "{role}: {text}");
                if !text.trim().is_empty() {
                    has_text = true;
                }
            }
        }
    }
    has_text.then_some(buf)
}

async fn chat(
    scope: crate::gateway::auth::AuthScope,
    State(state): State<Arc<AppState>>,
    headers: axum::http::HeaderMap,
    Json(req): Json<ChatRequest>,
) -> Result<Json<ChatResponse>, Response> {
    // Generated at handler entry so success-path responses, AMQP audit
    // events, and any tracing spans share the same id. Error-path uses its
    // own UUID via AppError — consolidation is a v1.5 follow-up.
    let correlation_id = uuid::Uuid::new_v4().to_string();

    let messages = req.into_messages().map_err(|e| {
        let body = Json(serde_json::json!({ "error": e }));
        (StatusCode::BAD_REQUEST, body).into_response()
    })?;
    let prompt = match messages.last().map(|m| &m.content[..]) {
        Some([ContentBlock::Text { text }]) => text.clone(),
        _ => String::new(),
    };
    let conversation_length = messages.len();
    let cache_key = conversation_cache_key(&messages);
    // Don't log the prompt body — it routinely contains GDPR-flagged CRM data
    // (names, emails, BTW numbers) and occasional accidental secrets pasted by
    // users. The full prompt+answer still rides on the RabbitMQ
    // `chat_completed` audit-event for downstream analytics.
    tracing::info!(
        correlation_id = %correlation_id,
        prompt_length = prompt.len(),
        conversation_length,
        scope = ?scope,
        "/chat received"
    );

    let mode = match scope {
        crate::gateway::auth::AuthScope::Read => {
            crate::agent::modes::AgentMode::ReadOnly(crate::agent::modes::ReadOnlyMode)
        }
        crate::gateway::auth::AuthScope::ReadAndAct => crate::agent::modes::AgentMode::Actionable(
            crate::agent::modes::ActionableMode::new(state.approval_flow.clone()),
        ),
    };
    // Thread the JWT sub claim through so a write-tool proposal stores the
    // proposer's user_id on the PendingAction. flow.confirm rejects an
    // approve-call whose user_id doesn't match — empty stored vs non-empty
    // caller would 403 every approve. Empty fallback is fine for the
    // legacy-bearer / skip-warn read-only paths (no approval-flow downstream).
    let user_id = crate::gateway::auth::current_user_id(&headers).unwrap_or_default();
    let ctx = crate::agent::modes::DispatchContext {
        correlation_id: correlation_id.clone(),
        user_id,
        scope,
    };

    if let (Some(cache), Some(key)) = (state.cache.as_ref(), cache_key.as_ref()) {
        match cache.lookup_response(key) {
            Ok(Some(answer)) => {
                tracing::info!(correlation_id = %correlation_id, "chat cache hit");
                crate::metrics::record_chat("sync", "cache_hit", &TokenUsage::default());
                return Ok(Json(ChatResponse {
                    answer,
                    cached: true,
                    tool_trace: Vec::new(),
                    tokens: TokenUsage::default(),
                    iterations: 0,
                    correlation_id,
                    suggestions: Vec::new(),
                }));
            }
            Ok(None) => {}
            Err(e) => return Err(AppError(e).into_response()),
        }
    }

    let system_prompt = match state.memory.as_deref() {
        Some(memory) => memory
            .augment_system_prompt(SETUP_PROMPT, &messages, Some(ctx.user_id.as_str()))
            .await
            .unwrap_or_else(|e| {
                tracing::warn!("memory retrieval failed; falling back to base prompt: {e:#}");
                SETUP_PROMPT.to_string()
            }),
        None => SETUP_PROMPT.to_string(),
    };
    let system_prompt = system_prompt_with_hints(system_prompt, &mode);

    let started = std::time::Instant::now();
    let outcome = orchestrator::run_with_messages_in_mode(
        messages,
        &system_prompt,
        &state.llm,
        &state.pool,
        &state.tool_specs,
        max_iterations_for(&mode),
        MAX_TOKENS,
        &mode,
        &ctx,
    )
    .await
    .map_err(|e| {
        crate::metrics::record_chat("sync", "error", &TokenUsage::default());
        AppError(e).into_response()
    })?;
    let duration_ms = started.elapsed().as_millis() as u64;
    crate::metrics::record_chat("sync", "ok", &outcome.tokens);

    let suggestions = if chat_suggestions_enabled() {
        crate::agent::orchestrator::generate_suggestions(
            &state.llm,
            &outcome.answer,
            &state.tool_specs,
            &correlation_id,
        )
        .await
    } else {
        Vec::new()
    };

    if let Some(publisher) = &state.publisher {
        let payload = serde_json::json!({
            "correlation_id": correlation_id,
            "prompt": prompt,
            "answer": outcome.answer,
            "answer_length": outcome.answer.len(),
            "duration_ms": duration_ms,
            "conversation_length": conversation_length,
            "tool_trace": outcome.tool_trace,
            "tokens": outcome.tokens,
            "iterations": outcome.iterations,
        });
        if let Err(e) = publisher.publish_event("chat_completed", payload).await {
            tracing::warn!("failed to publish chat_completed event: {e:#}");
        }
    }

    if let Some(memory) = state.memory.as_deref()
        && let Err(e) = memory
            .remember_interaction(MemoryInteraction::new(
                "default",
                MemorySource::Chat,
                correlation_id.clone(),
                Some(ctx.user_id.clone()),
                &prompt,
                &outcome.answer,
            ))
            .await
    {
        tracing::warn!("memory ingestion failed: {e:#}");
    }

    // Skip tool-backed answers: they embed live data that goes stale within the TTL.
    if outcome.tool_trace.is_empty()
        && let (Some(cache), Some(key)) = (state.cache.as_ref(), cache_key.as_ref())
        && let Err(e) = cache.store_response(key, &outcome.answer)
    {
        tracing::warn!("response cache store failed: {e:#}");
    }

    Ok(Json(ChatResponse {
        answer: outcome.answer,
        cached: false,
        tool_trace: outcome.tool_trace,
        tokens: outcome.tokens,
        iterations: outcome.iterations,
        correlation_id,
        suggestions,
    }))
}

/// Body for `POST /chat/approve`. The `action_id` was returned in the
/// original `/chat` response's `tool_trace[i].action_id`; the client echoes
/// it back here to authorise dispatch of the proposed write-tool.
#[derive(Debug, Deserialize)]
pub struct ApproveBody {
    pub action_id: uuid::Uuid,
}

/// Map approval state-machine errors to HTTP statuses.
///
/// - `NotFound` → 404 (action id never existed or was swept by TTL cleanup)
/// - `AlreadyDecided` → 409 (idempotent retry of confirm/reject)
/// - `WrongUser` → 403 (action-id hijack attempt across users)
/// - `Expired` → 410 (TTL elapsed before user clicked Approve)
///
/// Bodies stay short — the action_id alone is enough breadcrumb; the
/// `ApprovalError` Display impls can surface PII (proposer/caller IDs).
fn approval_error_response(e: ApprovalError) -> Response {
    let (status, message) = match &e {
        ApprovalError::NotFound(_) => (StatusCode::NOT_FOUND, "action not found"),
        ApprovalError::AlreadyDecided(_) => (StatusCode::CONFLICT, "action already decided"),
        ApprovalError::WrongUser { .. } => (StatusCode::FORBIDDEN, "user mismatch"),
        ApprovalError::Expired(_) => (StatusCode::GONE, "action expired"),
    };
    let body = Json(serde_json::json!({ "error": message }));
    (status, body).into_response()
}

/// Trigger a background fix-flow for an incident. Returns 202 immediately; the
/// spawned Actionable run proposes a `request_changes_with_files` PR (a pending
/// action) and publishes `fix_proposed` on `ai.events`. Approval via /chat/approve.
async fn fix_flow(
    scope: crate::gateway::auth::AuthScope,
    State(state): State<Arc<AppState>>,
    axum::Extension(shutdown_rx): axum::Extension<watch::Receiver<bool>>,
    headers: axum::http::HeaderMap,
    Json(req): Json<crate::fix_flow::FixFlowRequest>,
) -> Response {
    if scope != crate::gateway::auth::AuthScope::ReadAndAct {
        return scope_required_response();
    }
    // The proposed write's PendingAction.user_id must equal the later
    // /chat/approve caller's JWT sub, or confirm() 403s — require a real sub.
    let user_id = match crate::gateway::auth::current_user_id(&headers) {
        Some(id) => id,
        None => return scope_required_response(),
    };
    if req.service.trim().is_empty() || req.suggested_action.trim().is_empty() {
        let body = Json(serde_json::json!({
            "error": "service and suggested_action are required"
        }));
        return (StatusCode::BAD_REQUEST, body).into_response();
    }
    let permit = match state.fix_flow_limit.clone().try_acquire_owned() {
        Ok(p) => p,
        Err(_) => {
            let body = Json(serde_json::json!({
                "error": "fix-flow capacity reached — retry shortly"
            }));
            return (StatusCode::TOO_MANY_REQUESTS, body).into_response();
        }
    };

    let correlation_id = req
        .correlation_id
        .clone()
        .unwrap_or_else(|| uuid::Uuid::new_v4().to_string());
    tracing::info!(
        correlation_id = %correlation_id,
        service = %req.service,
        "/fix-flow accepted — spawning background fix-flow"
    );

    let mode = crate::agent::modes::AgentMode::Actionable(
        crate::agent::modes::ActionableMode::new(state.approval_flow.clone()),
    );
    let max_iterations = max_iterations_for(&mode);
    let ctx = crate::agent::modes::DispatchContext {
        correlation_id: correlation_id.clone(),
        user_id,
        scope,
    };

    let state_clone = state.clone();
    tokio::spawn(async move {
        let _permit = permit;
        crate::fix_flow::run_fix_flow(
            &state_clone.llm,
            &state_clone.pool,
            &state_clone.tool_specs,
            &mode,
            state_clone.publisher.as_deref(),
            &req,
            &ctx,
            max_iterations,
            MAX_TOKENS,
            shutdown_rx,
        )
        .await;
    });

    let body = Json(serde_json::json!({ "correlation_id": correlation_id }));
    (StatusCode::ACCEPTED, body).into_response()
}

fn scope_required_response() -> Response {
    let body = Json(serde_json::json!({ "error": "scope read+act required" }));
    (StatusCode::FORBIDDEN, body).into_response()
}

/// Approve a previously-proposed write-tool action and dispatch it.
///
/// Flow: scope-gate → re-decode JWT for `sub` claim → `flow.confirm` (atomic
/// CAS via DashMap entry-lock) → dispatch via `state.pool.call` → mark
/// executed. `mark_executed` is best-effort: if the AMQP broker is down the
/// SF write already succeeded, and operators replay via correlation_id.
async fn chat_approve(
    scope: crate::gateway::auth::AuthScope,
    State(state): State<Arc<AppState>>,
    headers: axum::http::HeaderMap,
    Json(body): Json<ApproveBody>,
) -> Response {
    if scope != crate::gateway::auth::AuthScope::ReadAndAct {
        return scope_required_response();
    }
    let user_id = match crate::gateway::auth::current_user_id(&headers) {
        Some(id) => id,
        None => return scope_required_response(),
    };

    let action = match state.approval_flow.confirm(body.action_id, &user_id).await {
        Ok(a) => a,
        Err(e) => return approval_error_response(e),
    };

    let started = std::time::Instant::now();
    let (result, mut trace) = match state
        .pool
        .call(&action.tool_name, action.tool_args.clone())
        .await
    {
        Ok(t) => t,
        Err(e) => return AppError(e).into_response(),
    };
    let duration_ms = started.elapsed().as_millis() as u64;
    trace.status = Some("executed".into());
    trace.action_id = Some(action.action_id.to_string());

    if let Err(e) = state
        .approval_flow
        .mark_executed(action.action_id, &result, duration_ms)
        .await
    {
        tracing::warn!(action_id = %body.action_id, "mark_executed failed: {e:#}");
    }

    Json(ChatResponse {
        answer: result,
        cached: false,
        tool_trace: vec![trace],
        tokens: TokenUsage::default(),
        iterations: 0,
        correlation_id: action.correlation_id,
        suggestions: Vec::new(),
    })
    .into_response()
}

/// Body for `POST /chat/reject`. Optional `reason` rides on the audit-log
/// envelope (and the rendered answer) so users see why their proposal was
/// rejected on retry.
#[derive(Debug, Deserialize)]
pub struct RejectBody {
    pub action_id: uuid::Uuid,
    pub reason: Option<String>,
}

/// Reject a previously-proposed write-tool action.
///
/// Mirror of `chat_approve` minus the dispatch — `flow.reject` runs the
/// same atomic CAS as `confirm`, but no MCP tool-call follows. The returned
/// `ChatResponse` shape matches `/chat` so Drupal `jarvis_chat` renders it
/// like any other turn (assistant bubble with the rejection notice).
async fn chat_reject(
    scope: crate::gateway::auth::AuthScope,
    State(state): State<Arc<AppState>>,
    headers: axum::http::HeaderMap,
    Json(body): Json<RejectBody>,
) -> Response {
    if scope != crate::gateway::auth::AuthScope::ReadAndAct {
        return scope_required_response();
    }
    let user_id = match crate::gateway::auth::current_user_id(&headers) {
        Some(id) => id,
        None => return scope_required_response(),
    };

    let action = match state
        .approval_flow
        .reject(body.action_id, &user_id, body.reason.clone())
        .await
    {
        Ok(a) => a,
        Err(e) => return approval_error_response(e),
    };

    let reason_text = body.reason.unwrap_or_else(|| "no reason given".to_string());
    Json(ChatResponse {
        answer: format!("Action rejected: {reason_text}"),
        cached: false,
        tool_trace: Vec::new(),
        tokens: TokenUsage::default(),
        iterations: 0,
        correlation_id: action.correlation_id,
        suggestions: Vec::new(),
    })
    .into_response()
}

/// Map a `ProgressEvent` to its SSE event-name field. The event-name lets
/// browser `EventSource` listeners filter; clients using `fetch` + a reader
/// can dispatch on it too. Mirrors the serde tag values for consistency.
fn progress_event_name(ev: &ProgressEvent) -> &'static str {
    match ev {
        ProgressEvent::Thinking { .. } => "thinking",
        ProgressEvent::TextChunk { .. } => "text_chunk",
        ProgressEvent::ToolCallStarted { .. } => "tool_call_started",
        ProgressEvent::ToolCallCompleted { .. } => "tool_call_completed",
        ProgressEvent::ApprovalPending { .. } => "approval_pending",
        ProgressEvent::Suggestions { .. } => "suggestions",
        ProgressEvent::Done { .. } => "done",
        ProgressEvent::Error { .. } => "error",
    }
}

/// `POST /chat/stream`. Same body validation + mode/scope mapping as `/chat`,
/// but the response is `text/event-stream` carrying [`ProgressEvent`]s. The
/// orchestrator runs in a spawned task whose tx side is forwarded to SSE
/// while in parallel accumulating the audit shape for the trailing
/// `chat_completed` AMQP event — so streaming clients still produce the same
/// audit-feed entry shape as `/chat` does for non-streaming.
///
/// Client disconnect mid-stream drops the SSE-rx; the orchestrator task
/// keeps running (its tx.send becomes Err and is ignored per orchestrator
/// channel discipline) until natural completion, and the AMQP event still
/// fires. Anthropic billing keeps running too — that's the price of "audit
/// continuity over partial cancellation" we accept for R2-grade audit.
async fn chat_stream(
    scope: crate::gateway::auth::AuthScope,
    State(state): State<Arc<AppState>>,
    axum::Extension(shutdown_rx): axum::Extension<watch::Receiver<bool>>,
    headers: axum::http::HeaderMap,
    Json(req): Json<ChatRequest>,
) -> Result<
    Sse<impl futures_util::Stream<Item = std::result::Result<SseEvent, std::convert::Infallible>>>,
    Response,
> {
    let correlation_id = uuid::Uuid::new_v4().to_string();

    let messages = req.into_messages().map_err(|e| {
        let body = Json(serde_json::json!({ "error": e }));
        (StatusCode::BAD_REQUEST, body).into_response()
    })?;
    let prompt = match messages.last().map(|m| &m.content[..]) {
        Some([ContentBlock::Text { text }]) => text.clone(),
        _ => String::new(),
    };
    let conversation_length = messages.len();
    tracing::info!(
        correlation_id = %correlation_id,
        prompt_length = prompt.len(),
        conversation_length,
        scope = ?scope,
        "/chat/stream received"
    );

    let mode = match scope {
        crate::gateway::auth::AuthScope::Read => {
            crate::agent::modes::AgentMode::ReadOnly(crate::agent::modes::ReadOnlyMode)
        }
        crate::gateway::auth::AuthScope::ReadAndAct => crate::agent::modes::AgentMode::Actionable(
            crate::agent::modes::ActionableMode::new(state.approval_flow.clone()),
        ),
    };
    let user_id = crate::gateway::auth::current_user_id(&headers).unwrap_or_default();
    let ctx = crate::agent::modes::DispatchContext {
        correlation_id: correlation_id.clone(),
        user_id,
        scope,
    };
    let memory_for_ingest = state.memory.clone();
    let user_id_for_ingest = ctx.user_id.clone();

    let system_prompt = match state.memory.as_deref() {
        Some(memory) => memory
            .augment_system_prompt(SETUP_PROMPT, &messages, Some(ctx.user_id.as_str()))
            .await
            .unwrap_or_else(|e| {
                tracing::warn!("memory retrieval failed; falling back to base prompt: {e:#}");
                SETUP_PROMPT.to_string()
            }),
        None => SETUP_PROMPT.to_string(),
    };
    let system_prompt = system_prompt_with_hints(system_prompt, &mode);

    let (sse_tx, mut sse_rx) = mpsc::channel::<ProgressEvent>(64);

    let state_clone = state.clone();
    let tool_specs = state.tool_specs.clone();
    let publisher = state.publisher.clone();
    let correlation_id_pub = correlation_id.clone();
    let prompt_pub = prompt;
    let started = std::time::Instant::now();

    tokio::spawn(async move {
        let (orch_tx, mut orch_rx) = mpsc::channel::<ProgressEvent>(64);
        let state_for_orch = state_clone;
        let specs_for_orch = tool_specs;
        let mode_for_orch = mode;
        let ctx_for_orch = ctx;
        let orch_handle = tokio::spawn(async move {
            orchestrator::run_with_messages_in_mode_streaming(
                messages,
                &system_prompt,
                &state_for_orch.llm,
                &state_for_orch.pool,
                &specs_for_orch,
                max_iterations_for(&mode_for_orch),
                MAX_TOKENS,
                &mode_for_orch,
                &ctx_for_orch,
                orch_tx,
            )
            .await
        });

        let mut answer = String::new();
        let mut tool_trace: Vec<serde_json::Value> = Vec::new();
        let mut tokens = TokenUsage::default();
        let mut iterations: u32 = 0;
        let mut succeeded = true;
        let mut timed_out = false;
        let mut shutdown_aborted = false;

        // Three-way race: orchestrator events, hard deadline (DoS bound
        // when no TimeoutLayer applies), graceful shutdown signal. The
        // deadline + shutdown branches both abort the inner orch_handle
        // so Anthropic billing stops as soon as one fires.
        let deadline = tokio::time::sleep(std::time::Duration::from_secs(STREAM_DEADLINE_SECS));
        tokio::pin!(deadline);
        let mut shutdown_rx = shutdown_rx;

        loop {
            tokio::select! {
                biased;
                _ = shutdown_rx.changed() => {
                    if *shutdown_rx.borrow() {
                        tracing::info!(
                            correlation_id = %correlation_id_pub,
                            "/chat/stream aborting on shutdown signal",
                        );
                        shutdown_aborted = true;
                        succeeded = false;
                        break;
                    }
                }
                _ = &mut deadline => {
                    tracing::warn!(
                        correlation_id = %correlation_id_pub,
                        elapsed_secs = STREAM_DEADLINE_SECS,
                        "/chat/stream wallclock timeout — aborting orchestrator",
                    );
                    timed_out = true;
                    succeeded = false;
                    break;
                }
                recv = orch_rx.recv() => {
                    match recv {
                        Some(ev) => {
                            match &ev {
                                ProgressEvent::TextChunk { text } => answer.push_str(text),
                                ProgressEvent::ToolCallCompleted {
                                    name,
                                    ok,
                                    ms,
                                    status,
                                    action_id,
                                } => {
                                    tool_trace.push(serde_json::json!({
                                        "tool": name,
                                        "ok": ok,
                                        "ms": ms,
                                        "status": status,
                                        "action_id": action_id,
                                    }));
                                }
                                ProgressEvent::Done {
                                    tokens: t,
                                    iterations: i,
                                    ..
                                } => {
                                    tokens = t.clone();
                                    iterations = *i;
                                }
                                ProgressEvent::Error { .. } => succeeded = false,
                                _ => {}
                            }
                            let _ = sse_tx.send(ev).await;
                        }
                        None => break,
                    }
                }
            }
        }

        if timed_out || shutdown_aborted {
            orch_handle.abort();
            let msg = if shutdown_aborted {
                "stream aborted by shutdown"
            } else {
                "stream timeout"
            };
            let _ = sse_tx
                .send(ProgressEvent::Error {
                    message: msg.into(),
                    correlation_id: correlation_id_pub.clone(),
                })
                .await;
        } else if let Ok(Err(_)) = orch_handle.await {
            succeeded = false;
        }

        let chat_outcome = if succeeded { "ok" } else { "error" };
        crate::metrics::record_chat("stream", chat_outcome, &tokens);

        if let Some(publisher) = publisher {
            let duration_ms = started.elapsed().as_millis() as u64;
            let payload = serde_json::json!({
                "correlation_id": correlation_id_pub,
                "prompt": prompt_pub,
                "answer": answer.clone(),
                "answer_length": answer.len(),
                "duration_ms": duration_ms,
                "conversation_length": conversation_length,
                "tool_trace": tool_trace,
                "tokens": tokens,
                "iterations": iterations,
                "succeeded": succeeded,
                "streamed": true,
                "timed_out": timed_out,
                "shutdown_aborted": shutdown_aborted,
            });
            if let Err(e) = publisher.publish_event("chat_completed", payload).await {
                tracing::warn!("failed to publish chat_completed event from stream: {e:#}");
            }
        }

        if succeeded
            && let Some(memory) = memory_for_ingest.as_deref()
            && let Err(e) = memory
                .remember_interaction(MemoryInteraction::new(
                    "default",
                    MemorySource::Chat,
                    correlation_id_pub.clone(),
                    Some(user_id_for_ingest.clone()),
                    prompt_pub.as_str(),
                    &answer,
                ))
                .await
        {
            tracing::warn!("memory ingestion failed: {e:#}");
        }
    });

    let event_stream = async_stream::stream! {
        yield Ok::<_, std::convert::Infallible>(
            SseEvent::default().comment(cloudflare_pad_comment()),
        );
        while let Some(ev) = sse_rx.recv().await {
            let name = progress_event_name(&ev);
            if let Ok(data) = serde_json::to_string(&ev) {
                yield Ok::<_, std::convert::Infallible>(SseEvent::default().event(name).data(data));
            }
        }
    };

    Ok(Sse::new(event_stream)
        .keep_alive(KeepAlive::new().interval(std::time::Duration::from_secs(keepalive_secs()))))
}

fn chat_suggestions_enabled() -> bool {
    std::env::var("CHAT_SUGGESTIONS_ENABLED")
        .ok()
        .map(|s| !s.trim().eq_ignore_ascii_case("false"))
        .unwrap_or(true)
}

/// Build the CORS layer from `CHAT_ALLOWED_ORIGINS` (comma-separated origins).
/// `CHAT_CORS_STRICT=true` upgrades misconfig from "fallback to permissive
/// with WARN" to "fail startup". Production sets it; dev leaves it off so
/// `cargo run` still works without an allow-list.
fn build_cors_layer() -> Result<CorsLayer> {
    let strict = std::env::var("CHAT_CORS_STRICT")
        .ok()
        .map(|s| s.trim().eq_ignore_ascii_case("true"))
        .unwrap_or(false);
    let raw = std::env::var("CHAT_ALLOWED_ORIGINS")
        .ok()
        .filter(|s| !s.trim().is_empty());
    parse_cors_allow_list(strict, raw.as_deref())
}

/// Pure decision-table for CORS layer construction. Tests can drive every
/// branch without touching process env.
fn parse_cors_allow_list(strict: bool, csv: Option<&str>) -> Result<CorsLayer> {
    let Some(csv) = csv else {
        if strict {
            bail!("CHAT_CORS_STRICT=true requires CHAT_ALLOWED_ORIGINS to be set");
        }
        tracing::warn!(
            "CHAT_ALLOWED_ORIGINS unset — using permissive CORS (dev-only, NOT for production)"
        );
        return Ok(CorsLayer::permissive());
    };

    let parsed: std::result::Result<Vec<HeaderValue>, _> = csv
        .split(',')
        .map(|o| o.trim())
        .filter(|o| !o.is_empty())
        .map(|o| o.parse::<HeaderValue>())
        .collect();

    match parsed {
        Ok(origins) if !origins.is_empty() => {
            tracing::info!(count = origins.len(), "CORS locked to allow-list");
            // Cache-Control is whitelisted for SSE clients that ask intermediate
            // proxies not to buffer. Last-Event-ID is intentionally NOT advertised:
            // the handler has no resume store, so promising the header is capability
            // we can't honor — add it back once we wire actual replay.
            Ok(CorsLayer::new()
                .allow_origin(origins)
                .allow_methods([Method::GET, Method::POST, Method::OPTIONS])
                .allow_headers([CONTENT_TYPE, axum::http::header::CACHE_CONTROL]))
        }
        Ok(_) if strict => {
            bail!("CHAT_ALLOWED_ORIGINS contained no usable origins under CHAT_CORS_STRICT=true")
        }
        Ok(_) => {
            tracing::warn!(
                "CHAT_ALLOWED_ORIGINS contained no usable origins — falling back to permissive"
            );
            Ok(CorsLayer::permissive())
        }
        Err(e) if strict => {
            bail!("CHAT_ALLOWED_ORIGINS parse failed under CHAT_CORS_STRICT=true: {e:#}")
        }
        Err(e) => {
            tracing::warn!("CHAT_ALLOWED_ORIGINS parse failed: {e:#} — falling back to permissive");
            Ok(CorsLayer::permissive())
        }
    }
}

fn should_trigger_analysis_now() -> bool {
    let now = Utc::now();
    let triggers = [
        NaiveTime::from_hms_opt(0, 0, 0).unwrap(),
        NaiveTime::from_hms_opt(8, 30, 0).unwrap(),
        NaiveTime::from_hms_opt(12, 30, 0).unwrap(),
        NaiveTime::from_hms_opt(16, 30, 0).unwrap(),
    ];
    let t = now.time();
    triggers
        .iter()
        .any(|&trig| t.hour() == trig.hour() && t.minute() == trig.minute())
}

async fn run_scheduled_trigger(
    state: Arc<AppState>,
    teams_config: Option<Arc<TeamsConfig>>,
    mut shutdown_rx: watch::Receiver<bool>,
) {
    // Seed the latch with the current minute so a container restart that
    // happens DURING a trigger minute (e.g. a CD redeploy at 08:30:15) does
    // not re-fire the same scheduled summary. Without this, the first
    // ticker.tick() returns immediately and `should_trigger_analysis_now()`
    // would dispatch a duplicate Teams + RabbitMQ summary.
    let now = Utc::now();
    let mut last_fired_minute: Option<(u32, u32)> = Some((now.hour(), now.minute()));
    let mut ticker = tokio::time::interval(std::time::Duration::from_secs(30));
    ticker.tick().await;

    loop {
        tokio::select! {
            biased;
            _ = shutdown_rx.changed() => {
                if *shutdown_rx.borrow() {
                    tracing::info!("scheduled trigger task shutting down");
                    return;
                }
            }
            _ = ticker.tick() => {
                let now = Utc::now();
                let key = (now.hour(), now.minute());
                if should_trigger_analysis_now() && last_fired_minute != Some(key) {
                    last_fired_minute = Some(key);
                    tracing::info!(?key, "scheduled trigger firing");
                    if let Err(e) = handle_scheduled(&state, teams_config.as_deref(), key).await {
                        tracing::error!("scheduled trigger error: {e:#}");
                    }
                }
            }
        }
    }
}

async fn handle_scheduled(
    state: &AppState,
    teams: Option<&TeamsConfig>,
    key: (u32, u32),
) -> anyhow::Result<()> {
    let messages = vec![Message {
        role: Role::User,
        content: vec![ContentBlock::Text {
            text: ANALYZE_CONTROLROOM_PROMPT.to_string(),
        }],
    }];
    let system_prompt = match state.memory.as_deref() {
        Some(memory) => memory
            .augment_system_prompt(SETUP_PROMPT, &messages, None)
            .await
            .unwrap_or_else(|e| {
                tracing::warn!("memory retrieval failed; falling back to base prompt: {e:#}");
                SETUP_PROMPT.to_string()
            }),
        None => SETUP_PROMPT.to_string(),
    };
    let outcome = orchestrator::run(
        ANALYZE_CONTROLROOM_PROMPT.to_string(),
        &system_prompt,
        &state.llm,
        &state.pool,
        &state.tool_specs,
        READ_ONLY_MAX_ITERATIONS,
        MAX_TOKENS,
    )
    .await
    .context("scheduled analyze prompt")?;
    let answer = outcome.answer;

    if let Some(memory) = state.memory.as_deref()
        && let Err(e) = memory
            .remember_interaction(MemoryInteraction::new(
                "default",
                MemorySource::ScheduledSummary,
                format!("{key:?}"),
                None::<String>,
                ANALYZE_CONTROLROOM_PROMPT,
                &answer,
            ))
            .await
    {
        tracing::warn!("memory ingestion failed: {e:#}");
    }

    if let Some(teams) = teams {
        publish_to_teams(teams, &answer).await?;
    }

    if let Some(publisher) = &state.publisher
        && let Err(e) = publisher
            .publish_event(
                "daily_summary_generated",
                serde_json::json!({
                    "trigger_time_utc": format!("{:02}:{:02}", key.0, key.1),
                    "answer": answer,
                    "answer_length": answer.len(),
                }),
            )
            .await
    {
        tracing::warn!("rabbitmq publish_event failed: {e:#}");
    }
    Ok(())
}

async fn forget_memory(
    scope: crate::gateway::auth::AuthScope,
    State(state): State<Arc<AppState>>,
    Path(user_id): Path<String>,
) -> Result<Json<serde_json::Value>, Response> {
    if scope != crate::gateway::auth::AuthScope::ReadAndAct {
        let body = Json(serde_json::json!({ "error": "forbidden" }));
        return Err((StatusCode::FORBIDDEN, body).into_response());
    }
    // Response cache isn't keyed by user; clear it wholesale so no cached
    // answer outlives the erasure. It's ephemeral and rebuilds in seconds.
    if let Some(cache) = state.cache.as_ref() {
        cache.clear().map_err(|e| AppError(e).into_response())?;
    }
    match state.memory.as_deref() {
        Some(memory) => {
            memory
                .forget_user(&user_id)
                .await
                .map_err(|e| AppError(e).into_response())?;
            tracing::info!(user_id = %user_id, "erased memory for user");
            Ok(Json(serde_json::json!({ "forgotten": user_id })))
        }
        None => Ok(Json(
            serde_json::json!({ "forgotten": user_id, "memory_enabled": false }),
        )),
    }
}

pub async fn serve(
    pool: McpPool,
    llm: AnthropicClient,
    teams_config: Option<TeamsConfig>,
    tool_specs: Vec<ToolSpec>,
    rabbitmq: Option<(Publisher, RabbitMqConfig)>,
    cache: Option<Arc<SqliteMemory>>,
    memory: Option<Arc<MemoryService>>,
) -> anyhow::Result<()> {
    let (publisher_arc, consumer_config) = match rabbitmq {
        Some((p, c)) => (Some(Arc::new(p)), Some(c)),
        None => (None, None),
    };

    // Attach publisher to the pool so every MCP tool-call fires a
    // `tool_called` event on `ai.events`. Must happen before the pool is
    // wrapped in `Arc<AppState>` (mutation requires &mut).
    let mut pool = pool;
    if let Some(p) = &publisher_arc {
        pool.attach_publisher(p.clone());
    }

    // Approval-flow primitives (PR-2). Cleanup task holds Arc<ApprovalStore>
    // — NOT Arc<AppState> — so the shutdown's Arc::try_unwrap on state
    // stays clean.
    let approval_store = Arc::new(crate::gateway::approval::state::ApprovalStore::new(
        approval_ttl(),
    ));
    let approval_audit = Arc::new(crate::gateway::audit::AuditPublisher::new(
        publisher_arc.clone(),
    ));
    let approval_flow = Arc::new(crate::gateway::approval::flow::ApprovalFlow::new(
        approval_store.clone(),
        approval_audit.clone(),
        approval_ttl(),
    ));

    let metrics_handle = crate::metrics::install()?;

    let state = Arc::new(AppState {
        llm,
        pool,
        tool_specs,
        publisher: publisher_arc,
        cache,
        memory,
        metrics_handle,
        approval_flow: approval_flow.clone(),
        fix_flow_limit: Arc::new(tokio::sync::Semaphore::new(MAX_CONCURRENT_FIX_FLOW)),
        heartbeat_state: Arc::new(crate::heartbeat::HeartbeatState::new()),
    });
    let teams_config = teams_config.map(Arc::new);

    let (shutdown_tx, shutdown_rx) = watch::channel(false);

    let trigger_handle = tokio::spawn(run_scheduled_trigger(
        state.clone(),
        teams_config.clone(),
        shutdown_rx.clone(),
    ));

    let cleanup_handle = tokio::spawn(crate::gateway::approval::state::run_cleanup_task(
        approval_store.clone(),
        approval_audit.clone(),
        shutdown_rx.clone(),
    ));

    let cache_sweeper_handle = state
        .cache
        .clone()
        .map(|cache| tokio::spawn(run_cache_sweeper(cache, shutdown_rx.clone())));

    let incident_pipeline: Arc<dyn crate::incident::diagnose::DiagnosePipeline> = Arc::new(
        crate::incident::diagnose::DefaultDiagnosePipeline::new(state.clone()),
    );
    let incident_handle = consumer_config.as_ref().map(|cfg| {
        tokio::spawn(crate::incident::consumer::run(
            cfg.clone(),
            state.publisher.clone(),
            Some(incident_pipeline.clone()),
            shutdown_rx.clone(),
        ))
    });

    let recovery_handle = consumer_config.as_ref().map(|cfg| {
        tokio::spawn(crate::incident::recovery::run(
            cfg.clone(),
            state.publisher.clone(),
            shutdown_rx.clone(),
        ))
    });

    let heartbeat_handle = consumer_config.as_ref().map(|cfg| {
        tokio::spawn(crate::rabbitmq::heartbeat::run(
            cfg.url.clone(),
            crate::rabbitmq::heartbeat::HeartbeatConfig::from_env(),
            shutdown_rx.clone(),
        ))
    });

    let heartbeat_tap_handle = consumer_config.as_ref().map(|cfg| {
        tokio::spawn(crate::heartbeat::run(
            cfg.clone(),
            state.heartbeat_state.clone(),
            shutdown_rx.clone(),
        ))
    });

    let consumer_handle =
        consumer_config.map(|cfg| tokio::spawn(rabbitmq_consumer::run(cfg, shutdown_rx.clone())));

    // `AuthScope` (see gateway::auth) gates each chat handler as an extractor,
    // i.e. inside the concurrency layer — an unauth request briefly holds a
    // slot until its cheap 401. Acceptable: internal-only, Drupal-proxied.
    match (
        std::env::var("CHAT_JWT_SECRET").is_ok_and(|s| !s.trim().is_empty()),
        std::env::var("CHAT_BEARER_TOKEN").is_ok_and(|s| !s.trim().is_empty()),
    ) {
        (false, false) => tracing::warn!(
            "chat-auth: no CHAT_JWT_SECRET or CHAT_BEARER_TOKEN — all bearer tokens accepted as scope=read (dev-only, NOT for production)"
        ),
        (jwt, static_token) => tracing::info!(
            jwt,
            static_token,
            "chat-auth enabled — unknown tokens rejected (401)"
        ),
    }
    // One concurrency layer shared across /chat + /chat/approve so both
    // routes draw from the same 8-slot pool. Plan §11.4: "8 concurrent
    // approvals + chats together is plenty for a single eventbeheerder."
    let chat_concurrency = tower::limit::GlobalConcurrencyLimitLayer::new(MAX_CONCURRENT_CHAT);

    let mut chat_route = post(chat);
    let mut approve_route = post(chat_approve);
    let mut reject_route = post(chat_reject);
    let mut stream_route = post(chat_stream);
    let mut fix_flow_route = post(fix_flow);
    let forget_route = delete(forget_memory);
    // GlobalConcurrencyLimit shares one Arc<Semaphore> across all per-request
    // service clones; the non-global variant builds a fresh semaphore per
    // Layer::layer() call → no actual capping. /chat/stream shares the same
    // 8-slot pool — long-lived SSE streams should not starve sync /chat
    // beyond MAX_CONCURRENT_CHAT, but the slot is held for the entire
    // stream duration (acceptable given Anthropic Tier 2 budget).
    chat_route = chat_route.route_layer(chat_concurrency.clone());
    approve_route = approve_route.route_layer(chat_concurrency.clone());
    reject_route = reject_route.route_layer(chat_concurrency.clone());
    fix_flow_route = fix_flow_route.route_layer(chat_concurrency.clone());
    stream_route = stream_route.route_layer(chat_concurrency);

    // Router-split: TimeoutLayer applies only to non-streaming routes. SSE
    // streams legitimately exceed 240s (e.g. multi-tool-cascade against
    // slow Salesforce) — applying the layer would kill them mid-flight.
    let timed_routes = Router::new()
        .route("/chat", chat_route)
        .route("/chat/approve", approve_route)
        .route("/chat/reject", reject_route)
        .route("/fix-flow", fix_flow_route)
        .route("/memory/user/{user_id}", forget_route)
        .layer(TimeoutLayer::with_status_code(
            StatusCode::REQUEST_TIMEOUT,
            std::time::Duration::from_secs(REQUEST_TIMEOUT_SECONDS),
        ));

    let app = Router::new()
        .merge(timed_routes)
        .route("/chat/stream", stream_route)
        .route("/health", get(health))
        .route("/metrics", get(metrics))
        .route("/status", get(status))
        // route_layer (not layer): runs after routing so MatchedPath is set.
        .route_layer(axum::middleware::from_fn(crate::metrics::track_http))
        .with_state(state.clone())
        .layer(DefaultBodyLimit::max(MAX_BODY_BYTES))
        .layer(build_cors_layer()?)
        // chat_stream pulls this via Extension<watch::Receiver<bool>> so the
        // spawned orchestrator task can abort cleanly on SIGTERM instead of
        // running to natural completion while the runtime drains.
        .layer(axum::Extension(shutdown_rx.clone()))
        .layer(TraceLayer::new_for_http());

    let listener = tokio::net::TcpListener::bind("0.0.0.0:8080")
        .await
        .context("binding 0.0.0.0:8080")?;
    tracing::info!("axum HTTP API listening on 0.0.0.0:8080");

    let shutdown_signal = {
        let shutdown_tx = shutdown_tx.clone();
        async move {
            let _ = tokio::signal::ctrl_c().await;
            tracing::info!("ctrl_c received, beginning graceful shutdown");
            let _ = shutdown_tx.send(true);
        }
    };

    axum::serve(listener, app)
        .with_graceful_shutdown(shutdown_signal)
        .await
        .context("axum::serve failed")?;

    if let Err(e) = trigger_handle.await {
        tracing::warn!("trigger task join error: {e:#}");
    }
    // The cleanup task uses tokio::select! over shutdown_rx + interval —
    // the watch::Receiver flip is observable within the tick interval, so
    // a 2s drain budget is enough.
    match tokio::time::timeout(std::time::Duration::from_secs(2), cleanup_handle).await {
        Ok(Ok(())) => {}
        Ok(Err(e)) => tracing::warn!("approval cleanup task join error: {e:#}"),
        Err(_) => tracing::warn!(
            "approval cleanup task didn't drain in 2s — task detached, runtime drop will reclaim"
        ),
    }
    if let Some(h) = cache_sweeper_handle {
        match tokio::time::timeout(std::time::Duration::from_secs(2), h).await {
            Ok(Ok(())) => {}
            Ok(Err(e)) => tracing::warn!("response cache sweeper join error: {e:#}"),
            Err(_) => tracing::warn!(
                "response cache sweeper didn't drain in 2s — task detached, runtime drop will reclaim"
            ),
        }
    }
    if let Some(h) = consumer_handle {
        // The consumer task uses `tokio::select!` against `shutdown_rx`, but
        // `consumer.next()` may not be cancellation-clean against all lapin
        // versions / broker states. Cap the drain time so a misbehaving
        // consumer can't keep the process alive past Ctrl+C.
        match tokio::time::timeout(std::time::Duration::from_secs(2), h).await {
            Ok(Ok(Ok(()))) => {}
            Ok(Ok(Err(e))) => tracing::warn!("rabbitmq consumer exited with error: {e:#}"),
            Ok(Err(e)) => tracing::warn!("rabbitmq consumer join error: {e:#}"),
            Err(_) => tracing::warn!(
                "rabbitmq consumer didn't drain in 2s — task left detached, runtime drop will reclaim"
            ),
        }
    }
    if let Some(h) = incident_handle {
        match tokio::time::timeout(std::time::Duration::from_secs(2), h).await {
            Ok(Ok(Ok(()))) => {}
            Ok(Ok(Err(e))) => tracing::warn!("incident consumer exited with error: {e:#}"),
            Ok(Err(e)) => tracing::warn!("incident consumer join error: {e:#}"),
            Err(_) => tracing::warn!(
                "incident consumer didn't drain in 2s — task left detached, runtime drop will reclaim"
            ),
        }
    }
    if let Some(h) = recovery_handle {
        match tokio::time::timeout(std::time::Duration::from_secs(2), h).await {
            Ok(Ok(Ok(()))) => {}
            Ok(Ok(Err(e))) => tracing::warn!("recovery consumer exited with error: {e:#}"),
            Ok(Err(e)) => tracing::warn!("recovery consumer join error: {e:#}"),
            Err(_) => tracing::warn!(
                "recovery consumer didn't drain in 2s — task left detached, runtime drop will reclaim"
            ),
        }
    }
    if let Some(h) = heartbeat_handle {
        match tokio::time::timeout(std::time::Duration::from_secs(2), h).await {
            Ok(Ok(Ok(()))) => {}
            Ok(Ok(Err(e))) => tracing::warn!("heartbeat publisher exited with error: {e:#}"),
            Ok(Err(e)) => tracing::warn!("heartbeat publisher join error: {e:#}"),
            Err(_) => tracing::warn!(
                "heartbeat publisher didn't drain in 2s — task left detached, runtime drop will reclaim"
            ),
        }
    }
    if let Some(h) = heartbeat_tap_handle {
        match tokio::time::timeout(std::time::Duration::from_secs(2), h).await {
            Ok(Ok(Ok(()))) => {}
            Ok(Ok(Err(e))) => tracing::warn!("heartbeat consumer exited with error: {e:#}"),
            Ok(Err(e)) => tracing::warn!("heartbeat consumer join error: {e:#}"),
            Err(_) => tracing::warn!(
                "heartbeat consumer didn't drain in 2s — task left detached, runtime drop will reclaim"
            ),
        }
    }

    match Arc::try_unwrap(state) {
        Ok(app_state) => {
            app_state.pool.shutdown().await?;
        }
        Err(_) => {
            tracing::warn!(
                "could not unwrap Arc<AppState> for clean pool shutdown — \
                 sessions die with the process"
            );
        }
    }

    Ok(())
}

#[cfg(test)]
mod tests;
