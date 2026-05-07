use std::sync::Arc;

use anyhow::Context;
use axum::{
    Json, Router,
    extract::State,
    http::StatusCode,
    response::{IntoResponse, Response},
    routing::{get, post},
};
use chrono::{NaiveTime, Timelike, Utc};
use serde::{Deserialize, Serialize};
use tokio::sync::watch;
use tower_http::{cors::CorsLayer, trace::TraceLayer};

use crate::{
    llm::{ContentBlock, Message, Role, ToolSpec, anthropic::AnthropicClient},
    mcp::McpPool,
    orchestrator,
    prompts::{ANALYZE_CONTROLROOM_PROMPT, SETUP_PROMPT},
    rabbitmq::{config::RabbitMqConfig, consumer as rabbitmq_consumer, publisher::Publisher},
    tcom::{TeamsConfig, publish_to_teams},
};

const MAX_ITERATIONS: usize = 10;
const MAX_TOKENS: u32 = 8192;

pub struct AppState {
    pub llm: AnthropicClient,
    pub pool: McpPool,
    pub tool_specs: Vec<ToolSpec>,
    pub publisher: Option<Arc<Publisher>>,
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

#[derive(Debug, Serialize)]
pub struct ChatResponse {
    pub answer: String,
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
        tracing::error!("/chat handler error: {:#}", self.0);
        let body = Json(serde_json::json!({ "error": format!("{:#}", self.0) }));
        (StatusCode::INTERNAL_SERVER_ERROR, body).into_response()
    }
}

async fn health() -> &'static str {
    "ok"
}

async fn metrics() -> (StatusCode, &'static str) {
    (StatusCode::NOT_IMPLEMENTED, "not implemented")
}

async fn chat(
    State(state): State<Arc<AppState>>,
    Json(req): Json<ChatRequest>,
) -> Result<Json<ChatResponse>, Response> {
    let messages = req.into_messages().map_err(|e| {
        let body = Json(serde_json::json!({ "error": e }));
        (StatusCode::BAD_REQUEST, body).into_response()
    })?;
    // Extract the latest user prompt for tracing + the chat_completed event
    // payload before `messages` is moved into the orchestrator.
    let prompt = match messages.last().map(|m| &m.content[..]) {
        Some([ContentBlock::Text { text }]) => text.clone(),
        _ => String::new(),
    };
    let conversation_length = messages.len();
    tracing::info!(
        prompt = %prompt,
        conversation_length,
        "/chat received"
    );

    let started = std::time::Instant::now();
    let answer = orchestrator::run_with_messages(
        messages,
        SETUP_PROMPT,
        &state.llm,
        &state.pool,
        &state.tool_specs,
        MAX_ITERATIONS,
        MAX_TOKENS,
    )
    .await
    .map_err(|e| AppError(e).into_response())?;
    let duration_ms = started.elapsed().as_millis() as u64;

    if let Some(publisher) = &state.publisher {
        let payload = serde_json::json!({
            "prompt": prompt,
            "answer": answer,
            "answer_length": answer.len(),
            "duration_ms": duration_ms,
            "conversation_length": conversation_length,
        });
        if let Err(e) = publisher.publish_event("chat_completed", payload).await {
            tracing::warn!("failed to publish chat_completed event: {e:#}");
        }
    }

    Ok(Json(ChatResponse { answer }))
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
    let mut last_fired_minute: Option<(u32, u32)> = None;
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
    let answer = orchestrator::run(
        ANALYZE_CONTROLROOM_PROMPT.to_string(),
        SETUP_PROMPT,
        &state.llm,
        &state.pool,
        &state.tool_specs,
        MAX_ITERATIONS,
        MAX_TOKENS,
    )
    .await
    .context("scheduled analyze prompt")?;

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

pub async fn serve(
    pool: McpPool,
    llm: AnthropicClient,
    teams_config: Option<TeamsConfig>,
    tool_specs: Vec<ToolSpec>,
    rabbitmq: Option<(Publisher, RabbitMqConfig)>,
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

    let state = Arc::new(AppState {
        llm,
        pool,
        tool_specs,
        publisher: publisher_arc,
    });
    let teams_config = teams_config.map(Arc::new);

    let (shutdown_tx, shutdown_rx) = watch::channel(false);

    let trigger_handle = tokio::spawn(run_scheduled_trigger(
        state.clone(),
        teams_config.clone(),
        shutdown_rx.clone(),
    ));

    let consumer_handle =
        consumer_config.map(|cfg| tokio::spawn(rabbitmq_consumer::run(cfg, shutdown_rx.clone())));

    let app = Router::new()
        .route("/chat", post(chat))
        .route("/health", get(health))
        .route("/metrics", get(metrics))
        .with_state(state.clone())
        .layer(CorsLayer::permissive())
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
    if let Some(h) = consumer_handle {
        match h.await {
            Ok(Ok(())) => {}
            Ok(Err(e)) => tracing::warn!("rabbitmq consumer exited with error: {e:#}"),
            Err(e) => tracing::warn!("rabbitmq consumer join error: {e:#}"),
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
mod tests {
    use super::*;

    fn parse(json: &str) -> ChatRequest {
        serde_json::from_str(json).expect("ChatRequest should deserialize")
    }

    #[test]
    fn parse_legacy_prompt_shape() {
        let msgs = parse(r#"{"prompt":"hi"}"#).into_messages().unwrap();
        assert_eq!(msgs.len(), 1);
        assert!(matches!(msgs[0].role, Role::User));
        match &msgs[0].content[0] {
            ContentBlock::Text { text } => assert_eq!(text, "hi"),
            other => panic!("expected Text block, got {other:?}"),
        }
    }

    #[test]
    fn parse_messages_shape_three_turns() {
        let json = r#"{"messages":[
            {"role":"user","content":"q1"},
            {"role":"assistant","content":"a1"},
            {"role":"user","content":"q2"}
        ]}"#;
        let msgs = parse(json).into_messages().unwrap();
        assert_eq!(msgs.len(), 3);
        assert!(matches!(msgs[0].role, Role::User));
        assert!(matches!(msgs[1].role, Role::Assistant));
        assert!(matches!(msgs[2].role, Role::User));
        match &msgs[2].content[0] {
            ContentBlock::Text { text } => assert_eq!(text, "q2"),
            other => panic!("expected Text block, got {other:?}"),
        }
    }

    #[test]
    fn reject_both_fields_present() {
        let req = parse(r#"{"prompt":"hi","messages":[{"role":"user","content":"x"}]}"#);
        let err = req.into_messages().unwrap_err();
        assert!(err.contains("either"), "unexpected error: {err}");
    }

    #[test]
    fn reject_neither_field() {
        let err = parse(r#"{}"#).into_messages().unwrap_err();
        assert!(err.contains("missing"), "unexpected error: {err}");
    }

    #[test]
    fn reject_empty_messages_array() {
        let err = parse(r#"{"messages":[]}"#).into_messages().unwrap_err();
        assert!(err.contains("empty"), "unexpected error: {err}");
    }

    #[test]
    fn reject_assistant_as_last_turn() {
        let json = r#"{"messages":[
            {"role":"user","content":"q"},
            {"role":"assistant","content":"a"}
        ]}"#;
        let err = parse(json).into_messages().unwrap_err();
        assert!(err.contains("last message"), "unexpected error: {err}");
    }

    #[test]
    fn reject_empty_or_whitespace_prompt() {
        let err = parse(r#"{"prompt":"   "}"#).into_messages().unwrap_err();
        assert!(err.contains("empty"), "unexpected error: {err}");
    }
}
