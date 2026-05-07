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
    llm::{ToolSpec, anthropic::AnthropicClient},
    mcp::McpPool,
    orchestrator,
    prompts::{ANALYZE_CONTROLROOM_PROMPT, SETUP_PROMPT},
    tcom::{TeamsConfig, publish_to_teams},
};

const MAX_ITERATIONS: usize = 10;
const MAX_TOKENS: u32 = 8192;

pub struct AppState {
    pub llm: AnthropicClient,
    pub pool: McpPool,
    pub tool_specs: Vec<ToolSpec>,
}

#[derive(Debug, Deserialize)]
pub struct ChatRequest {
    pub prompt: String,
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
    let prompt = req.prompt.trim().to_string();
    if prompt.is_empty() {
        let body = Json(serde_json::json!({ "error": "prompt is empty" }));
        return Err((StatusCode::BAD_REQUEST, body).into_response());
    }
    tracing::info!(prompt = %prompt, "/chat received");

    let answer = orchestrator::run(
        prompt,
        SETUP_PROMPT,
        &state.llm,
        &state.pool,
        &state.tool_specs,
        MAX_ITERATIONS,
        MAX_TOKENS,
    )
    .await
    .map_err(|e| AppError(e).into_response())?;

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
                    if let Err(e) = handle_scheduled(&state, teams_config.as_deref()).await {
                        tracing::error!("scheduled trigger error: {e:#}");
                    }
                }
            }
        }
    }
}

async fn handle_scheduled(state: &AppState, teams: Option<&TeamsConfig>) -> anyhow::Result<()> {
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
    Ok(())
}

pub async fn serve(
    pool: McpPool,
    llm: AnthropicClient,
    teams_config: Option<TeamsConfig>,
    tool_specs: Vec<ToolSpec>,
) -> anyhow::Result<()> {
    let state = Arc::new(AppState {
        llm,
        pool,
        tool_specs,
    });
    let teams_config = teams_config.map(Arc::new);

    let (shutdown_tx, shutdown_rx) = watch::channel(false);

    let trigger_handle = tokio::spawn(run_scheduled_trigger(
        state.clone(),
        teams_config.clone(),
        shutdown_rx,
    ));

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
