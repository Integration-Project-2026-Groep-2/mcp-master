mod agent;
mod http_api;
mod mcp;
mod rabbitmq;
mod retry;
mod teams;

use anyhow::{Context, Result};
use std::io::Read;
use tracing_subscriber::{layer::SubscriberExt, util::SubscriberInitExt};

use crate::agent::llm::ToolSpec;
use crate::agent::llm::anthropic::AnthropicClient;
use crate::teams::TeamsConfig;

const MAX_ITERATIONS: usize = 10;
const MAX_TOKENS: u32 = 8192;

async fn run_prompt(
    prompt: &str,
    llm: &AnthropicClient,
    pool: &mcp::McpPool,
    tool_specs: &[ToolSpec],
) -> Result<String> {
    let outcome = agent::orchestrator::run(
        prompt.to_string(),
        agent::prompts::SETUP_PROMPT,
        llm,
        pool,
        tool_specs,
        MAX_ITERATIONS,
        MAX_TOKENS,
    )
    .await?;
    Ok(outcome.answer)
}

pub async fn handle_prompt(
    prompt: &str,
    teams_config: &TeamsConfig,
    llm: &AnthropicClient,
    pool: &mcp::McpPool,
    tool_specs: &[ToolSpec],
) -> Result<()> {
    let answer = run_prompt(prompt, llm, pool, tool_specs).await?;
    teams::publish_to_teams(teams_config, &answer).await?;
    Ok(())
}

#[tokio::main]
async fn main() -> Result<()> {
    let _ = dotenvy::dotenv();

    tracing_subscriber::registry()
        .with(
            tracing_subscriber::EnvFilter::try_from_default_env()
                .unwrap_or_else(|_| "info,mcp_master=debug".into()),
        )
        .with(tracing_subscriber::fmt::layer())
        .init();

    let endpoints = resolve_endpoints();
    if endpoints.is_empty() {
        anyhow::bail!(
            "no MCP endpoints configured — set MCP_SERVERS=label@url,... \
             or legacy MCP_BASE_URL+MCP_PORT"
        );
    }

    let pool = mcp::McpPool::connect(endpoints).await?;
    let tool_specs: Vec<ToolSpec> = pool.tool_specs();
    tracing::info!(count = tool_specs.len(), "loaded tools");

    let args: Vec<String> = std::env::args().collect();

    if args.iter().any(|a| a == "--list-tools") {
        for spec in &tool_specs {
            println!("{}\t{}", spec.name, spec.description);
        }
        pool.shutdown().await?;
        return Ok(());
    }

    let llm = AnthropicClient::from_env()?;
    let teams_config = match teams::TeamsConfig::from_env() {
        Ok(c) => Some(c),
        Err(e) => {
            tracing::info!("Teams config absent: {e:#} — Teams-publish disabled");
            None
        }
    };

    let terminal_mode = args.iter().any(|a| a == "--terminal-mode");
    let server_mode = args.iter().any(|a| a == "--server-mode");

    if terminal_mode {
        let prompt = read_prompt(&args)?;
        if prompt.trim().is_empty() {
            anyhow::bail!("no prompt provided (pass as argv[1] or via stdin)");
        }
        let teams_config = teams_config
            .as_ref()
            .context("--terminal-mode requires TEAMS_ID, CHANNEL_ID, TEAMS_TOKEN")?;
        handle_prompt(&prompt, teams_config, &llm, &pool, &tool_specs).await?;
    } else if server_mode {
        tracing::info!("starting axum HTTP API on :8080");
        let rabbitmq = bootstrap_rabbitmq().await;
        http_api::serve(pool, llm, teams_config, tool_specs, rabbitmq).await?;
        return Ok(());
    } else {
        // No execution mode flag. The previous default — a Teams Graph API
        // polling loop — required write-permissions the AI-team doesn't have
        // and logged raw response bodies (potentially containing access
        // tokens). It was a foot-gun: deploys without a flag silently leaked
        // and never processed prompts. Bail with a clear hint instead.
        pool.shutdown().await?;
        anyhow::bail!(
            "no execution mode set; pass one of: \
             --server-mode (production HTTP API), \
             --terminal-mode (one-shot CLI prompt), \
             --list-tools (debug, prints aggregated MCP tools)"
        );
    }

    pool.shutdown().await?;
    Ok(())
}

fn resolve_endpoints() -> Vec<(String, String)> {
    if let Ok(value) = std::env::var("MCP_SERVERS") {
        let parsed = mcp::parse_endpoints(&value);
        if !parsed.is_empty() {
            return parsed;
        }
        tracing::warn!(
            "MCP_SERVERS is set but parsed to zero valid entries — \
             falling back to legacy single-server vars"
        );
    }

    let base = std::env::var("MCP_BASE_URL").unwrap_or_else(|_| "http://localhost".into());
    let port = std::env::var("MCP_PORT").unwrap_or_else(|_| "7002".into());
    vec![("default".into(), format!("{base}:{port}/mcp"))]
}

async fn bootstrap_rabbitmq() -> Option<(
    rabbitmq::publisher::Publisher,
    rabbitmq::config::RabbitMqConfig,
)> {
    let cfg = match rabbitmq::config::RabbitMqConfig::from_env() {
        Ok(c) => c,
        Err(e) => {
            tracing::info!("rabbitmq config absent: {e:#} — running without");
            return None;
        }
    };
    match rabbitmq::publisher::Publisher::connect(&cfg).await {
        Ok(p) => {
            tracing::info!(
                broker = %cfg.host_for_logging(),
                exchange = %cfg.exchange,
                "rabbitmq publisher connected"
            );
            Some((p, cfg))
        }
        Err(e) => {
            // Pass redacted broker as a separate field so the password embedded
            // in cfg.url cannot leak via the error chain into stdout.
            tracing::warn!(
                broker = %cfg.host_for_logging(),
                "rabbitmq publisher connect failed: {e:#} — running without"
            );
            None
        }
    }
}

fn read_prompt(args: &[String]) -> Result<String> {
    if let Some(arg) = args.iter().skip(1).find(|a| !a.starts_with("--")) {
        return Ok(arg.clone());
    }
    let mut buf = String::new();
    std::io::stdin()
        .lock()
        .read_to_string(&mut buf)
        .context("reading prompt from stdin")?;
    Ok(buf)
}
