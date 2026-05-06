mod llm;
mod mcp;
mod orchestrator;
mod prompts;
mod tcom;

use std::io::Read;
use anyhow::{Context, Result};
use tracing_subscriber::{layer::SubscriberExt, util::SubscriberInitExt};
use crate::{llm::anthropic::AnthropicClient, prompts::ANALYZE_CONTROLROOM_PROMPT};
use crate::llm::ToolSpec;
use crate::tcom::TeamsConfig;
use chrono::{NaiveTime, Timelike, Utc};

const MAX_ITERATIONS: usize = 10;
const MAX_TOKENS: u32 = 8192;


// event times for triggering events. 0.00 | 8.30 | 12.30 | 16.30
fn should_trigger_analysis() -> bool {
    let now = Utc::now();
    let triggers = [
        NaiveTime::from_hms_opt(0, 0, 0).unwrap(),
        NaiveTime::from_hms_opt(8, 30, 0).unwrap(),
        NaiveTime::from_hms_opt(12, 30, 0).unwrap(),
        NaiveTime::from_hms_opt(16, 30, 0).unwrap(),
    ];
    let current_time = now.time();
    triggers.iter().any(|&t| {
        current_time.hour() == t.hour() && current_time.minute() == t.minute()
    })
}


async fn run_prompt(
    prompt: &str,
    llm: &AnthropicClient,
    pool: &mcp::McpPool,
    tool_specs: &[ToolSpec],
) -> Result<String> {
    orchestrator::run(
        prompt.to_string(),
        prompts::SETUP_PROMPT,
        llm,
        pool,
        tool_specs,
        MAX_ITERATIONS,
        MAX_TOKENS,
    )
        .await
}

pub async fn handle_prompt(
    prompt: &str,
    teams_config: &TeamsConfig,
    llm: &AnthropicClient,
    pool: &mcp::McpPool,
    tool_specs: &[ToolSpec],
) -> Result<()> {
    let answer = run_prompt(prompt, llm, pool, tool_specs).await?;
    tcom::publish_to_teams(teams_config, &answer).await?;
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

    // NOTE(nasr): configuration stuff
    let llm = AnthropicClient::from_env()?;
    let teams_config = tcom::TeamsConfig::from_env()?;

    let terminal_mode = args.iter().any(|a| a == "--terminal-mode");
    let server_mode = args.iter().any(|a| a == "--server-mode");

    if terminal_mode {
        let prompt = read_prompt(&args)?;
        if prompt.trim().is_empty() {
            anyhow::bail!("no prompt provided (pass as argv[1] or via stdin)");
        }

        handle_prompt(&prompt, &teams_config, &llm, &pool, &tool_specs).await?;
    }  else if server_mode {

        // TODO(nasr): listen to 8080 for a future chat bot

        loop {

            if should_trigger_analysis()  {
                if let Err(e) = handle_prompt(ANALYZE_CONTROLROOM_PROMPT, &teams_config, &llm, &pool, &tool_specs).await {
                    tracing::error!("handle_prompt error: {e:#}");
                }
            }

        }

    } else {
        let teams_config = TeamsConfig::from_env()?;
        let client = reqwest::Client::builder()
            .timeout(std::time::Duration::from_secs(10))
            .build()?;

        tracing::info!("starting Teams poll loop");

        // Seed cursor
        tracing::info!("seeding cursor...");
        let seed_url = format!(
            "https://graph.microsoft.com/v1.0/teams/{}/channels/{}/messages?$top=1",
            teams_config.team_id, teams_config.channel_id
        );
        match client.get(&seed_url).bearer_auth(&teams_config.access_token).send().await {
            Ok(res) => {
                let status = res.status();
                tracing::info!(%status, "seed response received");
                match res.json::<serde_json::Value>().await {
                    Ok(body) => {
                        tracing::info!(body = %body, "seed body");
                    }
                    Err(e) => tracing::error!("failed to parse seed response: {e:#}"),
                }
            }
            Err(e) => tracing::error!("seed request failed: {e:#}"),
        }

        loop {
            match tcom::poll_messages(&client, &teams_config).await {
                Ok(messages) => {
                    for msg in messages {
                        let content = msg["body"]["content"].as_str().unwrap_or("");
                        tracing::info!(content, "incoming Teams message");

                        println!("{content}");
                    }


                }
                Err(e) => tracing::error!("poll error: {e:#}"),
            }
            tokio::time::sleep(std::time::Duration::from_secs(5)).await;
        }
    }

    pool.shutdown().await?;
    Ok(())
}


/// Resolve MCP endpoints from env. Prefers the multi-server format
/// `MCP_SERVERS=label@url,label@url,...`; falls back to legacy single-server
/// `MCP_BASE_URL` + `MCP_PORT` when the multi-server var is absent or empty.
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

/// Prompt comes from argv[1] preferred (one-shot), falling back to stdin
/// for pipe-friendly usage. Empty string is rejected by the caller.
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
