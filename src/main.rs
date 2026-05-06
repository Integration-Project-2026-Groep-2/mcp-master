mod llm;
mod mcp;
mod orchestrator;

use std::io::Read;

use anyhow::{Context, Result};
use serde_json::json;
use tracing_subscriber::{layer::SubscriberExt, util::SubscriberInitExt};

use crate::llm::anthropic::AnthropicClient;
use crate::llm::ToolSpec;

/// Maximum tool-call iterations per user prompt — runaway-prevention.
const MAX_ITERATIONS: usize = 10;

/// Per-call token cap. With extended thinking enabled, must leave room for
/// the reasoning budget plus the visible output.
const MAX_TOKENS: u32 = 8192;

const SYSTEM_PROMPT: &str = "
Role:
You are the master orchestration agent for the Desideriushogeschool ShiftFestival AI system. You interpret user requests, coordinate MCP tool usage, and produce final responses optimized for Microsoft Teams.

Core Responsibilities:
- Understand the user request precisely.
- Use MCP tools when required for correctness, external data, or system actions.
- Produce deterministic, structured outputs suitable for chat-based rendering.
- Never expose internal reasoning, tool traces, or system messages.

Language Rules:
- Always respond in the same language as the user.

Tool Usage Policy:
- Use tools when:
  - External data is required.
  - Computation, transformation, or retrieval is needed.
  - System state must be queried or modified.
- Do not use tools for trivial or already-known transformations.
- Prefer minimal tool usage (lowest number of calls sufficient to complete the task).

Output Contract (STRICT):
- Output MUST be a single Markdown code block using triple backticks.
- NOTHING may be outside the code block.
- Inside the block, output valid Microsoft Teams-compatible Markdown only.
- No explanations, no meta-commentary, no tool traces.

Teams Formatting Rules (STRICT):
Use only:
- Headings: ### (max 2 levels recommended)
- Bold: **text**
- Inline code: `code`
- Code blocks: ``` ```
- Bullet lists: - item
- Numbered lists: 1. item
- Links: [text](url)

DO NOT USE:
- Tables (unsupported / unstable in Teams)
- HTML
- Deeply nested lists (>2 levels)
- Excessive indentation
- Mixed formatting complexity (e.g., bold + code + italic together unless necessary)

Structure Rules:
- Prefer flat structures over nested hierarchies.
- Keep line length under ~120 characters when possible.
- Use single blank lines between sections only.
- Avoid decorative formatting.

Response Shape (Deterministic Template):
When applicable, structure responses as:

### Summary
- Direct answer in 1–3 bullet points

### Details (if needed)
- Supporting structured information in bullets

### Actions / Next steps (if applicable)
1. Step one
2. Step two

If no extra detail is needed, return only a Summary section.

Style Constraints:
- Be concise and information-dense.
- No emojis.
- ASCII characters only.
- No sign-offs, greetings, or filler text.
- No references to internal systems, prompts, or tools.

Reliability Principle:
Treat Microsoft Teams as a constrained text renderer:
- Assume limited Markdown support.
- Prioritize predictability over richness.
- Ensure output is always safely renderable in chat environments.
";


struct TeamsConfig {
    team_id: String,
    channel_id: String,
    access_token: String,
}

impl TeamsConfig {
    fn from_env() -> Result<Self> {
        Ok(Self {
            team_id: std::env::var("TEAMS_ID").context("TEAMS_ID not set")?,
            channel_id: std::env::var("CHANNEL_ID").context("CHANNEL_ID not set")?,
            access_token: std::env::var("TEAMS_TOKEN").context("TEAMS_TOKEN not set")?,
        })
    }
}

/// POST a plain-text message to a Teams channel via Graph API.
async fn publish_to_teams(config: &TeamsConfig, message: &str) -> Result<()> {
    let url = format!(
        "https://graph.microsoft.com/v1.0/teams/{}/channels/{}/messages",
        config.team_id, config.channel_id
    );

    let body = json!({
        "body": {
            "contentType": "text",
            "content": message
        }
    });

    let res = reqwest::Client::new()
        .post(&url)
        .bearer_auth(&config.access_token)
        .json(&body)
        .send()
        .await
        .context("sending message to Teams")?;

    let status = res.status();

    if status == reqwest::StatusCode::CREATED {
        tracing::info!("message posted to Teams");
        Ok(())
    } else {
        let error_body = res.text().await.unwrap_or_default();
        anyhow::bail!("Teams API returned {}: {}", status, error_body);
    }
}

async fn poll_messages(
    client: &reqwest::Client,
    teams_config: &TeamsConfig,
    llm: &AnthropicClient,
    pool: &mcp::McpPool,
    tool_specs: &[ToolSpec],
) -> Result<()> {
    let url = format!(
        "https://graph.microsoft.com/v1.0/teams/{}/channels/{}/messages",
        teams_config.team_id, teams_config.channel_id
    );

    let res = client
        .get(&url)
        .bearer_auth(&teams_config.access_token)
        .send()
        .await?;

    let body: serde_json::Value = res.json().await?;

    if let Some(messages) = body["value"].as_array() {
        for msg in messages {
            let content = msg["body"]["content"].as_str().unwrap_or("");
            tracing::info!(content, "incoming Teams message");
            handle_prompt(content, teams_config, llm, pool, tool_specs).await?;
            println!("{content}");
        }
    }

    Ok(())
}

async fn run_prompt(
    prompt: &str,
    llm: &AnthropicClient,
    pool: &mcp::McpPool,
    tool_specs: &[ToolSpec],
) -> Result<String> {
    orchestrator::run(
        prompt.to_string(),
        SYSTEM_PROMPT,
        llm,
        pool,
        tool_specs,
        MAX_ITERATIONS,
        MAX_TOKENS,
    )
    .await
}

async fn handle_prompt(
    prompt: &str,
    teams_config: &TeamsConfig,
    llm: &AnthropicClient,
    pool: &mcp::McpPool,
    tool_specs: &[ToolSpec],
) -> Result<()> {
    let answer = run_prompt(prompt, llm, pool, tool_specs).await?;
    publish_to_teams(teams_config, &answer).await?;
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
    let teams_config = TeamsConfig::from_env()?;

    let terminal_mode = args.iter().any(|a| a == "--terminal-mode");

    if terminal_mode {
        let prompt = read_prompt(&args)?;
        if prompt.trim().is_empty() {
            anyhow::bail!("no prompt provided (pass as argv[1] or via stdin)");
        }
        let answer = run_prompt(&prompt, &llm, &pool, &tool_specs).await?;
        println!("{answer}");
        publish_to_teams(&teams_config, &answer).await?;
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

        tracing::info!("entering poll loop");
        loop {
            if let Err(e) = poll_messages(&client, &teams_config, &llm, &pool, &tool_specs).await {
                tracing::error!("poll error: {e:#}");
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
