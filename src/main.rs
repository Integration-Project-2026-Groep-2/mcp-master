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

/// Per-call token cap.
const MAX_TOKENS: u32 = 8192;

struct TeamsConfig {
    team_id: String,
    channel_id: String,
    access_token: String,
}

struct LlmConfig {
    llm: AnthropicClient,
    mcp_client: mcp::McpClient,
    tool_specs: Vec<ToolSpec>,
}

impl LlmConfig {
    fn new(llm: AnthropicClient, mcp_client: mcp::McpClient, tool_specs: Vec<ToolSpec>) -> Self {
        Self {
            llm,
            mcp_client,
            tool_specs,
        }
    }

    async fn shutdown(self) -> Result<()> {
        self.mcp_client.shutdown().await
    }
}

impl TeamsConfig {
    fn from_env() -> Result<Self> {
        Ok(Self {
            team_id: std::env::var("TEAMS_ID").context("TEAMS_ID not set")?,
            channel_id: std::env::var("TEAMS_CHANNEL").context("TEAMS_CHANNEL not set")?,
            access_token: std::env::var("TEAMS_TOKEN").context("TEAMS_TOKEN not set")?,
        })
    }
}

async fn poll_messages(
    client: &reqwest::Client,
    teams_config: &TeamsConfig,
    llm_config: &LlmConfig,
) -> anyhow::Result<()> {
    let url = format!(
        "https://graph.microsoft.com/v1.0/teams/{}/channels/{}/messages",
        teams_config.team_id, teams_config.channel_id
    );

    let res = client
        .get(url)
        .bearer_auth(&teams_config.access_token)
        .send()
        .await?;

    let body: serde_json::Value = res.json().await?;

    // extract messages
    if let Some(messages) = body["value"].as_array() {
        for msg in messages {
            let content = msg["body"]["content"].as_str().unwrap_or("");

            // send to MCP pipeline
            handle_prompt(content, teams_config, llm_config).await?;
            println!("{content}");
        }
    }

    Ok(())
}

/// POST a plain-text message to a Teams channel via Graph API.
async fn publish_to_teams(config: &TeamsConfig, message: &str) -> Result<()> {
    let url = format!(
        "https://graph.microsoft.com/v1.0/teams/{}/channels/{}/messages",
        config.team_id, config.channel_id
    );

    // Graph API requires this exact JSON shape.
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

async fn run_prompt(prompt: &str, llm_config: &LlmConfig) -> anyhow::Result<String> {
    orchestrator::run(
        prompt.to_string(),
        SYSTEM_PROMPT,
        &llm_config.llm,
        &llm_config.mcp_client,
        &llm_config.tool_specs,
        MAX_ITERATIONS,
        MAX_TOKENS,
    )
    .await
}

async fn handle_prompt(
    prompt: &str,
    teams_config: &TeamsConfig,
    llm_config: &LlmConfig,
) -> anyhow::Result<()> {
    let answer = run_prompt(prompt, llm_config).await?;

    publish_to_teams(teams_config, &answer).await?;

    Ok(())
}

const SYSTEM_PROMPT: &str = "You are the master agent for the Desideriushogeschool ShiftFestival AI team. \
Answer the user's question by calling the appropriate MCP tools when relevant. \
Reply in the same language as the user.";

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

    let args: Vec<String> = std::env::args().collect();
    let terminal_mode = args.iter().any(|a| a == "--terminal-mode");

    let base_url = std::env::var("MCP_BASE_URL").unwrap_or_else(|_| "http://localhost".to_string());
    let port = std::env::var("MCP_PORT").unwrap_or_else(|_| "7002".to_string());

    let svc = mcp::connect(&base_url, &port).await?;

    let tools = svc.list_tools(Default::default()).await?;
    let tool_specs: Vec<ToolSpec> = tools.tools.iter().map(mcp::tool_to_spec).collect();
    tracing::info!(count = tool_specs.len(), "loaded tools");

    // --list-tools: debug shortcut to verify the MCP layer without burning
    // Anthropic credits.
    if args.iter().any(|a| a == "--list-tools") {
        for spec in &tool_specs {
            println!("{}\t{}", spec.name, spec.description);
        }
        let mcp_client = mcp::McpClient::new(svc);
        mcp_client.shutdown().await?;
        return Ok(());
    }

    let llm = AnthropicClient::from_env()?;
    let mcp_client = mcp::McpClient::new(svc);
    let llm_config = LlmConfig::new(llm, mcp_client, tool_specs);

    if terminal_mode {
        let prompt = read_prompt(&args)?;
        if prompt.trim().is_empty() {
            anyhow::bail!("no prompt provided (pass as argv[1] or via stdin)");
        }
        let answer = run_prompt(&prompt, &llm_config).await?;
        println!("{answer}");
    } else {
        let teams_config = TeamsConfig::from_env()?;
        let client = reqwest::Client::new();

        loop {

            poll_messages(&client, &teams_config, &llm_config).await?;
        }
    }

    llm_config.shutdown().await?;
    Ok(())
}
