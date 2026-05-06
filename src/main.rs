mod llm;
mod mcp;
mod orchestrator;

use std::io::Read;

use anyhow::{Context, Result};
use tracing_subscriber::{layer::SubscriberExt, util::SubscriberInitExt};

use crate::llm::ToolSpec;
use crate::llm::anthropic::AnthropicClient;

/// Maximum tool-call iterations per user prompt — runaway-prevention.
const MAX_ITERATIONS: usize = 10;

/// Per-call token cap. Cheaper than the model's hard limit and protects
/// the budget from prompt-bloat. With extended thinking enabled, must
/// leave room for the reasoning budget plus the visible output.
const MAX_TOKENS: u32 = 8192;

const SYSTEM_PROMPT: &str = "You are the master agent for the Desideriushogeschool ShiftFestival AI team. \
Answer the user's question by calling the appropriate MCP tools when relevant. \
Reply in the same language as the user.";

#[tokio::main]
async fn main() -> Result<()> {
    // Load .env (if present) into the process environment. Silent-OK when
    // missing — production sets these via the container env directly.
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

    // --list-tools: debug shortcut to verify the MCP layer without burning
    // Anthropic credits. Keep this small — promote to clap if a third flag
    // ever appears.
    if args.iter().any(|a| a == "--list-tools") {
        for spec in &tool_specs {
            println!("{}\t{}", spec.name, spec.description);
        }
        pool.shutdown().await?;
        return Ok(());
    }

    // Anthropic is only required for the agent path, not for --list-tools,
    // so we read the key only after the early-return.
    let llm = AnthropicClient::from_env()?;

    let prompt = read_prompt(&args)?;
    if prompt.trim().is_empty() {
        anyhow::bail!("no prompt provided (pass as argv[1] or via stdin)");
    }

    let answer = orchestrator::run(
        prompt,
        SYSTEM_PROMPT,
        &llm,
        &pool,
        &tool_specs,
        MAX_ITERATIONS,
        MAX_TOKENS,
    )
    .await?;

    println!("{answer}");

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

    // Legacy fallback. Default port 7002 preserved from prior behaviour.
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
