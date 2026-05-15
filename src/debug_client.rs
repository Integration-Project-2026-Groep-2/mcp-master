//! CLI debug client that emulates the frontend for testing without spinning up the UI.
//! Handles HTTP requests to /chat endpoint and streams responses to stdout.

use anyhow::{bail, Result};
use serde::{Deserialize, Serialize};
use std::io::{self, BufRead, Write};

/// Response shape from /chat endpoint
#[derive(Debug, Serialize, Deserialize)]
struct ChatResponse {
    answer: String,
    tool_trace: Vec<ToolTrace>,
    tokens: TokenUsage,
    iterations: u32,
    correlation_id: String,
    #[serde(default)]
    suggestions: Vec<String>,
}

#[derive(Debug, Serialize, Deserialize)]
struct ToolTrace {
    tool: String,
    server: String,
    ms: u64,
    ok: bool,
    #[serde(skip_serializing_if = "Option::is_none")]
    error: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    status: Option<String>,
}

#[derive(Debug, Serialize, Deserialize, Default)]
struct TokenUsage {
    input: u32,
    output: u32,
    #[serde(skip_serializing_if = "Option::is_none")]
    cache_creation_input: Option<u32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    cache_read_input: Option<u32>,
}

/// Request shape for /chat endpoint
#[derive(Debug, Serialize)]
struct ChatRequest {
    prompt: String,
}

/// Run the interactive CLI debug client
pub async fn run_debug_client(base_url: &str) -> Result<()> {
    println!("MCP Frontend Debug Client");
    println!("Backend: {}", base_url);
    println!("Commands: type your prompt and press Enter. Type 'exit' to quit.");
    println!();

    let bearer_token = std::env::var("CHAT_BEARER_TOKEN")
        .unwrap_or_else(|_| "debug-client-placeholder".to_string());
    if std::env::var("CHAT_BEARER_TOKEN").is_err() {
        println!("Auth: no CHAT_BEARER_TOKEN set; using placeholder token");
    }

    let stdin = io::stdin();
    let reader = stdin.lock();
    let mut lines = reader.lines();
    let client = reqwest::Client::new();

    loop {
        print!("> ");
        io::stdout().flush()?;

        match lines.next() {
            Some(Ok(input)) => {
                let trimmed = input.trim();
                if trimmed.is_empty() {
                    continue;
                }
                if trimmed.eq_ignore_ascii_case("exit") {
                    println!("\nGoodbye!");
                    break;
                }

                match send_chat_request(&client, base_url, &bearer_token, trimmed).await {
                    Ok(response) => {
                        display_response(&response);
                    }
                    Err(e) => {
                        eprintln!("Error: {}", e);
                    }
                }
            }
            Some(Err(e)) => {
                eprintln!("Error reading input: {}", e);
            }
            None => break,
        }
    }

    Ok(())
}

/// Send a chat request to the backend
async fn send_chat_request(
    client: &reqwest::Client,
    base_url: &str,
    bearer_token: &str,
    prompt: &str,
) -> Result<ChatResponse> {
    let url = format!("{}/chat", base_url);
    let request = ChatRequest {
        prompt: prompt.to_string(),
    };

    let response = client
        .post(&url)
        .json(&request)
        .bearer_auth(bearer_token)
        .timeout(std::time::Duration::from_secs(300))
        .send()
        .await?;

    if !response.status().is_success() {
        let status = response.status();
        let text = response.text().await.unwrap_or_default();
        bail!("HTTP {}: {}", status, text);
    }

    let chat_response: ChatResponse = response.json().await?;
    Ok(chat_response)
}

/// Pretty-print the response to stdout
fn display_response(response: &ChatResponse) {
    println!();
    println!("Response ID: {}", &response.correlation_id[..8]);
    println!();

    // Main answer
    println!("Answer:");
    println!("{}\n", response.answer);

    // Tool trace
    if !response.tool_trace.is_empty() {
        println!("Tool calls ({} calls, {} iterations):", response.tool_trace.len(), response.iterations);
        for (_i, trace) in response.tool_trace.iter().enumerate() {
            let status = if trace.ok { "ok" } else { "error" };
            println!("- [{}] {}/{} - {} ms", status, trace.server, trace.tool, trace.ms);
            if let Some(err) = &trace.error {
                println!("  Error: {}", err);
            }
        }
        println!();
    }

    // Token usage
    println!("Tokens: input={}, output={}", response.tokens.input, response.tokens.output);

    if let Some(cache_creation) = response.tokens.cache_creation_input {
        println!("Cache creation: {}", cache_creation);
    }
    if let Some(cache_read) = response.tokens.cache_read_input {
        println!("Cache read: {}", cache_read);
    }

    // Suggestions
    if !response.suggestions.is_empty() {
        println!();
        println!("Suggestions:");
        for (i, sugg) in response.suggestions.iter().enumerate() {
            println!("{}. {}", i + 1, sugg);
        }
    }

    println!();
}
