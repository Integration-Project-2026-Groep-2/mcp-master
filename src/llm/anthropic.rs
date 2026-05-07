//! Anthropic Messages API implementation of `LlmClient`.
//!
//! Translates between the provider-agnostic types in `super` and Anthropic's
//! wire shapes. The wire types are private to this module — nothing leaks.

use anyhow::{Context, bail};
use async_trait::async_trait;
use serde::{Deserialize, Serialize};
use serde_json::Value;

use super::{ChatResponse, ContentBlock, LlmClient, Message, Role, StopReason, ToolSpec};

/// Default Anthropic API host. Tests override via `with_base_url`.
const DEFAULT_BASE_URL: &str = "https://api.anthropic.com";

/// Default model. Latest Sonnet with extended thinking enabled by default.
const DEFAULT_MODEL: &str = "claude-sonnet-4-6";

/// Default reasoning budget (tokens) for extended thinking. Must be strictly
/// less than `max_tokens` per Anthropic; visible output gets the remainder.
const DEFAULT_THINKING_BUDGET: u32 = 2048;

/// Max number of retry attempts on 429/5xx/transient network errors.
/// Total attempts = MAX_RETRIES + 1 (initial). 4xx auth errors are NEVER
/// retried — they're deterministic and re-trying just wastes Anthropic quota.
const MAX_RETRIES: u32 = 3;

/// HTTP client for the Anthropic Messages API.
pub struct AnthropicClient {
    http: reqwest::Client,
    api_key: String,
    model: String,
    base_url: String,
    thinking_budget: Option<u32>,
}

impl AnthropicClient {
    pub fn new(api_key: String) -> Self {
        Self {
            http: reqwest::Client::new(),
            api_key,
            model: DEFAULT_MODEL.to_string(),
            base_url: DEFAULT_BASE_URL.to_string(),
            thinking_budget: Some(DEFAULT_THINKING_BUDGET),
        }
    }

    /// Disable extended thinking. Useful for cheap latency-sensitive paths.
    #[allow(dead_code)]
    pub fn without_thinking(mut self) -> Self {
        self.thinking_budget = None;
        self
    }

    /// Override the base URL — primarily for `wiremock`-driven tests.
    #[allow(dead_code)] // exercised under `#[cfg(test)]`; clippy can't see across cfg.
    pub fn with_base_url(mut self, url: String) -> Self {
        self.base_url = url;
        self
    }

    /// Read the API key from `ANTHROPIC_API_KEY` and fail fast if absent.
    pub fn from_env() -> anyhow::Result<Self> {
        let key =
            std::env::var("ANTHROPIC_API_KEY").context("ANTHROPIC_API_KEY env var is required")?;
        Ok(Self::new(key))
    }
}

#[async_trait]
impl LlmClient for AnthropicClient {
    async fn chat(
        &self,
        system: &str,
        messages: &[Message],
        tools: &[ToolSpec],
        max_tokens: u32,
    ) -> anyhow::Result<ChatResponse> {
        let req = AnthropicRequest {
            model: self.model.clone(),
            max_tokens,
            system: system.to_string(),
            thinking: self.thinking_budget.map(|b| ThinkingConfig {
                kind: "enabled",
                budget_tokens: b,
            }),
            tools: to_wire_tools(tools),
            messages: to_wire_messages(messages),
        };

        // Retry loop: up to MAX_RETRIES + 1 attempts total. Retries on 5xx,
        // 429, and transient network errors (timeout, connect reset). All
        // 4xx other than 429 are surfaced immediately — they're determ-
        // inistic (auth, malformed input) and retrying is pure waste.
        let mut attempt: u32 = 0;
        loop {
            if attempt > 0 {
                let delay = backoff_with_jitter(attempt);
                tracing::warn!(
                    attempt,
                    delay_ms = delay.as_millis() as u64,
                    "anthropic retry"
                );
                tokio::time::sleep(delay).await;
            }

            let send_result = self
                .http
                .post(format!("{}/v1/messages", self.base_url))
                .header("x-api-key", &self.api_key)
                .header("anthropic-version", "2023-06-01")
                .json(&req)
                .send()
                .await;

            let resp = match send_result {
                Ok(r) => r,
                Err(e) => {
                    let transient = e.is_timeout() || e.is_connect();
                    if attempt < MAX_RETRIES && transient {
                        attempt += 1;
                        continue;
                    }
                    return Err(
                        anyhow::Error::from(e).context("anthropic POST /v1/messages failed")
                    );
                }
            };

            let status = resp.status();
            if status.is_success() {
                let parsed: AnthropicResponse = resp
                    .json()
                    .await
                    .context("decoding Anthropic response JSON")?;
                return Ok(from_wire_response(parsed));
            }

            let retryable =
                status.is_server_error() || status == reqwest::StatusCode::TOO_MANY_REQUESTS;
            // Capture body on non-2xx — `error_for_status()` discards it, but
            // Anthropic puts useful error info inside the body JSON.
            let body = resp.text().await.unwrap_or_default();

            if attempt < MAX_RETRIES && retryable {
                tracing::warn!(%status, body = %body, "anthropic retryable status — backing off");
                attempt += 1;
                continue;
            }

            bail!("anthropic non-2xx: {status} {body}");
        }
    }
}

/// Exponential backoff with 0–500 ms jitter. Attempts 1/2/3 → ~1s/2s/4s
/// plus jitter to avoid retry-storms across instances.
fn backoff_with_jitter(attempt: u32) -> std::time::Duration {
    debug_assert!(
        attempt >= 1,
        "attempt must be 1-based when computing backoff"
    );
    let base_ms: u64 = 1000_u64 << (attempt.saturating_sub(1));
    let jitter_ms: u64 = rand::random::<u64>() % 500;
    std::time::Duration::from_millis(base_ms + jitter_ms)
}

// ---------- Wire types (private) -----------------------------------------

#[derive(Debug, Serialize)]
struct AnthropicRequest {
    model: String,
    max_tokens: u32,
    system: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    thinking: Option<ThinkingConfig>,
    #[serde(skip_serializing_if = "Vec::is_empty")]
    tools: Vec<AnthropicTool>,
    messages: Vec<AnthropicMessage>,
}

#[derive(Debug, Serialize)]
struct ThinkingConfig {
    #[serde(rename = "type")]
    kind: &'static str, // always "enabled" when emitted
    budget_tokens: u32,
}

#[derive(Debug, Serialize)]
struct AnthropicTool {
    name: String,
    description: String,
    input_schema: Value,
}

#[derive(Debug, Serialize, Deserialize)]
struct AnthropicMessage {
    role: String,
    content: Vec<AnthropicContent>,
}

/// Mirrors Anthropic's content-block taxonomy on the wire.
#[derive(Debug, Serialize, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case")]
enum AnthropicContent {
    Text {
        text: String,
    },
    ToolUse {
        id: String,
        name: String,
        input: Value,
    },
    ToolResult {
        tool_use_id: String,
        content: String,
        #[serde(default)]
        is_error: bool,
    },
    Thinking {
        thinking: String,
        signature: String,
    },
    RedactedThinking {
        data: String,
    },
}

#[derive(Debug, Deserialize)]
struct AnthropicResponse {
    content: Vec<AnthropicContent>,
    stop_reason: String,
}

// ---------- Translation helpers ------------------------------------------

fn to_wire_messages(messages: &[Message]) -> Vec<AnthropicMessage> {
    messages
        .iter()
        .map(|m| AnthropicMessage {
            role: match m.role {
                Role::User => "user".to_string(),
                Role::Assistant => "assistant".to_string(),
            },
            content: m.content.iter().map(content_to_wire).collect(),
        })
        .collect()
}

fn content_to_wire(block: &ContentBlock) -> AnthropicContent {
    match block {
        ContentBlock::Text { text } => AnthropicContent::Text { text: text.clone() },
        ContentBlock::ToolUse { id, name, input } => AnthropicContent::ToolUse {
            id: id.clone(),
            name: name.clone(),
            input: input.clone(),
        },
        ContentBlock::ToolResult {
            tool_use_id,
            content,
            is_error,
        } => AnthropicContent::ToolResult {
            tool_use_id: tool_use_id.clone(),
            content: content.clone(),
            is_error: *is_error,
        },
        ContentBlock::Thinking {
            thinking,
            signature,
        } => AnthropicContent::Thinking {
            thinking: thinking.clone(),
            signature: signature.clone(),
        },
        ContentBlock::RedactedThinking { data } => {
            AnthropicContent::RedactedThinking { data: data.clone() }
        }
    }
}

fn to_wire_tools(tools: &[ToolSpec]) -> Vec<AnthropicTool> {
    tools
        .iter()
        .map(|t| AnthropicTool {
            name: t.name.clone(),
            description: t.description.clone(),
            input_schema: t.input_schema.clone(),
        })
        .collect()
}

fn from_wire_response(resp: AnthropicResponse) -> ChatResponse {
    ChatResponse {
        content: resp.content.into_iter().map(content_from_wire).collect(),
        stop_reason: match resp.stop_reason.as_str() {
            "end_turn" => StopReason::EndTurn,
            "tool_use" => StopReason::ToolUse,
            "max_tokens" => StopReason::MaxTokens,
            other => StopReason::Other(other.to_string()),
        },
    }
}

fn content_from_wire(block: AnthropicContent) -> ContentBlock {
    match block {
        AnthropicContent::Text { text } => ContentBlock::Text { text },
        AnthropicContent::ToolUse { id, name, input } => ContentBlock::ToolUse { id, name, input },
        AnthropicContent::ToolResult {
            tool_use_id,
            content,
            is_error,
        } => ContentBlock::ToolResult {
            tool_use_id,
            content,
            is_error,
        },
        AnthropicContent::Thinking {
            thinking,
            signature,
        } => ContentBlock::Thinking {
            thinking,
            signature,
        },
        AnthropicContent::RedactedThinking { data } => ContentBlock::RedactedThinking { data },
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;
    use wiremock::matchers::{header, method, path};
    use wiremock::{Mock, MockServer, ResponseTemplate};

    #[test]
    fn translation_preserves_text_and_tool_use_shapes() {
        let messages = vec![
            Message {
                role: Role::User,
                content: vec![ContentBlock::Text {
                    text: "Wat zijn de heartbeats?".to_string(),
                }],
            },
            Message {
                role: Role::Assistant,
                content: vec![ContentBlock::ToolUse {
                    id: "toolu_1".to_string(),
                    name: "heartbeat_status".to_string(),
                    input: json!({"limit": 5}),
                }],
            },
            Message {
                role: Role::User,
                content: vec![ContentBlock::ToolResult {
                    tool_use_id: "toolu_1".to_string(),
                    content: "[]".to_string(),
                    is_error: false,
                }],
            },
        ];

        let wire = to_wire_messages(&messages);
        let json_value = serde_json::to_value(&wire).unwrap();

        let expected = json!([
            { "role": "user",      "content": [{ "type": "text", "text": "Wat zijn de heartbeats?" }] },
            { "role": "assistant", "content": [{ "type": "tool_use", "id": "toolu_1", "name": "heartbeat_status", "input": {"limit": 5} }] },
            { "role": "user",      "content": [{ "type": "tool_result", "tool_use_id": "toolu_1", "content": "[]", "is_error": false }] }
        ]);

        assert_eq!(json_value, expected);
    }

    #[test]
    fn from_wire_response_maps_stop_reasons() {
        let happy = AnthropicResponse {
            content: vec![AnthropicContent::Text { text: "ok".into() }],
            stop_reason: "end_turn".into(),
        };
        let mapped = from_wire_response(happy);
        assert_eq!(mapped.stop_reason, StopReason::EndTurn);

        let tu = AnthropicResponse {
            content: vec![],
            stop_reason: "tool_use".into(),
        };
        assert_eq!(from_wire_response(tu).stop_reason, StopReason::ToolUse);

        let mt = AnthropicResponse {
            content: vec![],
            stop_reason: "max_tokens".into(),
        };
        assert_eq!(from_wire_response(mt).stop_reason, StopReason::MaxTokens);

        let unknown = AnthropicResponse {
            content: vec![],
            stop_reason: "pause_turn".into(),
        };
        assert_eq!(
            from_wire_response(unknown).stop_reason,
            StopReason::Other("pause_turn".into())
        );
    }

    #[tokio::test]
    async fn chat_against_wiremock_end_turn_path() {
        let server = MockServer::start().await;
        Mock::given(method("POST"))
            .and(path("/v1/messages"))
            .and(header("x-api-key", "test-key"))
            .and(header("anthropic-version", "2023-06-01"))
            .respond_with(ResponseTemplate::new(200).set_body_json(json!({
                "content": [{ "type": "text", "text": "Klaar." }],
                "stop_reason": "end_turn"
            })))
            .mount(&server)
            .await;

        let client = AnthropicClient::new("test-key".into()).with_base_url(server.uri());
        let resp = client.chat("system", &[], &[], 4096).await.unwrap();

        assert_eq!(resp.stop_reason, StopReason::EndTurn);
        assert_eq!(
            resp.content,
            vec![ContentBlock::Text {
                text: "Klaar.".into()
            }]
        );
    }

    #[tokio::test]
    async fn chat_against_wiremock_tool_use_path() {
        let server = MockServer::start().await;
        Mock::given(method("POST"))
            .and(path("/v1/messages"))
            .respond_with(ResponseTemplate::new(200).set_body_json(json!({
                "content": [{
                    "type": "tool_use",
                    "id": "toolu_42",
                    "name": "heartbeat_status",
                    "input": {"limit": 5}
                }],
                "stop_reason": "tool_use"
            })))
            .mount(&server)
            .await;

        let client = AnthropicClient::new("k".into()).with_base_url(server.uri());
        let resp = client.chat("sys", &[], &[], 4096).await.unwrap();

        assert_eq!(resp.stop_reason, StopReason::ToolUse);
        match &resp.content[0] {
            ContentBlock::ToolUse { id, name, input } => {
                assert_eq!(id, "toolu_42");
                assert_eq!(name, "heartbeat_status");
                assert_eq!(input, &json!({"limit": 5}));
            }
            other => panic!("expected ToolUse, got {other:?}"),
        }
    }

    #[tokio::test]
    async fn chat_retries_on_429_then_succeeds() {
        let server = MockServer::start().await;
        // First request: 429
        Mock::given(method("POST"))
            .and(path("/v1/messages"))
            .respond_with(ResponseTemplate::new(429).set_body_string("rate limited"))
            .up_to_n_times(1)
            .mount(&server)
            .await;
        // Subsequent requests: 200
        Mock::given(method("POST"))
            .and(path("/v1/messages"))
            .respond_with(ResponseTemplate::new(200).set_body_json(json!({
                "content": [{ "type": "text", "text": "ok after retry" }],
                "stop_reason": "end_turn"
            })))
            .mount(&server)
            .await;

        let client = AnthropicClient::new("k".into()).with_base_url(server.uri());
        let resp = client.chat("sys", &[], &[], 4096).await.unwrap();
        assert_eq!(resp.stop_reason, StopReason::EndTurn);
        match &resp.content[0] {
            ContentBlock::Text { text } => assert_eq!(text, "ok after retry"),
            other => panic!("expected Text, got {other:?}"),
        }
    }

    #[tokio::test]
    async fn chat_does_not_retry_on_401() {
        let server = MockServer::start().await;
        Mock::given(method("POST"))
            .and(path("/v1/messages"))
            .respond_with(ResponseTemplate::new(401).set_body_string("invalid api key"))
            .expect(1) // Hard assert: must receive exactly 1 request, no retries
            .mount(&server)
            .await;

        let client = AnthropicClient::new("bad".into()).with_base_url(server.uri());
        let err = client.chat("sys", &[], &[], 4096).await.unwrap_err();
        let msg = format!("{err}");
        assert!(msg.contains("401"), "error should mention status: {msg}");
    }

    #[tokio::test]
    async fn chat_propagates_non_2xx_with_body() {
        let server = MockServer::start().await;
        Mock::given(method("POST"))
            .and(path("/v1/messages"))
            .respond_with(ResponseTemplate::new(401).set_body_string("invalid api key"))
            .mount(&server)
            .await;

        let client = AnthropicClient::new("bad".into()).with_base_url(server.uri());
        let err = client.chat("sys", &[], &[], 4096).await.unwrap_err();

        let msg = format!("{err}");
        assert!(msg.contains("401"), "error should mention status: {msg}");
        assert!(
            msg.contains("invalid api key"),
            "error should include body: {msg}"
        );
    }
}
