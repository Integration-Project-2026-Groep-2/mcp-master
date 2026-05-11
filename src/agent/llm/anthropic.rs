//! Anthropic Messages API implementation of `LlmClient`.
//!
//! Translates between the provider-agnostic types in `super` and Anthropic's
//! wire shapes. The wire types are private to this module — nothing leaks.

use std::collections::BTreeMap;

use anyhow::{Context, bail};
use async_trait::async_trait;
use eventsource_stream::Eventsource;
use futures_util::stream::{BoxStream, StreamExt};
use serde::{Deserialize, Serialize};
use serde_json::Value;

use super::{
    ChatResponse, ContentBlock, LlmClient, Message, Role, StopReason, StreamEvent, TokenUsage,
    ToolSpec,
};
use crate::retry::backoff_with_jitter;

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
            stream: None,
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

    async fn stream_chat(
        &self,
        system: &str,
        messages: &[Message],
        tools: &[ToolSpec],
        max_tokens: u32,
    ) -> anyhow::Result<BoxStream<'static, anyhow::Result<StreamEvent>>> {
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
            stream: Some(true),
        };

        // Pre-stream retry: same shape as chat(). Once bytes start flowing,
        // mid-stream errors cannot be retried — already-emitted deltas would
        // be lost. Only HTTP-status-level failures (decided before the body)
        // are retryable here.
        let mut attempt: u32 = 0;
        let response = loop {
            if attempt > 0 {
                let delay = backoff_with_jitter(attempt);
                tracing::warn!(
                    attempt,
                    delay_ms = delay.as_millis() as u64,
                    "anthropic stream retry"
                );
                tokio::time::sleep(delay).await;
            }

            let send_result = self
                .http
                .post(format!("{}/v1/messages", self.base_url))
                .header("x-api-key", &self.api_key)
                .header("anthropic-version", "2023-06-01")
                .header("accept", "text/event-stream")
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
                    return Err(anyhow::Error::from(e)
                        .context("anthropic POST /v1/messages (stream) failed"));
                }
            };

            let status = resp.status();
            if status.is_success() {
                break resp;
            }

            let retryable =
                status.is_server_error() || status == reqwest::StatusCode::TOO_MANY_REQUESTS;
            let body = resp.text().await.unwrap_or_default();

            if attempt < MAX_RETRIES && retryable {
                tracing::warn!(%status, body = %body, "anthropic stream retryable status — backing off");
                attempt += 1;
                continue;
            }

            bail!("anthropic non-2xx: {status} {body}");
        };

        let event_stream = response.bytes_stream().eventsource();
        Ok(Box::pin(translate_anthropic_stream(event_stream)))
    }
}

// ---------- SSE event types (private) ------------------------------------
//
// `#[allow(dead_code)]` blanket: every type/fn below is reachable only via
// `stream_chat`, which production code starts calling in PR3 of this stack
// (orchestrator streaming variant). Until then the bin target's dead-code
// analysis flags them; tests already exercise the full path.

/// Top-level shape of the `data:` field per SSE event. `type` is the
/// discriminator; the matching variant carries the typed payload.
#[allow(dead_code)]
#[derive(Debug, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case")]
enum AnthropicSseData {
    MessageStart {
        message: AnthropicSseMessage,
    },
    ContentBlockStart {
        index: u32,
        content_block: AnthropicSseBlockStart,
    },
    ContentBlockDelta {
        index: u32,
        delta: AnthropicSseDelta,
    },
    ContentBlockStop {
        #[allow(dead_code)]
        index: u32,
    },
    MessageDelta {
        delta: AnthropicSseMessageDelta,
        #[serde(default)]
        usage: Option<AnthropicUsage>,
    },
    MessageStop,
    Ping,
    Error {
        error: AnthropicSseError,
    },
}

#[allow(dead_code)]
#[derive(Debug, Deserialize)]
struct AnthropicSseMessage {
    #[serde(default)]
    usage: Option<AnthropicUsage>,
}

#[allow(dead_code)]
#[derive(Debug, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case")]
enum AnthropicSseBlockStart {
    Text {
        #[serde(default)]
        text: String,
    },
    ToolUse {
        id: String,
        name: String,
        #[allow(dead_code)] // input arrives via input_json_delta; this is usually `{}`
        #[serde(default)]
        input: Value,
    },
    Thinking {
        #[serde(default)]
        thinking: String,
        #[serde(default)]
        signature: Option<String>,
    },
    RedactedThinking {
        data: String,
    },
}

#[allow(dead_code, clippy::enum_variant_names)] // `*Delta` suffix is Anthropic's wire vocab
#[derive(Debug, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case")]
enum AnthropicSseDelta {
    TextDelta { text: String },
    InputJsonDelta { partial_json: String },
    ThinkingDelta { thinking: String },
    SignatureDelta { signature: String },
}

#[allow(dead_code)]
#[derive(Debug, Deserialize)]
struct AnthropicSseMessageDelta {
    #[serde(default)]
    stop_reason: Option<String>,
}

#[allow(dead_code)]
#[derive(Debug, Deserialize)]
struct AnthropicSseError {
    #[serde(rename = "type")]
    kind: String,
    message: String,
}

/// Per-block accumulator. Lives until `content_block_stop` for that index.
/// On stream-end, `finalize()` converts each into a `ContentBlock` for
/// `StreamEvent::Done.full_content` — preserving signatures byte-for-byte
/// and parsing the concatenated tool-use input once.
#[allow(dead_code)]
enum BlockAccumulator {
    Text(String),
    ToolUse {
        id: String,
        name: String,
        json_buf: String,
    },
    Thinking {
        thinking: String,
        signature: String,
    },
    RedactedThinking {
        data: String,
    },
}

impl BlockAccumulator {
    #[allow(dead_code)]
    fn finalize(self) -> anyhow::Result<ContentBlock> {
        Ok(match self {
            BlockAccumulator::Text(text) => ContentBlock::Text { text },
            BlockAccumulator::ToolUse { id, name, json_buf } => {
                let input = if json_buf.is_empty() {
                    serde_json::json!({})
                } else {
                    serde_json::from_str(&json_buf)
                        .with_context(|| format!("parsing tool_use input_json: {json_buf}"))?
                };
                ContentBlock::ToolUse { id, name, input }
            }
            BlockAccumulator::Thinking {
                thinking,
                signature,
            } => ContentBlock::Thinking {
                thinking,
                signature,
            },
            BlockAccumulator::RedactedThinking { data } => ContentBlock::RedactedThinking { data },
        })
    }
}

/// Translate Anthropic's SSE wire stream into provider-agnostic `StreamEvent`s.
///
/// Uses `BTreeMap` for accumulators so the finalized block-order in
/// `Done.full_content` matches the index order Anthropic emits — critical
/// for thinking-then-text turns where order is semantically meaningful.
///
/// On any transport / parse / schema error: yields `Err`, then ends.
#[allow(dead_code)]
fn translate_anthropic_stream<S, E>(
    upstream: S,
) -> impl futures_util::Stream<Item = anyhow::Result<StreamEvent>> + Send + 'static
where
    S: futures_util::Stream<Item = Result<eventsource_stream::Event, E>> + Send + 'static,
    E: std::fmt::Display + Send + 'static,
{
    async_stream::try_stream! {
        let mut upstream = std::pin::pin!(upstream);
        let mut accumulators: BTreeMap<u32, BlockAccumulator> = BTreeMap::new();
        let mut usage: Option<TokenUsage> = None;
        let mut stop_reason: Option<StopReason> = None;

        while let Some(event_result) = upstream.next().await {
            let event = event_result.map_err(|e| anyhow::anyhow!("SSE transport error: {e}"))?;
            if event.data.is_empty() {
                continue;
            }
            let data: AnthropicSseData = serde_json::from_str(&event.data)
                .with_context(|| format!("parsing SSE data: {}", event.data))?;

            match data {
                AnthropicSseData::MessageStart { message } => {
                    if let Some(u) = message.usage {
                        usage = Some(TokenUsage {
                            input: u.input_tokens.unwrap_or(0),
                            output: u.output_tokens.unwrap_or(0),
                            cache_creation_input: u.cache_creation_input_tokens,
                            cache_read_input: u.cache_read_input_tokens,
                        });
                    }
                }
                AnthropicSseData::ContentBlockStart { index, content_block } => {
                    match content_block {
                        AnthropicSseBlockStart::Text { text } => {
                            accumulators.insert(index, BlockAccumulator::Text(text));
                        }
                        AnthropicSseBlockStart::ToolUse { id, name, .. } => {
                            let started = StreamEvent::ToolUseStart {
                                id: id.clone(),
                                name: name.clone(),
                            };
                            accumulators.insert(
                                index,
                                BlockAccumulator::ToolUse { id, name, json_buf: String::new() },
                            );
                            yield started;
                        }
                        AnthropicSseBlockStart::Thinking { thinking, signature } => {
                            accumulators.insert(
                                index,
                                BlockAccumulator::Thinking {
                                    thinking,
                                    signature: signature.unwrap_or_default(),
                                },
                            );
                        }
                        AnthropicSseBlockStart::RedactedThinking { data } => {
                            accumulators.insert(index, BlockAccumulator::RedactedThinking { data });
                        }
                    }
                }
                AnthropicSseData::ContentBlockDelta { index, delta } => {
                    let acc = accumulators
                        .get_mut(&index)
                        .with_context(|| format!("content_block_delta for unknown index {index}"))?;
                    match (delta, acc) {
                        (AnthropicSseDelta::TextDelta { text }, BlockAccumulator::Text(buf)) => {
                            buf.push_str(&text);
                            yield StreamEvent::TextDelta(text);
                        }
                        (
                            AnthropicSseDelta::InputJsonDelta { partial_json },
                            BlockAccumulator::ToolUse { id, json_buf, .. },
                        ) => {
                            json_buf.push_str(&partial_json);
                            yield StreamEvent::ToolUseDelta {
                                id: id.clone(),
                                partial_json,
                            };
                        }
                        (
                            AnthropicSseDelta::ThinkingDelta { thinking },
                            BlockAccumulator::Thinking { thinking: buf, .. },
                        ) => {
                            buf.push_str(&thinking);
                            yield StreamEvent::ThinkingDelta(thinking);
                        }
                        (
                            AnthropicSseDelta::SignatureDelta { signature },
                            BlockAccumulator::Thinking { signature: sig_buf, .. },
                        ) => {
                            sig_buf.push_str(&signature);
                        }
                        _ => {
                            Err::<(), anyhow::Error>(anyhow::anyhow!(
                                "content_block_delta type mismatch at index {index}"
                            ))?;
                        }
                    }
                }
                AnthropicSseData::ContentBlockStop { index } => {
                    if let Some(BlockAccumulator::ToolUse { id, .. }) = accumulators.get(&index) {
                        yield StreamEvent::ToolUseStop { id: id.clone() };
                    }
                }
                AnthropicSseData::MessageDelta { delta, usage: u } => {
                    if let Some(sr) = delta.stop_reason {
                        stop_reason = Some(match sr.as_str() {
                            "end_turn" => StopReason::EndTurn,
                            "tool_use" => StopReason::ToolUse,
                            "max_tokens" => StopReason::MaxTokens,
                            other => StopReason::Other(other.to_string()),
                        });
                    }
                    if let Some(u) = u {
                        let entry = usage.get_or_insert_with(TokenUsage::default);
                        if let Some(o) = u.output_tokens {
                            entry.output = o;
                        }
                        if u.cache_creation_input_tokens.is_some() {
                            entry.cache_creation_input = u.cache_creation_input_tokens;
                        }
                        if u.cache_read_input_tokens.is_some() {
                            entry.cache_read_input = u.cache_read_input_tokens;
                        }
                    }
                }
                AnthropicSseData::MessageStop => {
                    let mut full_content = Vec::with_capacity(accumulators.len());
                    for (_, acc) in std::mem::take(&mut accumulators) {
                        full_content.push(acc.finalize()?);
                    }
                    let final_stop = stop_reason
                        .take()
                        .unwrap_or_else(|| StopReason::Other("missing_stop_reason".into()));
                    yield StreamEvent::Done {
                        stop_reason: final_stop,
                        usage: usage.take(),
                        full_content,
                    };
                    return;
                }
                AnthropicSseData::Ping => {}
                AnthropicSseData::Error { error } => {
                    Err::<(), anyhow::Error>(anyhow::anyhow!(
                        "anthropic stream error: {} ({})",
                        error.message,
                        error.kind
                    ))?;
                }
            }
        }

        // Loop exited without hitting MessageStop (which would `return`),
        // so the upstream ended prematurely.
        Err::<(), anyhow::Error>(anyhow::anyhow!(
            "anthropic stream ended without message_stop"
        ))?;
    }
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
    #[serde(skip_serializing_if = "Option::is_none")]
    stream: Option<bool>,
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
    #[serde(default)]
    usage: Option<AnthropicUsage>,
}

#[derive(Debug, Deserialize)]
struct AnthropicUsage {
    #[serde(default)]
    input_tokens: Option<u32>,
    #[serde(default)]
    output_tokens: Option<u32>,
    #[serde(default)]
    cache_creation_input_tokens: Option<u32>,
    #[serde(default)]
    cache_read_input_tokens: Option<u32>,
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
        usage: resp.usage.map(|u| TokenUsage {
            input: u.input_tokens.unwrap_or(0),
            output: u.output_tokens.unwrap_or(0),
            cache_creation_input: u.cache_creation_input_tokens,
            cache_read_input: u.cache_read_input_tokens,
        }),
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
    fn to_wire_tools_drops_requires_approval() {
        // requires_approval is mcp-master-internal metadata; the wire-payload
        // sent to Anthropic must contain only name/description/input_schema.
        let specs = vec![ToolSpec {
            name: "delete_company".into(),
            description: "Soft-delete an Account.".into(),
            input_schema: json!({"type": "object"}),
            requires_approval: true,
        }];
        let wire = to_wire_tools(&specs);
        let value = serde_json::to_value(&wire).unwrap();
        let obj = value.as_array().unwrap()[0].as_object().unwrap();
        assert!(obj.contains_key("name"));
        assert!(obj.contains_key("description"));
        assert!(obj.contains_key("input_schema"));
        assert!(
            !obj.contains_key("requires_approval"),
            "requires_approval must NOT leak to Anthropic wire-payload",
        );
    }

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
            usage: None,
        };
        let mapped = from_wire_response(happy);
        assert_eq!(mapped.stop_reason, StopReason::EndTurn);

        let tu = AnthropicResponse {
            content: vec![],
            stop_reason: "tool_use".into(),
            usage: None,
        };
        assert_eq!(from_wire_response(tu).stop_reason, StopReason::ToolUse);

        let mt = AnthropicResponse {
            content: vec![],
            stop_reason: "max_tokens".into(),
            usage: None,
        };
        assert_eq!(from_wire_response(mt).stop_reason, StopReason::MaxTokens);

        let unknown = AnthropicResponse {
            content: vec![],
            stop_reason: "pause_turn".into(),
            usage: None,
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
    async fn chat_decodes_usage_field() {
        let server = MockServer::start().await;
        Mock::given(method("POST"))
            .and(path("/v1/messages"))
            .respond_with(ResponseTemplate::new(200).set_body_json(json!({
                "content": [{ "type": "text", "text": "ok" }],
                "stop_reason": "end_turn",
                "usage": {
                    "input_tokens": 100,
                    "output_tokens": 50,
                    "cache_creation_input_tokens": 12,
                    "cache_read_input_tokens": 8
                }
            })))
            .mount(&server)
            .await;

        let client = AnthropicClient::new("k".into()).with_base_url(server.uri());
        let resp = client.chat("sys", &[], &[], 4096).await.unwrap();

        let usage = resp.usage.expect("usage should be Some");
        assert_eq!(usage.input, 100);
        assert_eq!(usage.output, 50);
        assert_eq!(usage.cache_creation_input, Some(12));
        assert_eq!(usage.cache_read_input, Some(8));
    }

    #[tokio::test]
    async fn chat_handles_missing_usage_field() {
        let server = MockServer::start().await;
        Mock::given(method("POST"))
            .and(path("/v1/messages"))
            .respond_with(ResponseTemplate::new(200).set_body_json(json!({
                "content": [{ "type": "text", "text": "ok" }],
                "stop_reason": "end_turn"
            })))
            .mount(&server)
            .await;

        let client = AnthropicClient::new("k".into()).with_base_url(server.uri());
        let resp = client.chat("sys", &[], &[], 4096).await.unwrap();

        assert_eq!(resp.usage, None);
    }

    #[tokio::test]
    async fn stream_chat_text_delta_path() {
        use futures_util::StreamExt;
        let server = MockServer::start().await;
        let sse_body = concat!(
            "event: message_start\n",
            "data: {\"type\":\"message_start\",\"message\":{\"id\":\"msg_1\",\"usage\":{\"input_tokens\":5,\"output_tokens\":0}}}\n\n",
            "event: content_block_start\n",
            "data: {\"type\":\"content_block_start\",\"index\":0,\"content_block\":{\"type\":\"text\",\"text\":\"\"}}\n\n",
            "event: ping\n",
            "data: {\"type\":\"ping\"}\n\n",
            "event: content_block_delta\n",
            "data: {\"type\":\"content_block_delta\",\"index\":0,\"delta\":{\"type\":\"text_delta\",\"text\":\"Hello\"}}\n\n",
            "event: content_block_delta\n",
            "data: {\"type\":\"content_block_delta\",\"index\":0,\"delta\":{\"type\":\"text_delta\",\"text\":\" world\"}}\n\n",
            "event: content_block_stop\n",
            "data: {\"type\":\"content_block_stop\",\"index\":0}\n\n",
            "event: message_delta\n",
            "data: {\"type\":\"message_delta\",\"delta\":{\"stop_reason\":\"end_turn\"},\"usage\":{\"output_tokens\":3}}\n\n",
            "event: message_stop\n",
            "data: {\"type\":\"message_stop\"}\n\n",
        );
        Mock::given(method("POST"))
            .and(path("/v1/messages"))
            .respond_with(
                ResponseTemplate::new(200)
                    .insert_header("content-type", "text/event-stream")
                    .set_body_string(sse_body),
            )
            .mount(&server)
            .await;

        let client = AnthropicClient::new("k".into()).with_base_url(server.uri());
        let mut stream = client.stream_chat("sys", &[], &[], 4096).await.unwrap();
        let mut events = Vec::new();
        while let Some(e) = stream.next().await {
            events.push(e.unwrap());
        }
        assert_eq!(events.len(), 3, "got: {events:#?}");
        match &events[0] {
            StreamEvent::TextDelta(t) => assert_eq!(t, "Hello"),
            other => panic!("expected TextDelta, got {other:?}"),
        }
        match &events[1] {
            StreamEvent::TextDelta(t) => assert_eq!(t, " world"),
            other => panic!("expected TextDelta, got {other:?}"),
        }
        match &events[2] {
            StreamEvent::Done {
                stop_reason,
                usage,
                full_content,
            } => {
                assert_eq!(*stop_reason, StopReason::EndTurn);
                let u = usage.as_ref().expect("usage on Done");
                assert_eq!(u.input, 5);
                assert_eq!(u.output, 3);
                assert_eq!(
                    full_content,
                    &vec![ContentBlock::Text {
                        text: "Hello world".into()
                    }]
                );
            }
            other => panic!("expected Done, got {other:?}"),
        }
    }

    #[tokio::test]
    async fn stream_chat_reassembles_input_json_delta_fragments() {
        use futures_util::StreamExt;
        let server = MockServer::start().await;
        let sse_body = concat!(
            "event: message_start\n",
            "data: {\"type\":\"message_start\",\"message\":{}}\n\n",
            "event: content_block_start\n",
            "data: {\"type\":\"content_block_start\",\"index\":0,\"content_block\":{\"type\":\"tool_use\",\"id\":\"toolu_42\",\"name\":\"heartbeat_status\",\"input\":{}}}\n\n",
            "event: content_block_delta\n",
            "data: {\"type\":\"content_block_delta\",\"index\":0,\"delta\":{\"type\":\"input_json_delta\",\"partial_json\":\"{\\\"lim\"}}\n\n",
            "event: content_block_delta\n",
            "data: {\"type\":\"content_block_delta\",\"index\":0,\"delta\":{\"type\":\"input_json_delta\",\"partial_json\":\"it\\\":5}\"}}\n\n",
            "event: content_block_stop\n",
            "data: {\"type\":\"content_block_stop\",\"index\":0}\n\n",
            "event: message_delta\n",
            "data: {\"type\":\"message_delta\",\"delta\":{\"stop_reason\":\"tool_use\"}}\n\n",
            "event: message_stop\n",
            "data: {\"type\":\"message_stop\"}\n\n",
        );
        Mock::given(method("POST"))
            .and(path("/v1/messages"))
            .respond_with(
                ResponseTemplate::new(200)
                    .insert_header("content-type", "text/event-stream")
                    .set_body_string(sse_body),
            )
            .mount(&server)
            .await;

        let client = AnthropicClient::new("k".into()).with_base_url(server.uri());
        let mut stream = client.stream_chat("sys", &[], &[], 4096).await.unwrap();
        let mut events = Vec::new();
        while let Some(e) = stream.next().await {
            events.push(e.unwrap());
        }

        match &events[0] {
            StreamEvent::ToolUseStart { id, name } => {
                assert_eq!(id, "toolu_42");
                assert_eq!(name, "heartbeat_status");
            }
            other => panic!("expected ToolUseStart, got {other:?}"),
        }
        // ToolUseDelta fragments are forwarded individually...
        assert!(matches!(events[1], StreamEvent::ToolUseDelta { .. }));
        assert!(matches!(events[2], StreamEvent::ToolUseDelta { .. }));
        assert!(matches!(events[3], StreamEvent::ToolUseStop { .. }));
        // ...but Done.full_content has the parsed result, not the raw concat.
        match &events[4] {
            StreamEvent::Done {
                stop_reason,
                full_content,
                ..
            } => {
                assert_eq!(*stop_reason, StopReason::ToolUse);
                assert_eq!(full_content.len(), 1);
                match &full_content[0] {
                    ContentBlock::ToolUse { id, name, input } => {
                        assert_eq!(id, "toolu_42");
                        assert_eq!(name, "heartbeat_status");
                        assert_eq!(input, &json!({"limit": 5}));
                    }
                    other => panic!("expected ToolUse, got {other:?}"),
                }
            }
            other => panic!("expected Done, got {other:?}"),
        }
    }

    #[tokio::test]
    async fn stream_chat_preserves_thinking_signature_byte_for_byte() {
        use futures_util::StreamExt;
        let server = MockServer::start().await;
        let sse_body = concat!(
            "event: message_start\n",
            "data: {\"type\":\"message_start\",\"message\":{}}\n\n",
            "event: content_block_start\n",
            "data: {\"type\":\"content_block_start\",\"index\":0,\"content_block\":{\"type\":\"thinking\",\"thinking\":\"\"}}\n\n",
            "event: content_block_delta\n",
            "data: {\"type\":\"content_block_delta\",\"index\":0,\"delta\":{\"type\":\"thinking_delta\",\"thinking\":\"Let me\"}}\n\n",
            "event: content_block_delta\n",
            "data: {\"type\":\"content_block_delta\",\"index\":0,\"delta\":{\"type\":\"thinking_delta\",\"thinking\":\" consider.\"}}\n\n",
            "event: content_block_delta\n",
            "data: {\"type\":\"content_block_delta\",\"index\":0,\"delta\":{\"type\":\"signature_delta\",\"signature\":\"sig_part_one_\"}}\n\n",
            "event: content_block_delta\n",
            "data: {\"type\":\"content_block_delta\",\"index\":0,\"delta\":{\"type\":\"signature_delta\",\"signature\":\"sig_part_two\"}}\n\n",
            "event: content_block_stop\n",
            "data: {\"type\":\"content_block_stop\",\"index\":0}\n\n",
            "event: content_block_start\n",
            "data: {\"type\":\"content_block_start\",\"index\":1,\"content_block\":{\"type\":\"text\",\"text\":\"\"}}\n\n",
            "event: content_block_delta\n",
            "data: {\"type\":\"content_block_delta\",\"index\":1,\"delta\":{\"type\":\"text_delta\",\"text\":\"OK.\"}}\n\n",
            "event: content_block_stop\n",
            "data: {\"type\":\"content_block_stop\",\"index\":1}\n\n",
            "event: message_delta\n",
            "data: {\"type\":\"message_delta\",\"delta\":{\"stop_reason\":\"end_turn\"}}\n\n",
            "event: message_stop\n",
            "data: {\"type\":\"message_stop\"}\n\n",
        );
        Mock::given(method("POST"))
            .and(path("/v1/messages"))
            .respond_with(
                ResponseTemplate::new(200)
                    .insert_header("content-type", "text/event-stream")
                    .set_body_string(sse_body),
            )
            .mount(&server)
            .await;

        let client = AnthropicClient::new("k".into()).with_base_url(server.uri());
        let mut stream = client.stream_chat("sys", &[], &[], 4096).await.unwrap();
        let mut events = Vec::new();
        while let Some(e) = stream.next().await {
            events.push(e.unwrap());
        }
        let done = events.last().expect("at least one event");
        match done {
            StreamEvent::Done { full_content, .. } => {
                assert_eq!(full_content.len(), 2);
                match &full_content[0] {
                    ContentBlock::Thinking {
                        thinking,
                        signature,
                    } => {
                        assert_eq!(thinking, "Let me consider.");
                        // Byte-for-byte signature reconstruction is the critical
                        // invariant: Anthropic rejects the next call otherwise.
                        assert_eq!(signature, "sig_part_one_sig_part_two");
                    }
                    other => panic!("expected Thinking, got {other:?}"),
                }
                match &full_content[1] {
                    ContentBlock::Text { text } => assert_eq!(text, "OK."),
                    other => panic!("expected Text, got {other:?}"),
                }
            }
            other => panic!("expected Done, got {other:?}"),
        }
    }

    #[tokio::test]
    async fn stream_chat_retries_on_429_then_succeeds() {
        use futures_util::StreamExt;
        let server = MockServer::start().await;
        Mock::given(method("POST"))
            .and(path("/v1/messages"))
            .respond_with(ResponseTemplate::new(429).set_body_string("rate limited"))
            .up_to_n_times(1)
            .mount(&server)
            .await;
        let sse_body = concat!(
            "event: message_start\n",
            "data: {\"type\":\"message_start\",\"message\":{}}\n\n",
            "event: message_delta\n",
            "data: {\"type\":\"message_delta\",\"delta\":{\"stop_reason\":\"end_turn\"}}\n\n",
            "event: message_stop\n",
            "data: {\"type\":\"message_stop\"}\n\n",
        );
        Mock::given(method("POST"))
            .and(path("/v1/messages"))
            .respond_with(
                ResponseTemplate::new(200)
                    .insert_header("content-type", "text/event-stream")
                    .set_body_string(sse_body),
            )
            .mount(&server)
            .await;

        let client = AnthropicClient::new("k".into()).with_base_url(server.uri());
        let mut stream = client.stream_chat("sys", &[], &[], 4096).await.unwrap();
        let mut events = Vec::new();
        while let Some(e) = stream.next().await {
            events.push(e.unwrap());
        }
        assert!(matches!(events.last(), Some(StreamEvent::Done { .. })));
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
