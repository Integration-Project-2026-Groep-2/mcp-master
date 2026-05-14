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

        // Loop exited without hitting MessageStop. If any block accumulated
        // content, emit a partial `Done` with stop_reason=Other("premature_close")
        // so the client keeps the 95% answer they already saw via TextDelta
        // events instead of getting a hard error. Partial tool_use blocks
        // whose json_buf is mid-token are dropped — the orchestrator won't
        // iterate (terminal Done), so a malformed tool_use never reaches
        // Anthropic on a follow-up call.
        if !accumulators.is_empty() {
            let mut full_content = Vec::with_capacity(accumulators.len());
            for (_, acc) in std::mem::take(&mut accumulators) {
                match acc.finalize() {
                    Ok(block) => full_content.push(block),
                    Err(e) => {
                        tracing::warn!(error = ?e, "dropping partial block on premature close")
                    }
                }
            }
            yield StreamEvent::Done {
                stop_reason: StopReason::Other("premature_close".into()),
                usage: usage.take(),
                full_content,
            };
            return;
        }

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
mod tests;
