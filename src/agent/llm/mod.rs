//! LLM provider abstraction.
//!
//! Defines provider-agnostic types and the `LlmClient` trait. Concrete
//! providers (Anthropic now, OpenAI/Ollama later) live in submodules and
//! translate between the wire-shape they speak and the types here.
//!
//! The content-block shape mirrors Anthropic's wire format on purpose: a
//! second provider's translation layer becomes a rename + repackage rather
//! than a structural rewrite.

pub mod anthropic;

use futures_util::stream::{self, BoxStream};
use serde::Serialize;
use serde_json::Value;

/// Conversation turn role. `system` is not modelled here because it is a
/// top-level field on the chat request, not a turn.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Role {
    User,
    Assistant,
}

/// One conversation turn.
#[derive(Debug, Clone, PartialEq)]
pub struct Message {
    pub role: Role,
    pub content: Vec<ContentBlock>,
}

/// One block within a turn. Mirrors the Anthropic content-block taxonomy
/// so provider translation is repackage, not compute.
#[derive(Debug, Clone, PartialEq)]
pub enum ContentBlock {
    /// Plain text from user or assistant.
    Text { text: String },
    /// Assistant requesting a tool call.
    ToolUse {
        id: String,
        name: String,
        input: Value,
    },
    /// User-side reply to a previous `ToolUse` carrying the tool result.
    ToolResult {
        tool_use_id: String,
        content: String,
        is_error: bool,
    },
    /// Extended-thinking reasoning emitted by Anthropic before the visible
    /// response. Must be preserved verbatim in subsequent calls so the
    /// signature verifies. Other providers never produce these.
    Thinking { thinking: String, signature: String },
    /// Encrypted thinking block — content is opaque to us, must be
    /// passed through unchanged to subsequent calls.
    RedactedThinking { data: String },
}

/// Tool schema as advertised to the LLM. `description` is mandatory because
/// the model's tool selection accuracy depends heavily on it.
///
/// `requires_approval` is mcp-master-internal metadata; it is **not** sent to
/// the LLM (verified in `anthropic::tests::to_wire_tools_drops_requires_approval`).
/// The orchestrator (PR-3) reads it to decide whether to dispatch directly
/// (false) or route through the approval store (true).
#[derive(Debug, Clone, PartialEq)]
pub struct ToolSpec {
    pub name: String,
    pub description: String,
    pub input_schema: Value,
    pub requires_approval: bool,
}

/// Result of one `chat()` round-trip.
#[derive(Debug, Clone, PartialEq)]
pub struct ChatResponse {
    pub content: Vec<ContentBlock>,
    pub stop_reason: StopReason,
    /// Token counts for billing/observability. `None` when the provider
    /// doesn't report (e.g. a future Ollama impl). Cache fields stay `Option`
    /// so adding more later is non-breaking for downstream pattern-matchers.
    pub usage: Option<TokenUsage>,
}

/// Per-request token counts reported by the provider.
///
/// Cache fields are `Option` so providers without prompt-caching (and clients
/// who don't care) can ignore them without ABI churn. `add()` is used by the
/// orchestrator to sum across tool-loop iterations.
#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize)]
pub struct TokenUsage {
    pub input: u32,
    pub output: u32,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub cache_creation_input: Option<u32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub cache_read_input: Option<u32>,
}

impl TokenUsage {
    pub fn add(&mut self, other: &TokenUsage) {
        self.input += other.input;
        self.output += other.output;
        self.cache_creation_input =
            sum_optional(self.cache_creation_input, other.cache_creation_input);
        self.cache_read_input = sum_optional(self.cache_read_input, other.cache_read_input);
    }
}

fn sum_optional(a: Option<u32>, b: Option<u32>) -> Option<u32> {
    match (a, b) {
        (None, None) => None,
        (a, b) => Some(a.unwrap_or(0) + b.unwrap_or(0)),
    }
}

/// Why the LLM stopped generating. `Other` is a forward-compat escape valve
/// so a new provider-defined stop reason does not panic the orchestrator.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum StopReason {
    EndTurn,
    ToolUse,
    MaxTokens,
    Other(String),
}

/// Provider-agnostic streaming event yielded by `LlmClient::stream_chat`.
///
/// Mirrors the conceptual deltas of Anthropic's SSE format but stays
/// vendor-neutral so OpenAI/Ollama impls can map their own protocols here.
/// The terminal `Done` event carries the *reassembled* `full_content`
/// — the orchestrator feeds it back into the next iteration unchanged,
/// preserving `Thinking { signature }` byte-for-byte (Anthropic rejects
/// the next call otherwise) and the parsed `ToolUse.input` JSON.
#[allow(dead_code)] // reachable via orchestrator streaming variant (PR3 in this stack)
#[derive(Debug, Clone, PartialEq)]
pub enum StreamEvent {
    /// Incremental text-block bytes from the assistant's visible answer.
    TextDelta(String),
    /// Assistant has begun a `ToolUse` block; subsequent `ToolUseDelta`
    /// fragments belong to this `id` until `ToolUseStop`.
    ToolUseStart {
        id: String,
        name: String,
    },
    /// One `input_json_delta`-style fragment. Cannot be parsed in isolation;
    /// consumers must accumulate per `id` and parse once on `ToolUseStop`.
    ToolUseDelta {
        id: String,
        partial_json: String,
    },
    ToolUseStop {
        id: String,
    },
    /// Incremental extended-thinking text. Signature arrives separately and
    /// only ends up in `Done.full_content`'s `Thinking { signature }` block.
    ThinkingDelta(String),
    /// Terminal event. Always emitted exactly once at the end of a successful
    /// stream. `full_content` is the orchestrator-feedable assistant turn
    /// reconstructed from the deltas plus any non-streamed blocks
    /// (e.g. `RedactedThinking`).
    Done {
        stop_reason: StopReason,
        usage: Option<TokenUsage>,
        full_content: Vec<ContentBlock>,
    },
}

/// Provider-agnostic LLM client.
///
/// `Send + Sync` so trait objects (`&dyn LlmClient`) can cross `.await`
/// points and be shared across tasks. `max_tokens` is explicit and
/// mandatory on every call (cost + runaway prevention). Borrows the
/// conversation history — the orchestrator owns it, providers only read.
///
/// Uses `async-trait` rather than native `async fn in trait` because
/// `&dyn LlmClient` against native async-fn-in-trait still has unresolved
/// `Send` bound friction on stable Rust as of 1.85. `async-trait` desugars
/// to `Box<dyn Future + Send>` which sidesteps that cleanly.
#[async_trait::async_trait]
pub trait LlmClient: Send + Sync {
    async fn chat(
        &self,
        system: &str,
        messages: &[Message],
        tools: &[ToolSpec],
        max_tokens: u32,
    ) -> anyhow::Result<ChatResponse>;

    /// Streaming counterpart of `chat`. Returns a boxed `Stream` so the
    /// trait remains object-safe behind `&dyn LlmClient` and the returned
    /// stream can be moved across `tokio::spawn`.
    ///
    /// The default implementation drains `chat` and emits a single terminal
    /// `Done` event — providers without native streaming (e.g. a future
    /// Ollama impl) opt into this path for free. Providers that do speak
    /// SSE (Anthropic) override to forward token-level deltas.
    #[allow(dead_code)] // reachable via orchestrator streaming variant (PR3 in this stack)
    async fn stream_chat(
        &self,
        system: &str,
        messages: &[Message],
        tools: &[ToolSpec],
        max_tokens: u32,
    ) -> anyhow::Result<BoxStream<'static, anyhow::Result<StreamEvent>>> {
        let resp = self.chat(system, messages, tools, max_tokens).await?;
        let event = StreamEvent::Done {
            stop_reason: resp.stop_reason,
            usage: resp.usage,
            full_content: resp.content,
        };
        Ok(Box::pin(stream::once(async move { Ok(event) })))
    }
}

#[cfg(test)]
pub mod tests;
