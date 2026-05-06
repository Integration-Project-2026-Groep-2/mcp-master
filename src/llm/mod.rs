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
#[derive(Debug, Clone, PartialEq)]
pub struct ToolSpec {
    pub name: String,
    pub description: String,
    pub input_schema: Value,
}

/// Result of one `chat()` round-trip.
#[derive(Debug, Clone, PartialEq)]
pub struct ChatResponse {
    pub content: Vec<ContentBlock>,
    pub stop_reason: StopReason,
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
}

#[cfg(test)]
pub mod tests {
    use super::*;
    use tokio::sync::Mutex;

    /// Captured per `chat()` invocation.
    #[derive(Debug, Clone)]
    pub struct MockCall {
        /// Reserved for tests that want to assert on the system prompt.
        #[allow(dead_code)]
        pub system: String,
        pub messages: Vec<Message>,
    }

    /// Test double: pops responses from a queue in order, records calls.
    pub struct MockLlmClient {
        responses: Mutex<Vec<ChatResponse>>,
        calls: Mutex<Vec<MockCall>>,
    }

    impl MockLlmClient {
        pub fn new(responses: Vec<ChatResponse>) -> Self {
            Self {
                responses: Mutex::new(responses),
                calls: Mutex::new(Vec::new()),
            }
        }

        pub async fn calls(&self) -> Vec<MockCall> {
            self.calls.lock().await.clone()
        }
    }

    #[async_trait::async_trait]
    impl LlmClient for MockLlmClient {
        async fn chat(
            &self,
            system: &str,
            messages: &[Message],
            _tools: &[ToolSpec],
            _max_tokens: u32,
        ) -> anyhow::Result<ChatResponse> {
            self.calls.lock().await.push(MockCall {
                system: system.to_string(),
                messages: messages.to_vec(),
            });
            let mut q = self.responses.lock().await;
            if q.is_empty() {
                anyhow::bail!("MockLlmClient: response queue exhausted");
            }
            Ok(q.remove(0))
        }
    }

    // Orchestrator tests live next to the orchestrator (in src/orchestrator.rs)
    // because they need the test-only `TestExecutor` from that module. This
    // module's job is the trait + types + the MockLlmClient test double.
}
