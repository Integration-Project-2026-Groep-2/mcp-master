//! Thin wrapper around rmcp: open a Streamable-HTTP session, translate
//! rmcp `Tool`s into the LLM-facing `ToolSpec`, expose a production
//! `McpExecutor` impl that delegates to `RunningService::call_tool`.

use anyhow::Context;
use async_trait::async_trait;
use rmcp::{
    ServiceExt,
    model::{
        CallToolRequestParams, CallToolResult, ClientCapabilities, ClientInfo, Implementation,
        RawContent, Tool,
    },
    service::{RoleClient, RunningService},
    transport::StreamableHttpClientTransport,
};
use serde_json::Value;

use crate::llm::ToolSpec;
use crate::orchestrator::McpExecutor;

/// Open a Streamable-HTTP MCP session against `<base_url>:<port>/mcp`.
pub async fn connect(
    base_url: &str,
    port: &str,
) -> anyhow::Result<RunningService<RoleClient, ClientInfo>> {
    let url = format!("{base_url}:{port}/mcp");
    tracing::info!(%url, "connecting to MCP server");

    let transport = StreamableHttpClientTransport::from_uri(url);
    let client_info = ClientInfo::new(
        ClientCapabilities::default(),
        Implementation::new("mcp-master", env!("CARGO_PKG_VERSION")),
    );
    let svc = client_info
        .serve(transport)
        .await
        .context("MCP initialize handshake failed")?;
    tracing::info!(server = ?svc.peer_info(), "MCP session initialized");
    Ok(svc)
}

/// Translate an rmcp `Tool` into the provider-agnostic `ToolSpec`.
///
/// `description` is mandatory in our model — empty string if rmcp gave
/// `None`, because Claude's tool selection accuracy depends on prose.
pub fn tool_to_spec(tool: &Tool) -> ToolSpec {
    ToolSpec {
        name: tool.name.to_string(),
        description: tool.description.as_deref().unwrap_or("").to_string(),
        input_schema: Value::Object((*tool.input_schema).clone()),
    }
}

/// Flatten the rmcp content vector into a single string for the LLM.
///
/// Rules: concat `Text` blocks with `\n`, warn on non-text blocks (image /
/// resource / etc.), prefix `TOOL_ERROR: ` if `is_error` so the model sees
/// a recoverable error signal it can react to. Falls back to a JSON
/// serialisation of `structured_content` if `content` is empty.
//
// TODO(future): wrap the call site in a moka cache (TTL 30s,
// key=name+sorted(args)). Deferred to a follow-up branch.
pub fn extract_tool_result_text(result: &CallToolResult) -> String {
    let mut buf = String::new();

    for block in &result.content {
        match &block.raw {
            RawContent::Text(t) => {
                if !buf.is_empty() {
                    buf.push('\n');
                }
                buf.push_str(&t.text);
            }
            other => {
                tracing::warn!(?other, "non-text content block ignored");
            }
        }
    }

    if buf.is_empty()
        && let Some(structured) = &result.structured_content
    {
        buf = serde_json::to_string(structured).unwrap_or_default();
    }

    if result.is_error == Some(true) {
        format!("TOOL_ERROR: {buf}")
    } else {
        buf
    }
}

/// Production `McpExecutor`: wraps a live rmcp session and dispatches
/// tool calls. Owns the session — call `shutdown` for a clean disconnect.
pub struct McpClient {
    svc: RunningService<RoleClient, ClientInfo>,
}

impl McpClient {
    pub fn new(svc: RunningService<RoleClient, ClientInfo>) -> Self {
        Self { svc }
    }

    /// Send the MCP DELETE-session shutdown. Consumes self because after
    /// shutdown the session is gone — no further calls valid. Avoids the
    /// `Drop`-for-async pain point.
    pub async fn shutdown(self) -> anyhow::Result<()> {
        self.svc.cancel().await?;
        Ok(())
    }
}

#[async_trait]
impl McpExecutor for McpClient {
    async fn call(&self, name: &str, arguments: Value) -> anyhow::Result<String> {
        let map = arguments
            .as_object()
            .cloned()
            .context("MCP tool arguments must be a JSON object")?;
        let res = self
            .svc
            .call_tool(CallToolRequestParams::new(name.to_string()).with_arguments(map))
            .await
            .with_context(|| format!("MCP tools/call failed for {name}"))?;
        Ok(extract_tool_result_text(&res))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use rmcp::model::Tool;
    use serde_json::json;
    use std::sync::Arc;

    #[test]
    fn tool_to_spec_translates_name_description_and_schema() {
        let schema_obj = json!({
            "type": "object",
            "properties": { "limit": { "type": "number" } },
            "required": []
        })
        .as_object()
        .cloned()
        .unwrap();

        // `rmcp::model::Tool` is `#[non_exhaustive]`; build via Default + mutate.
        let mut tool = Tool::default();
        tool.name = "heartbeat_status".into();
        tool.description = Some("Recent heartbeats".into());
        tool.input_schema = Arc::new(schema_obj.clone());

        let spec = tool_to_spec(&tool);

        assert_eq!(spec.name, "heartbeat_status");
        assert_eq!(spec.description, "Recent heartbeats");
        assert_eq!(spec.input_schema, Value::Object(schema_obj));
    }

    #[test]
    fn tool_to_spec_uses_empty_description_when_none() {
        let mut tool = Tool::default();
        tool.name = "anon".into();
        tool.description = None;
        tool.input_schema = Arc::new(serde_json::Map::new());
        let spec = tool_to_spec(&tool);
        assert_eq!(spec.description, "");
    }
}
