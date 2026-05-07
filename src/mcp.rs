//! Thin wrappers around rmcp:
//! - [`open_session`]: open one Streamable-HTTP MCP session.
//! - [`McpPool`]: multi-server `McpExecutor` with tool-name routing —
//!   used by `main.rs` (a single-element pool covers the single-server case).
//! - [`tool_to_spec`], [`extract_tool_result_text`]: protocol helpers.
//! - [`parse_endpoints`]: parse the `MCP_SERVERS` env-var value.

use std::collections::HashMap;
use std::sync::Arc;

use anyhow::{Context, bail};
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
use crate::rabbitmq::publisher::Publisher;

/// Open a Streamable-HTTP MCP session against the given URL.
///
/// URL must include the full path, e.g. `http://localhost:7001/mcp`.
/// Used internally by `McpPool::connect` per endpoint.
pub async fn open_session(url: &str) -> anyhow::Result<RunningService<RoleClient, ClientInfo>> {
    tracing::info!(%url, "connecting to MCP server");

    let transport = StreamableHttpClientTransport::from_uri(url.to_string());
    let client_info = ClientInfo::new(
        ClientCapabilities::default(),
        Implementation::new("mcp-master", env!("CARGO_PKG_VERSION")),
    );
    let svc = client_info
        .serve(transport)
        .await
        .with_context(|| format!("MCP initialize handshake failed for {url}"))?;
    tracing::info!(%url, server = ?svc.peer_info(), "MCP session initialized");
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

/// Parse the `MCP_SERVERS` env-var value into `(label, url)` pairs.
///
/// Format: `label@url,label@url,...` — `,` separates entries, `@` separates
/// label from url within an entry. Whitespace around items is trimmed.
/// Malformed entries (missing `@`, empty label, empty url) are silently
/// skipped — the caller decides how to react to an empty result.
pub fn parse_endpoints(value: &str) -> Vec<(String, String)> {
    value
        .split(',')
        .filter_map(|s| s.trim().split_once('@'))
        .map(|(label, url)| (label.trim().to_string(), url.trim().to_string()))
        .filter(|(l, u)| !l.is_empty() && !u.is_empty())
        .collect()
}

/// Build the `(tool_specs, tool_to_session_idx)` pair from per-server
/// tool-lists.
///
/// `server_tools[idx].0` is the server's label, `[idx].1` its tool list.
/// The returned map sends a tool-name to the `idx` of the server that
/// provides it.
///
/// Errors if two servers expose the same tool-name. Today no collisions
/// exist (CRM + Controlroom tool-names are disjoint); we fail fast so
/// future overlap is caught at startup rather than silently picking one
/// server's version.
fn build_routing_table(
    server_tools: &[(String, Vec<Tool>)],
) -> anyhow::Result<(Vec<ToolSpec>, HashMap<String, usize>)> {
    let mut specs = Vec::new();
    let mut idx_map: HashMap<String, usize> = HashMap::new();
    let mut owner_map: HashMap<String, String> = HashMap::new();

    for (idx, (label, tools)) in server_tools.iter().enumerate() {
        for tool in tools {
            let name = tool.name.to_string();
            if let Some(prior) = owner_map.get(&name) {
                bail!(
                    "tool name collision: '{}' provided by both '{}' and '{}' \
                     — rename one (mcp-master refuses to choose arbitrarily)",
                    name,
                    prior,
                    label,
                );
            }
            owner_map.insert(name.clone(), label.clone());
            idx_map.insert(name, idx);
            specs.push(tool_to_spec(tool));
        }
    }

    Ok((specs, idx_map))
}

/// Pool of MCP-server sessions with tool-name routing.
///
/// Construct via [`McpPool::connect`]. Sessions are owned by the pool
/// (not `Arc`-shared) so [`McpPool::shutdown`] can `cancel()` each cleanly
/// without needing extraction from `Arc`.
pub struct McpPool {
    /// `(label, session)` pairs in connection-order. Index into this `Vec`
    /// is the stable identity used by `tool_to_session_idx`.
    sessions: Vec<(String, RunningService<RoleClient, ClientInfo>)>,

    /// Tool-name → index in `sessions`. Built once at startup from each
    /// server's `tools/list`. O(1) dispatch lookup.
    tool_to_session_idx: HashMap<String, usize>,

    /// Aggregate `ToolSpec`s across all connected servers. Cached so the
    /// orchestrator can borrow without re-querying.
    tool_specs: Vec<ToolSpec>,

    /// Optional event-sink for `tool_called` events on `ai.events`. Set via
    /// `attach_publisher`; absent means no events fired (skip-warn pattern).
    publisher: Option<Arc<Publisher>>,
}

impl McpPool {
    /// Connect to all endpoints sequentially, list each server's tools,
    /// build the routing table.
    ///
    /// Resilience: a server that fails to connect logs a `WARN` and is
    /// skipped — the pool starts with whatever connected. Bails only when
    /// **all** servers failed (no tools available means no agent).
    ///
    /// Errors at the routing-table stage on tool-name collisions across
    /// servers (see [`build_routing_table`]).
    pub async fn connect(endpoints: Vec<(String, String)>) -> anyhow::Result<Self> {
        if endpoints.is_empty() {
            bail!("no MCP endpoints provided");
        }

        let mut connected: Vec<(String, RunningService<RoleClient, ClientInfo>)> = Vec::new();
        for (label, url) in endpoints {
            match open_session(&url).await {
                Ok(svc) => connected.push((label, svc)),
                Err(e) => {
                    tracing::warn!(%label, %url, error = ?e, "MCP server unreachable, skipping");
                }
            }
        }

        if connected.is_empty() {
            bail!("no MCP servers could be reached — agent has no tools to dispatch");
        }

        // tools/list per server. Same skip-WARN posture as `open_session`:
        // a single server that handshook OK but errors on `tools/list` is
        // dropped from the pool rather than crashing the whole agent. Bail
        // only if NO server returned a usable tool-list.
        let mut server_tools: Vec<(String, Vec<Tool>)> = Vec::with_capacity(connected.len());
        let mut keep_indices: Vec<usize> = Vec::with_capacity(connected.len());
        for (i, (label, svc)) in connected.iter().enumerate() {
            match svc.list_tools(Default::default()).await {
                Ok(result) => {
                    server_tools.push((label.clone(), result.tools.clone()));
                    keep_indices.push(i);
                }
                Err(e) => {
                    tracing::warn!(
                        %label,
                        error = ?e,
                        "list_tools failed — dropping server from pool"
                    );
                }
            }
        }
        // Drop the servers that failed list_tools. Walk indices in reverse
        // so each `swap_remove` doesn't disturb earlier ones.
        let dropped_total = connected.len() - keep_indices.len();
        if dropped_total > 0 {
            let keep: std::collections::HashSet<usize> = keep_indices.iter().copied().collect();
            connected = connected
                .into_iter()
                .enumerate()
                .filter(|(i, _)| keep.contains(i))
                .map(|(_, s)| s)
                .collect();
        }
        if connected.is_empty() {
            bail!("all MCP servers failed tools/list — agent has no tools to dispatch");
        }

        let (tool_specs, tool_to_session_idx) = build_routing_table(&server_tools)?;

        tracing::info!(
            servers = connected.len(),
            tools = tool_specs.len(),
            "MCP pool ready",
        );

        Ok(Self {
            sessions: connected,
            tool_to_session_idx,
            tool_specs,
            publisher: None,
        })
    }

    /// Aggregate tool-specs across all connected servers — to be passed to
    /// the orchestrator as the agent's tool surface.
    pub fn tool_specs(&self) -> Vec<ToolSpec> {
        self.tool_specs.clone()
    }

    /// Wire up an AMQP publisher so each `tool_called` event lands on
    /// `ai.events`. Call before the pool is shared via `Arc<AppState>`.
    pub fn attach_publisher(&mut self, publisher: Arc<Publisher>) {
        self.publisher = Some(publisher);
    }

    /// Send DELETE-session to every connected server. Best-effort: a
    /// failed shutdown on one server logs a `WARN` but does not stop us
    /// trying the rest. Consumes self because sessions are unusable
    /// post-shutdown.
    pub async fn shutdown(self) -> anyhow::Result<()> {
        for (label, svc) in self.sessions {
            if let Err(e) = svc.cancel().await {
                tracing::warn!(%label, error = ?e, "MCP session shutdown failed");
            }
        }
        Ok(())
    }
}

/// Defence-in-depth validation of LLM-generated tool args against the
/// cached `input_schema` advertised by the MCP server. Returns Ok if the
/// tool is unknown (caller's existing 'no MCP server provides tool' check
/// will fire), or if validation passes. On failure, bails with a message
/// the orchestrator surfaces back to the LLM.
fn validate_args_against_schema(
    specs: &[ToolSpec],
    name: &str,
    args: &Value,
) -> anyhow::Result<()> {
    let Some(spec) = specs.iter().find(|t| t.name == name) else {
        return Ok(());
    };
    let validator = jsonschema::validator_for(&spec.input_schema)
        .map_err(|e| anyhow::anyhow!("invalid input_schema for tool '{name}': {e}"))?;
    if !validator.is_valid(args) {
        let errs: Vec<String> = validator.iter_errors(args).map(|e| e.to_string()).collect();
        anyhow::bail!(
            "tool args for '{name}' violate input_schema: {}",
            errs.join("; ")
        );
    }
    Ok(())
}

#[async_trait]
impl McpExecutor for McpPool {
    async fn call(&self, name: &str, arguments: Value) -> anyhow::Result<String> {
        validate_args_against_schema(&self.tool_specs, name, &arguments)?;
        let map = arguments
            .as_object()
            .cloned()
            .context("MCP tool arguments must be a JSON object")?;
        let idx = *self
            .tool_to_session_idx
            .get(name)
            .with_context(|| format!("no MCP server provides tool '{name}'"))?;
        let (label, svc) = &self.sessions[idx];
        tracing::debug!(tool = %name, server = %label, "dispatching MCP tool call");

        let started = std::time::Instant::now();
        let outcome = svc
            .call_tool(CallToolRequestParams::new(name.to_string()).with_arguments(map))
            .await;
        let duration_ms = started.elapsed().as_millis() as u64;
        let success = outcome.is_ok();

        if let Some(publisher) = &self.publisher {
            let payload = serde_json::json!({
                "tool": name,
                "server": label,
                "success": success,
                "duration_ms": duration_ms,
            });
            if let Err(e) = publisher.publish_event("tool_called", payload).await {
                tracing::warn!("failed to publish tool_called event: {e:#}");
            }
        }

        let res = outcome
            .with_context(|| format!("MCP tools/call failed for '{name}' on server '{label}'"))?;
        Ok(extract_tool_result_text(&res))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use rmcp::model::Tool;
    use serde_json::json;
    use std::sync::Arc;

    fn tool(name: &str) -> Tool {
        let mut t = Tool::default();
        // `Tool.name` is `Cow<'static, str>`; force owned (`String -> Cow::Owned`)
        // so the borrow on `name` doesn't constrain its lifetime to `'static`.
        t.name = name.to_string().into();
        t.input_schema = Arc::new(serde_json::Map::new());
        t
    }

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
        let mut t = Tool::default();
        t.name = "heartbeat_status".into();
        t.description = Some("Recent heartbeats".into());
        t.input_schema = Arc::new(schema_obj.clone());

        let spec = tool_to_spec(&t);

        assert_eq!(spec.name, "heartbeat_status");
        assert_eq!(spec.description, "Recent heartbeats");
        assert_eq!(spec.input_schema, Value::Object(schema_obj));
    }

    #[test]
    fn tool_to_spec_uses_empty_description_when_none() {
        let mut t = Tool::default();
        t.name = "anon".into();
        t.description = None;
        t.input_schema = Arc::new(serde_json::Map::new());
        let spec = tool_to_spec(&t);
        assert_eq!(spec.description, "");
    }

    #[test]
    fn parse_endpoints_handles_comma_separated_pairs() {
        let v =
            parse_endpoints("crm@http://localhost:7001/mcp,controlroom@http://localhost:7002/mcp");
        assert_eq!(
            v,
            vec![
                ("crm".to_string(), "http://localhost:7001/mcp".to_string()),
                (
                    "controlroom".to_string(),
                    "http://localhost:7002/mcp".to_string(),
                ),
            ],
        );
    }

    #[test]
    fn parse_endpoints_skips_malformed_entries() {
        // Bad entries: no `@`, empty label, empty url. Only well-formed pairs survive.
        let v =
            parse_endpoints("crm@http://x,bad-no-at,@empty-label,empty-url@,controlroom@http://y");
        assert_eq!(
            v,
            vec![
                ("crm".to_string(), "http://x".to_string()),
                ("controlroom".to_string(), "http://y".to_string()),
            ],
        );
    }

    #[test]
    fn parse_endpoints_handles_empty_string() {
        assert!(parse_endpoints("").is_empty());
    }

    #[test]
    fn build_routing_table_assigns_tools_to_sessions() {
        let server_tools = vec![
            (
                "crm".to_string(),
                vec![tool("search_contact"), tool("count_contacts")],
            ),
            (
                "controlroom".to_string(),
                vec![tool("error_analysis"), tool("heartbeat_status")],
            ),
        ];

        let (specs, idx) = build_routing_table(&server_tools).expect("should build");
        assert_eq!(specs.len(), 4, "all tools collected into specs");
        assert_eq!(idx.get("search_contact").copied(), Some(0));
        assert_eq!(idx.get("count_contacts").copied(), Some(0));
        assert_eq!(idx.get("error_analysis").copied(), Some(1));
        assert_eq!(idx.get("heartbeat_status").copied(), Some(1));
    }

    #[test]
    fn validate_args_rejects_oversized_limit() {
        let specs = vec![ToolSpec {
            name: "search_contact".into(),
            description: "Fuzzy contact search".into(),
            input_schema: json!({
                "type": "object",
                "properties": {
                    "query": {"type": "string"},
                    "limit": {"type": "integer", "minimum": 1, "maximum": 100}
                },
                "required": ["query"]
            }),
        }];
        let bad = json!({"query": "x", "limit": 999_999});
        let err = validate_args_against_schema(&specs, "search_contact", &bad)
            .expect_err("limit > maximum must fail");
        let msg = format!("{err:#}");
        assert!(msg.contains("input_schema"), "got: {msg}");
    }

    #[test]
    fn validate_args_accepts_valid_args() {
        let specs = vec![ToolSpec {
            name: "search_contact".into(),
            description: "..".into(),
            input_schema: json!({
                "type": "object",
                "properties": {"query": {"type": "string"}, "limit": {"type": "integer"}},
                "required": ["query"]
            }),
        }];
        validate_args_against_schema(
            &specs,
            "search_contact",
            &json!({"query": "Brend", "limit": 10}),
        )
        .expect("valid args pass");
    }

    #[test]
    fn validate_args_passes_unknown_tool_through() {
        let specs: Vec<ToolSpec> = vec![];
        validate_args_against_schema(&specs, "anything", &json!({}))
            .expect("unknown tool returns Ok — caller's routing-table check handles it");
    }

    #[test]
    fn validate_args_rejects_missing_required_field() {
        let specs = vec![ToolSpec {
            name: "get_contact".into(),
            description: "..".into(),
            input_schema: json!({
                "type": "object",
                "properties": {"contact_id": {"type": "string"}},
                "required": ["contact_id"]
            }),
        }];
        let err = validate_args_against_schema(&specs, "get_contact", &json!({}))
            .expect_err("missing required field must fail");
        let msg = format!("{err:#}");
        assert!(msg.contains("input_schema"), "got: {msg}");
    }

    #[test]
    fn build_routing_table_errors_on_collision() {
        let server_tools = vec![
            ("crm".to_string(), vec![tool("ping")]),
            ("controlroom".to_string(), vec![tool("ping")]),
        ];

        let err =
            build_routing_table(&server_tools).expect_err("should bail on duplicate tool name");
        let msg = format!("{err}");
        assert!(msg.contains("ping"), "error msg should mention tool: {msg}");
        assert!(
            msg.contains("crm"),
            "error msg should mention first owner: {msg}"
        );
        assert!(
            msg.contains("controlroom"),
            "error msg should mention second owner: {msg}",
        );
    }
}
