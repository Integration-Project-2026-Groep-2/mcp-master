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
use tokio::sync::Mutex;

use crate::agent::llm::ToolSpec;
use crate::agent::orchestrator::{McpExecutor, ToolCallTrace};
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
///
/// `requires_approval` is derived from `!annotations.read_only_hint`. rmcp 1.6's
/// `ToolAnnotations` is closed (no `#[serde(flatten)]` / catch-all), so a
/// FastMCP-side `annotations.requires_approval=true` field is silently dropped
/// during deserialization. Inverting `read_only_hint` is reliable because all
/// 6 CRM-MCP write-tools publish `readOnlyHint=false` and all 7 read-tools
/// publish `readOnlyHint=true`. Tools without annotations default to read-only
/// (safer).
//
// TODO(R3): switch to `tool.meta.get("requires_approval")` once CRM-MCP
// publishes via the spec-aligned `_meta` channel (rmcp preserves `Tool.meta`).
pub fn tool_to_spec(tool: &Tool) -> ToolSpec {
    let requires_approval = tool
        .annotations
        .as_ref()
        .and_then(|a| a.read_only_hint)
        .map(|read_only| !read_only)
        .unwrap_or(false);
    ToolSpec {
        name: tool.name.to_string(),
        description: tool.description.as_deref().unwrap_or("").to_string(),
        input_schema: Value::Object((*tool.input_schema).clone()),
        requires_approval,
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

/// One MCP-server session with the URL kept around so we can reopen the
/// transport after a transient failure (SSE break, idle timeout, broker
/// restart, etc.). The inner `RunningService` is locked behind an async
/// `Mutex` because reconnect requires `&mut` to swap the service.
struct ManagedSession {
    label: String,
    url: String,
    inner: Mutex<RunningService<RoleClient, ClientInfo>>,
}

/// Pool of MCP-server sessions with tool-name routing.
///
/// Construct via [`McpPool::connect`]. Sessions are owned by the pool
/// (not `Arc`-shared) so [`McpPool::shutdown`] can `cancel()` each cleanly
/// without needing extraction from `Arc`.
///
/// Resilience: each `tools/call` does a one-shot retry-with-reconnect on
/// transport-shaped errors (heuristic — see [`is_transport_error`]).
/// Per-session `Mutex` serialises concurrent calls to the same server;
/// calls to different servers remain parallel. Acceptable trade-off until
/// rmcp surfaces structured transport errors we can detect without lock.
pub struct McpPool {
    /// Sessions in connection-order. Index into this `Vec` is the stable
    /// identity used by `tool_to_session_idx`.
    sessions: Vec<ManagedSession>,

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

        // Triple of (label, url, RunningService) so we keep the URL through
        // tools/list (needed for ManagedSession's reconnect path).
        let mut connected: Vec<(String, String, RunningService<RoleClient, ClientInfo>)> =
            Vec::new();
        for (label, url) in endpoints {
            match open_session(&url).await {
                Ok(svc) => connected.push((label, url, svc)),
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
        for (i, (label, _url, svc)) in connected.iter().enumerate() {
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

        let sessions: Vec<ManagedSession> = connected
            .into_iter()
            .map(|(label, url, svc)| ManagedSession {
                label,
                url,
                inner: Mutex::new(svc),
            })
            .collect();

        tracing::info!(
            servers = sessions.len(),
            tools = tool_specs.len(),
            "MCP pool ready",
        );

        Ok(Self {
            sessions,
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
        for ms in self.sessions {
            // Mutex::into_inner consumes the lock; no contention possible
            // since shutdown ran after the executor stopped.
            let svc = ms.inner.into_inner();
            if let Err(e) = svc.cancel().await {
                tracing::warn!(label = %ms.label, error = ?e, "MCP session shutdown failed");
            }
        }
        Ok(())
    }
}

/// Heuristic for "this error indicates the underlying transport is dead".
/// rmcp's error types are `#[non_exhaustive]` enums whose payload mostly
/// formats as text — string-matching is the pragmatic detection until they
/// expose structured discriminants. False-positive: one wasted reconnect
/// attempt; false-negative: one user-visible failure that the next call
/// will recover from naturally.
fn is_transport_error<E: std::fmt::Display>(err: &E) -> bool {
    let msg = format!("{err}").to_ascii_lowercase();
    [
        "transport",
        "connection",
        "broken pipe",
        "reset",
        "stream",
        "closed",
        "decode",       // rmcp SSE: "error decoding response body"
        "send message", // rmcp transport worker: "Send message error Transport ..."
        "channel closed",
        "io error",
    ]
    .iter()
    .any(|needle| msg.contains(needle))
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
    fn server_label_for(&self, tool_name: &str) -> Option<String> {
        self.tool_to_session_idx
            .get(tool_name)
            .map(|&idx| self.sessions[idx].label.clone())
    }

    async fn call(&self, name: &str, arguments: Value) -> anyhow::Result<(String, ToolCallTrace)> {
        validate_args_against_schema(&self.tool_specs, name, &arguments)?;
        let map = arguments
            .as_object()
            .cloned()
            .context("MCP tool arguments must be a JSON object")?;
        let idx = *self
            .tool_to_session_idx
            .get(name)
            .with_context(|| format!("no MCP server provides tool '{name}'"))?;
        let session = &self.sessions[idx];
        let server_label = session.label.clone();
        tracing::debug!(tool = %name, server = %server_label, "dispatching MCP tool call");

        let started = std::time::Instant::now();
        let outcome = call_with_reconnect(session, name, &map).await;
        let duration_ms = started.elapsed().as_millis() as u64;
        let success = outcome.is_ok();

        if let Some(publisher) = &self.publisher {
            let payload = serde_json::json!({
                "tool": name,
                "server": server_label,
                "success": success,
                "duration_ms": duration_ms,
            });
            if let Err(e) = publisher.publish_event("tool_called", payload).await {
                tracing::warn!("failed to publish tool_called event: {e:#}");
            }
        }

        // Args in trace are gated by env-flag — default off so BTW/email-shaped
        // values stay out of response bodies and AMQP audit events.
        let args_for_trace = if trace_args_enabled() {
            Some(arguments.clone())
        } else {
            None
        };

        match outcome {
            Ok(res) => {
                let text = extract_tool_result_text(&res);
                let trace = ToolCallTrace {
                    tool: name.to_string(),
                    server: server_label,
                    ms: duration_ms,
                    ok: true,
                    error: None,
                    args: args_for_trace,
                    status: None,
                };
                Ok((text, trace))
            }
            Err(e) => {
                // First line only — no anyhow chain leak (would surface
                // RABBITMQ_URL credentials, file paths, env-var names).
                let short = format!("{e}").lines().next().unwrap_or("").to_string();
                let trace = ToolCallTrace {
                    tool: name.to_string(),
                    server: server_label,
                    ms: duration_ms,
                    ok: false,
                    error: Some(short.clone()),
                    args: args_for_trace,
                    status: None,
                };
                // Surface to LLM via is_error=true ToolResult (built by
                // orchestrator from trace.ok). Conversation continues so
                // Anthropic can plan recovery instead of bailing the run.
                Ok((short, trace))
            }
        }
    }
}

fn trace_args_enabled() -> bool {
    std::env::var("CHAT_TRACE_INCLUDE_ARGS")
        .ok()
        .map(|s| s.trim().eq_ignore_ascii_case("true"))
        .unwrap_or(false)
}

/// Try `tools/call`; on transport-shaped failure, reopen the session
/// (single-flight, holding the per-session Mutex throughout) and retry once.
/// All other errors propagate directly — they're either deterministic
/// (validation, no-such-tool, server-side bug) or already wrapped with
/// enough context to debug.
async fn call_with_reconnect(
    session: &ManagedSession,
    name: &str,
    args: &serde_json::Map<String, Value>,
) -> anyhow::Result<CallToolResult> {
    let mut guard = session.inner.lock().await;

    let req = || CallToolRequestParams::new(name.to_string()).with_arguments(args.clone());

    match guard.call_tool(req()).await {
        Ok(r) => return Ok(r),
        Err(e) if is_transport_error(&e) => {
            tracing::warn!(
                label = %session.label,
                "MCP transport error: {e:#} — reopening session"
            );
        }
        Err(e) => return Err(anyhow::Error::from(e)),
    }

    // Reopen + retry once. If reopen itself fails, propagate — there's no
    // point in further retry inside one user request; the next call will
    // try again with a fresh attempt counter.
    let new_svc = open_session(&session.url)
        .await
        .with_context(|| format!("reopening MCP session for '{}'", session.label))?;
    *guard = new_svc;

    guard.call_tool(req()).await.map_err(|e| {
        anyhow::Error::from(e).context(format!(
            "MCP tools/call retry after reconnect failed for '{name}' on '{}'",
            session.label
        ))
    })
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
    fn tool_to_spec_extracts_requires_approval_from_read_only_hint_false() {
        // Write-tool: readOnlyHint=false → requires_approval=true.
        let mut t = Tool::default();
        t.name = "create_company".into();
        t.input_schema = Arc::new(serde_json::Map::new());
        let mut ann = rmcp::model::ToolAnnotations::default();
        ann.read_only_hint = Some(false);
        t.annotations = Some(ann);
        let spec = tool_to_spec(&t);
        assert!(spec.requires_approval);
    }

    #[test]
    fn tool_to_spec_extracts_requires_approval_from_read_only_hint_true() {
        // Read-tool: readOnlyHint=true → requires_approval=false.
        let mut t = Tool::default();
        t.name = "search_contact".into();
        t.input_schema = Arc::new(serde_json::Map::new());
        let mut ann = rmcp::model::ToolAnnotations::default();
        ann.read_only_hint = Some(true);
        t.annotations = Some(ann);
        let spec = tool_to_spec(&t);
        assert!(!spec.requires_approval);
    }

    #[test]
    fn tool_to_spec_defaults_requires_approval_false_when_no_annotations() {
        // Tools without annotations (legacy MCP servers) treated as read-only:
        // safer default that won't accidentally require approval for read-tools.
        let mut t = Tool::default();
        t.name = "legacy_tool".into();
        t.input_schema = Arc::new(serde_json::Map::new());
        t.annotations = None;
        let spec = tool_to_spec(&t);
        assert!(!spec.requires_approval);
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
            requires_approval: false,
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
            requires_approval: false,
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
            requires_approval: false,
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

    #[test]
    fn is_transport_error_classifies_known_strings() {
        // Real-world rmcp / lapin error messages observed in production logs.
        let positives = [
            "sse stream error: body error: error decoding response body",
            "Send message error Transport [...] Client error",
            "Connection reset by peer",
            "Channel closed",
            "transport channel closed",
            "broken pipe",
            "io error: connection reset",
        ];
        for msg in positives {
            let err = anyhow::anyhow!("{msg}");
            assert!(
                is_transport_error(&err),
                "should classify as transport-error: {msg}"
            );
        }
    }

    #[test]
    fn is_transport_error_rejects_deterministic_failures() {
        // These are NOT transport errors — they're bugs in tool args or
        // server logic. Reconnect won't fix them and would just delay
        // surfacing the real problem to the LLM.
        let negatives = [
            "tool 'unknown_tool' not found",
            "input_schema validation failed: missing required field 'query'",
            "ACCESS_REFUSED login",
            "tool execution failed: invalid Salesforce ID",
        ];
        for msg in negatives {
            let err = anyhow::anyhow!("{msg}");
            assert!(
                !is_transport_error(&err),
                "should NOT classify as transport-error: {msg}"
            );
        }
    }

    #[test]
    fn is_transport_error_is_case_insensitive() {
        let err = anyhow::anyhow!("CONNECTION RESET BY PEER");
        assert!(is_transport_error(&err));
    }

    #[test]
    #[serial_test::serial]
    fn trace_args_enabled_returns_false_when_unset() {
        // SAFETY: env mutation is gated by #[serial_test::serial] across the
        // crate; no other test runs concurrently in this thread group.
        unsafe {
            std::env::remove_var("CHAT_TRACE_INCLUDE_ARGS");
        }
        assert!(!trace_args_enabled());
    }

    #[test]
    #[serial_test::serial]
    fn trace_args_enabled_returns_true_when_set_to_true() {
        unsafe {
            std::env::set_var("CHAT_TRACE_INCLUDE_ARGS", "true");
        }
        assert!(trace_args_enabled());
        unsafe {
            std::env::remove_var("CHAT_TRACE_INCLUDE_ARGS");
        }
    }

    #[test]
    #[serial_test::serial]
    fn trace_args_enabled_is_case_insensitive() {
        unsafe {
            std::env::set_var("CHAT_TRACE_INCLUDE_ARGS", "TRUE");
        }
        assert!(trace_args_enabled());
        unsafe {
            std::env::set_var("CHAT_TRACE_INCLUDE_ARGS", "False");
        }
        assert!(!trace_args_enabled());
        unsafe {
            std::env::remove_var("CHAT_TRACE_INCLUDE_ARGS");
        }
    }
}
