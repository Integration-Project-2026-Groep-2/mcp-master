//! Agent operating modes.
//!
//! - [`ReadOnlyMode`]: dispatches read-tools only. Write-tool requests are
//!   surfaced to the LLM as a recoverable `is_error=true` result via
//!   [`build_blocked_read_only_result`]. Compile-time guarantee:
//!   `ReadOnlyMode` does **not** carry an [`ApprovalFlow`] and has **no**
//!   `dispatch_write_tool` method, so a write-dispatch site can only compile
//!   against [`ActionableMode`].
//! - [`ActionableMode`]: read-tools pass through, write-tools route through
//!   `flow.propose` and the LLM sees an `ACTION_PROPOSED:` marker. Actual
//!   execution waits for `flow.confirm` (PR-4 wires `/chat/approve`).
//!
//! [`DispatchContext`] threads per-request identity (correlation_id,
//! user_id, scope) into the proposal so audit envelopes don't need to
//! reach back into the HTTP layer.

// Module-level allow because PR-4 is the first construction site for
// `AgentMode::Actionable` + `ActionableMode::new` + the `Mode` trait
// (the orchestrator only pattern-matches against them in PR-3).
#![allow(dead_code)]

use std::sync::Arc;

use serde_json::Value;

use crate::agent::orchestrator::{McpExecutor, ToolCallTrace};
use crate::gateway::approval::flow::ApprovalFlow;
use crate::gateway::approval::types::PendingActionDraft;
use crate::gateway::auth::AuthScope;

#[derive(Clone, Debug)]
pub struct DispatchContext {
    pub correlation_id: String,
    pub user_id: String,
    pub scope: AuthScope,
}

impl Default for DispatchContext {
    /// Empty correlation/user; scope=Read. Used by the legacy `run` /
    /// `run_with_messages` shims that have no JWT context.
    fn default() -> Self {
        Self {
            correlation_id: String::new(),
            user_id: String::new(),
            scope: AuthScope::Read,
        }
    }
}

#[derive(Clone)]
pub enum AgentMode {
    ReadOnly(ReadOnlyMode),
    Actionable(ActionableMode),
}

pub trait Mode: Send + Sync {
    fn label(&self) -> &'static str;
    fn allows_write_tools(&self) -> bool;
}

impl Mode for AgentMode {
    fn label(&self) -> &'static str {
        match self {
            Self::ReadOnly(m) => m.label(),
            Self::Actionable(m) => m.label(),
        }
    }
    fn allows_write_tools(&self) -> bool {
        match self {
            Self::ReadOnly(m) => m.allows_write_tools(),
            Self::Actionable(m) => m.allows_write_tools(),
        }
    }
}

#[derive(Clone, Default)]
pub struct ReadOnlyMode;

impl Mode for ReadOnlyMode {
    fn label(&self) -> &'static str {
        "read-only"
    }
    fn allows_write_tools(&self) -> bool {
        false
    }
}

impl ReadOnlyMode {
    /// Dispatch a read-tool; pure delegation to the executor.
    pub async fn dispatch_read_tool(
        &self,
        executor: &dyn McpExecutor,
        name: &str,
        args: Value,
    ) -> anyhow::Result<(String, ToolCallTrace)> {
        executor.call(name, args).await
    }
}

#[derive(Clone)]
pub struct ActionableMode {
    flow: Arc<ApprovalFlow>,
}

impl ActionableMode {
    pub fn new(flow: Arc<ApprovalFlow>) -> Self {
        Self { flow }
    }

    /// Dispatch a read-tool; pure delegation, mirrors `ReadOnlyMode` so the
    /// orchestrator's match-arms unify on (`ReadOnly`, false) +
    /// (`Actionable`, false).
    pub async fn dispatch_read_tool(
        &self,
        executor: &dyn McpExecutor,
        name: &str,
        args: Value,
    ) -> anyhow::Result<(String, ToolCallTrace)> {
        executor.call(name, args).await
    }

    /// Intercept a write-tool: propose to the approval flow, return an
    /// `ACTION_PROPOSED:` marker for the LLM to render to the user. The
    /// actual MCP-server dispatch happens in PR-4's `/chat/approve` handler
    /// after `flow.confirm` succeeds.
    pub async fn dispatch_write_tool(
        &self,
        executor: &dyn McpExecutor,
        ctx: &DispatchContext,
        name: &str,
        args: Value,
    ) -> anyhow::Result<(String, ToolCallTrace)> {
        let server_label = executor
            .server_label_for(name)
            .unwrap_or_else(|| "<unknown>".into());
        let draft = PendingActionDraft {
            correlation_id: ctx.correlation_id.clone(),
            user_id: ctx.user_id.clone(),
            scope: ctx.scope,
            tool_name: name.to_string(),
            tool_args: args,
            server_label: server_label.clone(),
        };
        let action_id = self.flow.propose(draft).await;
        let result = format!(
            "ACTION_PROPOSED: action_id={action_id}; tool={name} requires user approval. \
             Tell the user a preview of the action they need to approve via /chat/approve."
        );
        let trace = ToolCallTrace {
            tool: name.to_string(),
            server: server_label,
            ms: 0,
            ok: true,
            error: None,
            args: None,
            status: Some("pending".into()),
        };
        Ok((result, trace))
    }
}

impl Mode for ActionableMode {
    fn label(&self) -> &'static str {
        "actionable"
    }
    fn allows_write_tools(&self) -> bool {
        true
    }
}

/// Synthesize the LLM-visible result for a write-tool denied by `ReadOnlyMode`.
///
/// Returns an `is_error=true` `ToolCallTrace` plus a recoverable text the
/// LLM should rephrase as a polite "you need higher privileges" message.
/// `status="blocked_read_only"` discriminates it from regular tool failures
/// in the v1.4 audit feed.
pub fn build_blocked_read_only_result(
    tool_name: &str,
    server_label: &str,
) -> (String, ToolCallTrace) {
    let result = format!(
        "TOOL_BLOCKED_READ_ONLY: tool '{tool_name}' requires scope read+act; current scope is read. \
         Politely tell the user this needs higher privileges; do not retry the same tool."
    );
    let trace = ToolCallTrace {
        tool: tool_name.to_string(),
        server: server_label.to_string(),
        ms: 0,
        ok: false,
        error: Some("blocked_read_only".into()),
        args: None,
        status: Some("blocked_read_only".into()),
    };
    (result, trace)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::gateway::approval::state::ApprovalStore;
    use crate::gateway::approval::types::ApprovalStatus;
    use crate::gateway::audit::AuditPublisher;
    use async_trait::async_trait;
    use serde_json::json;
    use std::time::Duration;
    use tokio::sync::Mutex;

    fn make_flow() -> Arc<ApprovalFlow> {
        let store = Arc::new(ApprovalStore::new(Duration::from_secs(900)));
        let audit = Arc::new(AuditPublisher::new(None));
        Arc::new(ApprovalFlow::new(store, audit, Duration::from_secs(900)))
    }

    fn make_ctx() -> DispatchContext {
        DispatchContext {
            correlation_id: "cid-test".into(),
            user_id: "alice".into(),
            scope: AuthScope::ReadAndAct,
        }
    }

    /// Test executor that records calls and answers from a queue.
    struct StubExecutor {
        canned_response: Mutex<Option<(String, ToolCallTrace)>>,
        server_label: Option<String>,
    }

    #[async_trait]
    impl McpExecutor for StubExecutor {
        async fn call(&self, name: &str, _args: Value) -> anyhow::Result<(String, ToolCallTrace)> {
            self.canned_response
                .lock()
                .await
                .clone()
                .ok_or_else(|| anyhow::anyhow!("StubExecutor: no canned response for {name}"))
        }
        fn server_label_for(&self, _name: &str) -> Option<String> {
            self.server_label.clone()
        }
    }

    fn stub_with_response(text: &str, label: Option<&str>) -> StubExecutor {
        let trace = ToolCallTrace {
            tool: "search_contact".into(),
            server: label.unwrap_or("test").into(),
            ms: 1,
            ok: true,
            error: None,
            args: None,
            status: None,
        };
        StubExecutor {
            canned_response: Mutex::new(Some((text.into(), trace))),
            server_label: label.map(str::to_string),
        }
    }

    #[test]
    fn read_only_mode_advertises_no_write_tools() {
        let m = ReadOnlyMode;
        assert_eq!(m.label(), "read-only");
        assert!(!m.allows_write_tools());
    }

    #[test]
    fn actionable_mode_advertises_write_tools() {
        let m = ActionableMode::new(make_flow());
        assert_eq!(m.label(), "actionable");
        assert!(m.allows_write_tools());
    }

    #[test]
    fn agent_mode_dispatches_to_inner_via_trait() {
        let read = AgentMode::ReadOnly(ReadOnlyMode);
        let act = AgentMode::Actionable(ActionableMode::new(make_flow()));
        assert_eq!(read.label(), "read-only");
        assert_eq!(act.label(), "actionable");
        assert!(!read.allows_write_tools());
        assert!(act.allows_write_tools());
    }

    #[tokio::test]
    async fn read_only_dispatch_read_tool_delegates_to_executor() {
        let mode = ReadOnlyMode;
        let exec = stub_with_response("44 active contacts", None);
        let (result, trace) = mode
            .dispatch_read_tool(&exec, "count_contacts", json!({}))
            .await
            .unwrap();
        assert_eq!(result, "44 active contacts");
        assert!(trace.ok);
        assert!(trace.status.is_none());
    }

    #[tokio::test]
    async fn actionable_dispatch_read_tool_delegates_to_executor() {
        let mode = ActionableMode::new(make_flow());
        let exec = stub_with_response("44 active contacts", None);
        let (result, trace) = mode
            .dispatch_read_tool(&exec, "count_contacts", json!({}))
            .await
            .unwrap();
        assert_eq!(result, "44 active contacts");
        assert!(trace.ok);
    }

    #[tokio::test]
    async fn actionable_dispatch_write_tool_proposes_via_flow() {
        // Construct store+flow side-by-side so the test can inspect the
        // store after the dispatch — ApprovalFlow's own store is private.
        let store = Arc::new(ApprovalStore::new(Duration::from_secs(900)));
        let audit = Arc::new(AuditPublisher::new(None));
        let flow = Arc::new(ApprovalFlow::new(
            store.clone(),
            audit,
            Duration::from_secs(900),
        ));
        let mode = ActionableMode::new(flow);
        let exec = stub_with_response("unused", Some("crm"));
        let (text, trace) = mode
            .dispatch_write_tool(
                &exec,
                &make_ctx(),
                "create_company",
                json!({"name": "Acme"}),
            )
            .await
            .unwrap();

        // Pull the action_id out of the marker so we can fetch from the store.
        let action_id_str = text
            .split("action_id=")
            .nth(1)
            .and_then(|s| s.split(';').next())
            .expect("marker contains action_id=<uuid>");
        let action_id: uuid::Uuid = action_id_str.parse().expect("uuid parse");

        let stored = store.get(action_id).expect("flow inserted action");
        assert_eq!(stored.tool_name, "create_company");
        assert_eq!(stored.status, ApprovalStatus::Proposed);
        assert_eq!(stored.user_id, "alice");
        assert_eq!(trace.status.as_deref(), Some("pending"));
        assert_eq!(trace.server, "crm");
    }

    #[tokio::test]
    async fn actionable_dispatch_write_tool_returns_action_proposed_marker() {
        let mode = ActionableMode::new(make_flow());
        let exec = stub_with_response("unused", Some("crm"));
        let (text, _trace) = mode
            .dispatch_write_tool(&exec, &make_ctx(), "create_company", json!({}))
            .await
            .unwrap();
        assert!(
            text.starts_with("ACTION_PROPOSED:"),
            "result must start with the marker, got: {text}",
        );
        assert!(text.contains("action_id="));
        assert!(text.contains("create_company"));
    }

    #[tokio::test]
    async fn actionable_dispatch_write_tool_falls_back_when_executor_has_no_label() {
        let mode = ActionableMode::new(make_flow());
        let exec = stub_with_response("unused", None);
        let (_text, trace) = mode
            .dispatch_write_tool(&exec, &make_ctx(), "create_company", json!({}))
            .await
            .unwrap();
        assert_eq!(trace.server, "<unknown>");
    }

    #[test]
    fn build_blocked_read_only_marks_is_error_and_status() {
        let (text, trace) = build_blocked_read_only_result("delete_company", "crm");
        assert!(text.contains("TOOL_BLOCKED_READ_ONLY"));
        assert!(text.contains("delete_company"));
        assert!(!trace.ok);
        assert_eq!(trace.error.as_deref(), Some("blocked_read_only"));
        assert_eq!(trace.status.as_deref(), Some("blocked_read_only"));
        assert_eq!(trace.server, "crm");
    }
}
