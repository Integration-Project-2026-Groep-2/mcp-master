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
            action_id: Some(action_id.to_string()),
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
        action_id: None,
    };
    (result, trace)
}

#[cfg(test)]
mod tests;
