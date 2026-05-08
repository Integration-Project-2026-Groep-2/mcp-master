//! Agent operating modes.
//!
//! Two modes today:
//! - [`ReadOnlyMode`]: tool-loop dispatches read-only tools only. Write-tool
//!   requests are surfaced to the LLM as a recoverable error instead of being
//!   sent to the MCP-server. Compile-time guarantee: this struct **does not
//!   carry an [`ApprovalFlow`]**, so it cannot construct a write-tool dispatch.
//! - [`ActionableMode`]: tool-loop intercepts write-tools, proposes them to
//!   the approval store, and waits for an explicit `/chat/approve` call before
//!   executing.
//!
//! PR-1 declares the types only — dispatch methods land in PR-3 once the
//! approval primitives (PR-2) are in place. The `_placeholder` field on
//! [`ActionableMode`] is the parking-lot for `Arc<ApprovalFlow>`.

#![allow(dead_code)] // PR-3 will read these; PR-1 only declares the types.

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

#[derive(Clone)]
pub struct ActionableMode {
    /// Reserved for PR-2's `Arc<crate::gateway::approval::ApprovalFlow>`.
    /// Marked private so callers must use [`ActionableMode::new`].
    _placeholder: (),
}

impl ActionableMode {
    pub fn new() -> Self {
        Self { _placeholder: () }
    }
}

impl Default for ActionableMode {
    fn default() -> Self {
        Self::new()
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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn read_only_mode_advertises_no_write_tools() {
        let m = ReadOnlyMode;
        assert_eq!(m.label(), "read-only");
        assert!(!m.allows_write_tools());
    }

    #[test]
    fn actionable_mode_advertises_write_tools() {
        let m = ActionableMode::new();
        assert_eq!(m.label(), "actionable");
        assert!(m.allows_write_tools());
    }

    #[test]
    fn agent_mode_dispatches_to_inner_via_trait() {
        let read: AgentMode = AgentMode::ReadOnly(ReadOnlyMode);
        let act: AgentMode = AgentMode::Actionable(ActionableMode::new());
        assert_eq!(read.label(), "read-only");
        assert_eq!(act.label(), "actionable");
        assert!(!read.allows_write_tools());
        assert!(act.allows_write_tools());
    }
}
