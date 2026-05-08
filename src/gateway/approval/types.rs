//! Data types for the approval state-machine.
//!
//! PR-2 declares only data + errors; the store and flow logic build on top.
//! `PendingAction` is `Serialize` so the audit publisher can ship it on
//! `ai.events` without manual envelope-construction.

#![allow(dead_code)] // Wired into store/flow/audit in subsequent commits.

use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};
use thiserror::Error;
use uuid::Uuid;

use crate::gateway::auth::AuthScope;

#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum ApprovalStatus {
    Proposed,
    Approved,
    Rejected,
    Expired,
    Executed,
}

#[derive(Clone, Debug)]
pub struct PendingActionDraft {
    pub correlation_id: String,
    pub user_id: String,
    pub scope: AuthScope,
    pub tool_name: String,
    pub tool_args: serde_json::Value,
    pub server_label: String,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct PendingAction {
    pub action_id: Uuid,
    pub correlation_id: String,
    pub user_id: String,
    pub scope: AuthScope,
    pub tool_name: String,
    pub tool_args: serde_json::Value,
    pub server_label: String,
    pub proposed_at: DateTime<Utc>,
    pub expires_at: DateTime<Utc>,
    pub status: ApprovalStatus,
    /// Set by `flow.mark_executed` once the dispatched tool returns.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub executed_result: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub executed_duration_ms: Option<u64>,
}

#[derive(Debug, Error)]
pub enum ApprovalError {
    #[error("no pending action with id {0}")]
    NotFound(Uuid),
    #[error("action already decided (status={0:?})")]
    AlreadyDecided(ApprovalStatus),
    #[error("user mismatch: action proposed by '{proposer}', confirm by '{caller}'")]
    WrongUser { proposer: String, caller: String },
    #[error("action expired at {0}")]
    Expired(DateTime<Utc>),
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    fn make_action() -> PendingAction {
        let now = Utc::now();
        PendingAction {
            action_id: Uuid::new_v4(),
            correlation_id: "cid-abc".to_string(),
            user_id: "drupal-uid-42".to_string(),
            scope: AuthScope::ReadAndAct,
            tool_name: "create_company".to_string(),
            tool_args: json!({"name": "Acme NV"}),
            server_label: "crm".to_string(),
            proposed_at: now,
            expires_at: now + chrono::Duration::minutes(15),
            status: ApprovalStatus::Proposed,
            executed_result: None,
            executed_duration_ms: None,
        }
    }

    #[test]
    fn approval_status_serializes_to_lowercase() {
        let json = serde_json::to_string(&ApprovalStatus::Proposed).unwrap();
        assert_eq!(json, "\"proposed\"");
        let json = serde_json::to_string(&ApprovalStatus::Executed).unwrap();
        assert_eq!(json, "\"executed\"");
    }

    #[test]
    fn pending_action_round_trips_via_serde() {
        let action = make_action();
        let json = serde_json::to_string(&action).unwrap();
        let back: PendingAction = serde_json::from_str(&json).unwrap();
        assert_eq!(back.action_id, action.action_id);
        assert_eq!(back.tool_name, action.tool_name);
        assert_eq!(back.scope, action.scope);
        assert_eq!(back.status, action.status);
    }

    #[test]
    fn approval_error_display_includes_context() {
        let err = ApprovalError::AlreadyDecided(ApprovalStatus::Approved);
        assert!(format!("{err}").contains("Approved"));
        let err = ApprovalError::WrongUser {
            proposer: "alice".to_string(),
            caller: "mallory".to_string(),
        };
        let s = format!("{err}");
        assert!(s.contains("alice"));
        assert!(s.contains("mallory"));
    }
}
