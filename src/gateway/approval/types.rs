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
mod tests;
