//! Approval state-machine: orchestrates `ApprovalStore` + `AuditPublisher`.
//!
//! The state-machine has four happy transitions:
//! ```text
//!     [propose]  → Proposed
//!     [confirm]  Proposed → Approved
//!     [reject]   Proposed → Rejected
//!     [execute]  Approved → Executed
//! ```
//! Plus implicit `Proposed → Expired` driven by `state::run_cleanup_task`.
//!
//! All four methods are atomic via `ApprovalStore::try_transition`. The
//! predicate runs inside DashMap's per-entry lock so a double-`confirm`
//! race resolves with exactly one Ok and one `AlreadyDecided`.
//!
//! `confirm` and `reject` enforce that the caller's `user_id` matches the
//! original proposer (defence against action-id hijack across users).

#![allow(dead_code)] // Wired into orchestrator (PR-3) + http_api (PR-4).

use std::sync::Arc;
use std::time::Duration;

use chrono::Utc;
use uuid::Uuid;

use crate::gateway::approval::state::ApprovalStore;
use crate::gateway::approval::types::{
    ApprovalError, ApprovalStatus, PendingAction, PendingActionDraft,
};
use crate::gateway::audit::AuditPublisher;

pub struct ApprovalFlow {
    store: Arc<ApprovalStore>,
    audit: Arc<AuditPublisher>,
    ttl: Duration,
}

impl ApprovalFlow {
    pub fn new(store: Arc<ApprovalStore>, audit: Arc<AuditPublisher>, ttl: Duration) -> Self {
        Self { store, audit, ttl }
    }

    pub async fn propose(&self, draft: PendingActionDraft) -> Uuid {
        let action_id = Uuid::new_v4();
        let proposed_at = Utc::now();
        let expires_at = proposed_at
            + chrono::Duration::from_std(self.ttl).expect("approval TTL fits in chrono::Duration");
        let action = PendingAction {
            action_id,
            correlation_id: draft.correlation_id,
            user_id: draft.user_id,
            scope: draft.scope,
            tool_name: draft.tool_name,
            tool_args: draft.tool_args,
            server_label: draft.server_label,
            proposed_at,
            expires_at,
            status: ApprovalStatus::Proposed,
            executed_result: None,
            executed_duration_ms: None,
        };
        self.store.insert(action.clone());
        self.audit.proposed(&action).await;
        tracing::info!(
            action_id = %action_id,
            correlation_id = %action.correlation_id,
            tool = %action.tool_name,
            "approval proposed",
        );
        action_id
    }

    pub async fn confirm(&self, id: Uuid, user_id: &str) -> Result<PendingAction, ApprovalError> {
        let updated = self.store.try_transition(id, |action| {
            if action.status != ApprovalStatus::Proposed {
                return Err(ApprovalError::AlreadyDecided(action.status.clone()));
            }
            if action.user_id != user_id {
                return Err(ApprovalError::WrongUser {
                    proposer: action.user_id.clone(),
                    caller: user_id.to_string(),
                });
            }
            if Utc::now() >= action.expires_at {
                return Err(ApprovalError::Expired(action.expires_at));
            }
            Ok(ApprovalStatus::Approved)
        })?;
        self.audit.approved(&updated).await;
        tracing::info!(
            action_id = %id,
            correlation_id = %updated.correlation_id,
            "approval confirmed",
        );
        Ok(updated)
    }

    pub async fn reject(
        &self,
        id: Uuid,
        user_id: &str,
        reason: Option<String>,
    ) -> Result<PendingAction, ApprovalError> {
        let updated = self.store.try_transition(id, |action| {
            if action.status != ApprovalStatus::Proposed {
                return Err(ApprovalError::AlreadyDecided(action.status.clone()));
            }
            if action.user_id != user_id {
                return Err(ApprovalError::WrongUser {
                    proposer: action.user_id.clone(),
                    caller: user_id.to_string(),
                });
            }
            Ok(ApprovalStatus::Rejected)
        })?;
        self.audit.rejected(&updated, reason.as_deref()).await;
        tracing::info!(
            action_id = %id,
            correlation_id = %updated.correlation_id,
            "approval rejected",
        );
        Ok(updated)
    }

    pub async fn mark_executed(
        &self,
        id: Uuid,
        result: &str,
        duration_ms: u64,
    ) -> Result<PendingAction, ApprovalError> {
        let updated = self.store.try_transition(id, |action| {
            if action.status != ApprovalStatus::Approved {
                return Err(ApprovalError::AlreadyDecided(action.status.clone()));
            }
            Ok(ApprovalStatus::Executed)
        })?;
        // Persist execution metadata on the canonical store entry —
        // try_transition cloned for the return, so the original needs the
        // result + duration written back here.
        self.store
            .set_execution_metadata(id, result.to_string(), duration_ms);
        self.audit.executed(&updated, result, duration_ms).await;
        tracing::info!(
            action_id = %id,
            correlation_id = %updated.correlation_id,
            duration_ms,
            "approval executed",
        );
        Ok(updated)
    }
}

#[cfg(test)]
mod tests;
