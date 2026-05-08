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
mod tests {
    use super::*;
    use crate::gateway::auth::AuthScope;
    use serde_json::json;

    fn make_flow() -> ApprovalFlow {
        let store = Arc::new(ApprovalStore::new(Duration::from_secs(900)));
        let audit = Arc::new(AuditPublisher::new(None));
        ApprovalFlow::new(store, audit, Duration::from_secs(900))
    }

    fn make_draft(user: &str) -> PendingActionDraft {
        PendingActionDraft {
            correlation_id: "cid".into(),
            user_id: user.into(),
            scope: AuthScope::ReadAndAct,
            tool_name: "create_company".into(),
            tool_args: json!({"name": "Acme"}),
            server_label: "crm".into(),
        }
    }

    #[tokio::test]
    async fn propose_inserts_proposed_action_in_store() {
        let flow = make_flow();
        let id = flow.propose(make_draft("alice")).await;
        let stored = flow.store.get(id).expect("inserted");
        assert_eq!(stored.status, ApprovalStatus::Proposed);
        assert_eq!(stored.user_id, "alice");
    }

    #[tokio::test]
    async fn confirm_happy_path_transitions_to_approved() {
        let flow = make_flow();
        let id = flow.propose(make_draft("alice")).await;
        let updated = flow.confirm(id, "alice").await.expect("confirm ok");
        assert_eq!(updated.status, ApprovalStatus::Approved);
    }

    #[tokio::test]
    async fn double_confirm_returns_already_decided() {
        let flow = make_flow();
        let id = flow.propose(make_draft("alice")).await;
        flow.confirm(id, "alice").await.expect("first confirm ok");
        let err = flow.confirm(id, "alice").await.expect_err("second fails");
        assert!(matches!(err, ApprovalError::AlreadyDecided(_)));
    }

    #[tokio::test]
    async fn confirm_with_wrong_user_returns_wrong_user() {
        let flow = make_flow();
        let id = flow.propose(make_draft("alice")).await;
        let err = flow
            .confirm(id, "mallory")
            .await
            .expect_err("hijack must be rejected");
        assert!(matches!(err, ApprovalError::WrongUser { .. }));
        // Status unchanged in store.
        assert_eq!(flow.store.get(id).unwrap().status, ApprovalStatus::Proposed);
    }

    #[tokio::test]
    async fn confirm_after_expiry_returns_expired() {
        let store = Arc::new(ApprovalStore::new(Duration::from_secs(900)));
        let audit = Arc::new(AuditPublisher::new(None));
        let flow = ApprovalFlow::new(store.clone(), audit, Duration::from_secs(900));
        let id = flow.propose(make_draft("alice")).await;
        // Manually age the action by overwriting expires_at in the store.
        store
            .try_transition(id, |a| {
                // Returning current status keeps this pure-introspection;
                // the side-effect we want is mutating expires_at, but
                // try_transition's pred only sees &PendingAction. Fall back:
                // remove + re-insert with aged timestamp.
                Ok(a.status.clone())
            })
            .expect("noop ok");
        if let Some(mut action) = store.remove(id) {
            action.expires_at = Utc::now() - chrono::Duration::seconds(1);
            store.insert(action);
        }
        let err = flow.confirm(id, "alice").await.expect_err("expired");
        assert!(matches!(err, ApprovalError::Expired(_)));
    }

    #[tokio::test]
    async fn reject_happy_path_transitions_to_rejected() {
        let flow = make_flow();
        let id = flow.propose(make_draft("alice")).await;
        let updated = flow
            .reject(id, "alice", Some("nope".into()))
            .await
            .expect("reject ok");
        assert_eq!(updated.status, ApprovalStatus::Rejected);
    }

    #[tokio::test]
    async fn mark_executed_only_after_approved() {
        let flow = make_flow();
        let id = flow.propose(make_draft("alice")).await;
        // From Proposed: must fail.
        let err = flow
            .mark_executed(id, "ok", 100)
            .await
            .expect_err("not yet approved");
        assert!(matches!(err, ApprovalError::AlreadyDecided(_)));
        flow.confirm(id, "alice").await.expect("confirm");
        let updated = flow
            .mark_executed(id, "1 record affected", 412)
            .await
            .expect("execute ok");
        assert_eq!(updated.status, ApprovalStatus::Executed);
        let stored = flow.store.get(id).unwrap();
        assert_eq!(stored.executed_result.as_deref(), Some("1 record affected"),);
        assert_eq!(stored.executed_duration_ms, Some(412));
    }

    #[tokio::test]
    async fn concurrent_confirm_exactly_one_succeeds() {
        let flow = Arc::new(make_flow());
        let id = flow.propose(make_draft("alice")).await;

        let mut handles = Vec::new();
        for _ in 0..5 {
            let f = flow.clone();
            handles.push(tokio::spawn(async move { f.confirm(id, "alice").await }));
        }
        let results: Vec<_> = futures_util::future::join_all(handles).await;
        let oks = results
            .iter()
            .filter(|r| r.as_ref().unwrap().is_ok())
            .count();
        let errs = results
            .iter()
            .filter(|r| r.as_ref().unwrap().is_err())
            .count();
        assert_eq!(oks, 1, "exactly one confirm wins the race");
        assert_eq!(errs, 4, "the other four see AlreadyDecided");
    }
}
