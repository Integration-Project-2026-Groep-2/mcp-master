//! Pending-action store with TTL.
//!
//! `ApprovalStore` wraps a `DashMap<Uuid, PendingAction>` keyed by action-id.
//! State transitions go through [`ApprovalStore::try_transition`], which
//! takes a closure that runs *inside* DashMap's per-entry lock — that's the
//! atomic compare-and-swap that prevents two concurrent `confirm` calls
//! from both succeeding.
//!
//! `cleanup_expired` removes only `Proposed` actions whose `expires_at` is
//! in the past. `Approved` actions are awaiting dispatch and must NOT be
//! swept; `Executed` actions are already published as `action_executed`
//! events and remain in the store as a short-lived audit trail (next
//! cleanup pass picks them up after TTL — acceptable for in-memory R2).
//!
//! Wired into the runtime in PR-4 via [`run_cleanup_task`] spawned from
//! `http_api::serve()`.

#![allow(dead_code)] // Wired into AppState in PR-4.

use std::sync::Arc;
use std::time::Duration;

use chrono::{DateTime, Utc};
use dashmap::DashMap;
use tokio::sync::watch;
use uuid::Uuid;

use crate::gateway::approval::types::{ApprovalError, ApprovalStatus, PendingAction};
use crate::gateway::audit::AuditPublisher;

const CLEANUP_INTERVAL_SECONDS: u64 = 60;

pub struct ApprovalStore {
    entries: DashMap<Uuid, PendingAction>,
    ttl: Duration,
}

impl ApprovalStore {
    pub fn new(ttl: Duration) -> Self {
        Self {
            entries: DashMap::new(),
            ttl,
        }
    }

    pub fn ttl(&self) -> Duration {
        self.ttl
    }

    pub fn insert(&self, action: PendingAction) {
        self.entries.insert(action.action_id, action);
    }

    pub fn get(&self, id: Uuid) -> Option<PendingAction> {
        self.entries.get(&id).map(|r| r.clone())
    }

    pub fn remove(&self, id: Uuid) -> Option<PendingAction> {
        self.entries.remove(&id).map(|(_, v)| v)
    }

    /// Atomically transition the action's status under the entry-lock.
    ///
    /// `pred` decides the new status given the current action; returning
    /// `Err` aborts the transition without mutating the store. The returned
    /// `PendingAction` reflects the post-transition state.
    pub fn try_transition<F>(&self, id: Uuid, pred: F) -> Result<PendingAction, ApprovalError>
    where
        F: FnOnce(&PendingAction) -> Result<ApprovalStatus, ApprovalError>,
    {
        let mut outcome: Option<Result<PendingAction, ApprovalError>> = None;
        self.entries
            .entry(id)
            .and_modify(|action| match pred(action) {
                Ok(new_status) => {
                    action.status = new_status;
                    outcome = Some(Ok(action.clone()));
                }
                Err(e) => {
                    outcome = Some(Err(e));
                }
            });
        outcome.unwrap_or(Err(ApprovalError::NotFound(id)))
    }

    /// Sweep expired-and-still-Proposed entries. Returns the removed actions
    /// so the caller can fire `action_expired` audit events.
    pub fn cleanup_expired(&self, now: DateTime<Utc>) -> Vec<PendingAction> {
        let expired_ids: Vec<Uuid> = self
            .entries
            .iter()
            .filter(|r| r.status == ApprovalStatus::Proposed && r.expires_at < now)
            .map(|r| r.action_id)
            .collect();

        let mut out = Vec::with_capacity(expired_ids.len());
        for id in expired_ids {
            if let Some((_, mut action)) = self.entries.remove(&id) {
                action.status = ApprovalStatus::Expired;
                out.push(action);
            }
        }
        out
    }
}

/// Background loop: every 60s, sweep expired actions and publish
/// `action_expired` events. Exits when `shutdown_rx` flips to `true`.
///
/// Holds `Arc<ApprovalStore>` (NOT `Arc<AppState>`) so PR-4's `serve()`
/// shutdown can `Arc::try_unwrap` state cleanly while this task drains.
pub async fn run_cleanup_task(
    store: Arc<ApprovalStore>,
    audit: Arc<AuditPublisher>,
    mut shutdown_rx: watch::Receiver<bool>,
) {
    let mut ticker = tokio::time::interval(Duration::from_secs(CLEANUP_INTERVAL_SECONDS));
    ticker.tick().await; // skip immediate first fire

    loop {
        tokio::select! {
            biased;
            _ = shutdown_rx.changed() => {
                if *shutdown_rx.borrow() {
                    tracing::info!("approval-store cleanup task shutting down");
                    return;
                }
            }
            _ = ticker.tick() => {
                let expired = store.cleanup_expired(Utc::now());
                if !expired.is_empty() {
                    tracing::info!(count = expired.len(), "swept expired pending actions");
                }
                for action in expired {
                    audit.expired(&action).await;
                }
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::gateway::approval::types::PendingAction;
    use crate::gateway::auth::AuthScope;
    use chrono::Duration as ChronoDuration;
    use serde_json::json;

    fn make_action(status: ApprovalStatus, expires_in: ChronoDuration) -> PendingAction {
        let now = Utc::now();
        PendingAction {
            action_id: Uuid::new_v4(),
            correlation_id: "cid".into(),
            user_id: "alice".into(),
            scope: AuthScope::ReadAndAct,
            tool_name: "create_company".into(),
            tool_args: json!({"name": "Acme"}),
            server_label: "crm".into(),
            proposed_at: now,
            expires_at: now + expires_in,
            status,
            executed_result: None,
            executed_duration_ms: None,
        }
    }

    #[test]
    fn insert_then_get_returns_action() {
        let store = ApprovalStore::new(Duration::from_secs(900));
        let action = make_action(ApprovalStatus::Proposed, ChronoDuration::minutes(15));
        let id = action.action_id;
        store.insert(action.clone());
        let got = store.get(id).expect("inserted action retrievable");
        assert_eq!(got.action_id, action.action_id);
        assert_eq!(got.tool_name, action.tool_name);
    }

    #[test]
    fn cleanup_removes_only_expired_proposed() {
        let store = ApprovalStore::new(Duration::from_secs(900));
        let fresh = make_action(ApprovalStatus::Proposed, ChronoDuration::minutes(15));
        let expired_proposed = make_action(ApprovalStatus::Proposed, ChronoDuration::minutes(-1));
        let expired_approved = make_action(ApprovalStatus::Approved, ChronoDuration::minutes(-1));
        let fresh_id = fresh.action_id;
        let expired_proposed_id = expired_proposed.action_id;
        let expired_approved_id = expired_approved.action_id;
        store.insert(fresh);
        store.insert(expired_proposed);
        store.insert(expired_approved);

        let swept = store.cleanup_expired(Utc::now());
        assert_eq!(swept.len(), 1, "only expired+proposed swept");
        assert_eq!(swept[0].action_id, expired_proposed_id);
        assert_eq!(swept[0].status, ApprovalStatus::Expired);
        assert!(store.get(fresh_id).is_some(), "fresh proposed retained");
        assert!(
            store.get(expired_approved_id).is_some(),
            "expired but Approved retained — must wait for execute",
        );
        assert!(
            store.get(expired_proposed_id).is_none(),
            "expired+proposed removed",
        );
    }

    #[tokio::test]
    async fn try_transition_concurrent_only_first_wins() {
        let store = Arc::new(ApprovalStore::new(Duration::from_secs(900)));
        let action = make_action(ApprovalStatus::Proposed, ChronoDuration::minutes(15));
        let id = action.action_id;
        store.insert(action);

        // Two parallel callers race to transition Proposed → Approved.
        // The first wins; the second sees status != Proposed and returns
        // AlreadyDecided.
        let s1 = store.clone();
        let s2 = store.clone();
        let pred = |a: &PendingAction| -> Result<ApprovalStatus, ApprovalError> {
            if a.status != ApprovalStatus::Proposed {
                return Err(ApprovalError::AlreadyDecided(a.status.clone()));
            }
            Ok(ApprovalStatus::Approved)
        };
        let (r1, r2) = tokio::join!(
            tokio::task::spawn_blocking(move || s1.try_transition(id, pred)),
            tokio::task::spawn_blocking(move || s2.try_transition(id, pred)),
        );
        let r1 = r1.unwrap();
        let r2 = r2.unwrap();

        let ok_count = [&r1, &r2].iter().filter(|r| r.is_ok()).count();
        let err_count = [&r1, &r2].iter().filter(|r| r.is_err()).count();
        assert_eq!(ok_count, 1, "exactly one transition succeeds");
        assert_eq!(err_count, 1, "the other returns AlreadyDecided");
    }

    #[test]
    fn try_transition_propagates_predicate_error() {
        let store = ApprovalStore::new(Duration::from_secs(900));
        let action = make_action(ApprovalStatus::Approved, ChronoDuration::minutes(15));
        let id = action.action_id;
        store.insert(action);

        let result = store.try_transition(id, |a| {
            Err::<ApprovalStatus, _>(ApprovalError::AlreadyDecided(a.status.clone()))
        });
        assert!(matches!(result, Err(ApprovalError::AlreadyDecided(_))));
        // status unchanged in store
        assert_eq!(store.get(id).unwrap().status, ApprovalStatus::Approved);
    }

    #[test]
    fn try_transition_unknown_id_returns_not_found() {
        let store = ApprovalStore::new(Duration::from_secs(900));
        let result = store.try_transition(Uuid::new_v4(), |_| Ok(ApprovalStatus::Approved));
        assert!(matches!(result, Err(ApprovalError::NotFound(_))));
    }

    #[tokio::test]
    async fn cleanup_task_exits_on_shutdown_signal() {
        let store = Arc::new(ApprovalStore::new(Duration::from_secs(900)));
        let audit = Arc::new(AuditPublisher::new(None));
        let (tx, rx) = watch::channel(false);

        let handle = tokio::spawn(run_cleanup_task(store, audit, rx));

        // Give the task a moment to enter its select loop, then shut down.
        tokio::time::sleep(Duration::from_millis(50)).await;
        tx.send(true).expect("shutdown channel still open");
        let result = tokio::time::timeout(Duration::from_millis(500), handle)
            .await
            .expect("cleanup task exits within 500ms of shutdown signal");
        result.expect("task panic-free");
    }
}
