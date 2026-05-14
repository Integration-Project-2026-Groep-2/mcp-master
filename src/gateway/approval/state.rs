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

    /// Persist execution metadata after a successful Approved → Executed
    /// transition. Separate from `try_transition` because the result + duration
    /// are only known *after* the dispatched tool returns.
    pub fn set_execution_metadata(&self, id: Uuid, result: String, duration_ms: u64) {
        if let Some(mut entry) = self.entries.get_mut(&id) {
            entry.executed_result = Some(result);
            entry.executed_duration_ms = Some(duration_ms);
        }
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
mod tests;
