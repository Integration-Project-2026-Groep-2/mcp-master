//! Approval primitives for the actionable agent.
//!
//! - [`types`]: data types — `PendingAction`, `ApprovalStatus`, `ApprovalError`.
//! - `state` (commit 2): in-memory store with TTL.
//! - `flow` (commit 4): state-machine combining store + audit.
//!
//! PR-2 provides these as standalone library modules; PR-3 integrates with
//! the orchestrator and PR-4 wires them into `AppState` + `serve()`.

pub mod types;

// Re-exports become reachable in PR-3/PR-4 once `flow` and `state` import
// them; suppressed until then.
#[allow(unused_imports)]
pub use types::{ApprovalError, ApprovalStatus, PendingAction, PendingActionDraft};
