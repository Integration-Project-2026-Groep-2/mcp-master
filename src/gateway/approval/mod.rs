//! Approval primitives for the actionable agent.
//!
//! - [`types`]: data types — `PendingAction`, `ApprovalStatus`, `ApprovalError`.
//! - `state` (commit 2): in-memory store with TTL.
//! - `flow` (commit 4): state-machine combining store + audit.
//!
//! PR-2 provides these as standalone library modules; PR-3 integrates with
//! the orchestrator and PR-4 wires them into `AppState` + `serve()`.

pub mod state;
pub mod types;

#[allow(unused_imports)] // wired into flow + http_api in subsequent commits
pub use state::{ApprovalStore, run_cleanup_task};
#[allow(unused_imports)]
pub use types::{ApprovalError, ApprovalStatus, PendingAction, PendingActionDraft};
