//! Gateway layer: auth, policy, approval, audit.
//!
//! Pairs with `crate::agent` (LLM, prompts, orchestrator) to form the hybrid
//! agentic-gateway architecture documented in `.claude/rules/R2_PLANNING_AGENTIC_GATEWAY.md`.
//!
//! Auth lands first; approval store + state-machine + audit publisher follow
//! in PR-2.

pub mod approval;
pub mod auth;
