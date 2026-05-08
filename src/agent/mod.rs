//! Agent layer: LLM reasoning, prompts, and tool-loop orchestration.
//!
//! Pairs with `crate::gateway` (auth, policy, approval) to form the hybrid
//! agentic-gateway architecture documented in `.claude/rules/R2_PLANNING_AGENTIC_GATEWAY.md`.

pub mod orchestrator;
