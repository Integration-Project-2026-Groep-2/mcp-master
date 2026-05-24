//! Fix-flow as an async job.
//!
//! `POST /fix` (handler in `http_api`) spawns this on a background task: it runs
//! the agent in Actionable mode so the Controlroom `request_changes_with_files`
//! write-tool is proposed via the approval flow (a pending `action_id`), then
//! publishes `fix_proposed` on `ai.events`. No held HTTP connection; approval
//! stays on the unchanged `/chat/approve`.

use serde::Deserialize;
use serde_json::{Value, json};
use tokio::sync::watch;

use crate::agent::llm::{ContentBlock, LlmClient, Message, Role, ToolSpec};
use crate::agent::modes::{AgentMode, DispatchContext};
use crate::agent::orchestrator::{self, McpExecutor, RunOutcome};
use crate::agent::prompts::SETUP_PROMPT;
use crate::rabbitmq::publisher::Publisher;

/// Wall-clock cap on one fix run (matches the SSE stream deadline).
const FIX_FLOW_TIMEOUT_SECS: u64 = 600;

#[derive(Debug, Deserialize)]
pub struct FixFlowRequest {
    pub service: String,
    pub suggested_action: String,
    #[serde(default)]
    pub root_cause: Option<String>,
    /// The incident's correlation id, so `fix_proposed` links back to it; the
    /// handler generates a fresh one when absent.
    #[serde(default)]
    pub correlation_id: Option<String>,
}

/// Actionable seed prompt: investigate the service repo, read-before-write,
/// then propose a `request_changes_with_files` PR.
pub fn seed_prompt(req: &FixFlowRequest) -> String {
    let root = req.root_cause.as_deref().unwrap_or("(not provided)");
    format!(
        "A '{service}' service incident needs a code fix.\n\
         Root cause: {root}\n\
         Suggested action: {suggested}\n\n\
         Propose a concrete fix as a pull request:\n\
         1. Investigate the '{service}' repository with the Controlroom GitHub tools \
         (recent commits/deploys for context).\n\
         2. READ each file you intend to change with fetch_file BEFORE writing it — \
         never reconstruct it from memory; the write replaces the whole file.\n\
         3. Open a PR to the default branch via request_changes_with_files with the \
         minimal corrected file(s) addressing the root cause. Use the exact owner/repo \
         from the known-repositories list.",
        service = req.service,
        root = root,
        suggested = req.suggested_action,
    )
}

/// Build the `fix_proposed` payload from a completed run. `status` is
/// `proposed` (a write awaits approval) or `no_action` (nothing proposed).
pub fn outcome_event(outcome: &RunOutcome, service: &str, correlation_id: &str) -> Value {
    match outcome
        .tool_trace
        .iter()
        .find(|t| t.status.as_deref() == Some("pending"))
    {
        Some(t) => json!({
            "correlation_id": correlation_id,
            "service": service,
            "status": "proposed",
            "action_id": t.action_id,
            "tool": t.tool,
            "summary": outcome.answer,
        }),
        None => json!({
            "correlation_id": correlation_id,
            "service": service,
            "status": "no_action",
            "summary": outcome.answer,
        }),
    }
}

fn failed_event(service: &str, correlation_id: &str, reason: &str) -> Value {
    json!({
        "correlation_id": correlation_id,
        "service": service,
        "status": "failed",
        "reason": reason,
    })
}

/// Background job: run the Actionable agent then publish `fix_proposed`. Glue —
/// the pure parts (`seed_prompt`, `outcome_event`) are unit-tested; the
/// orchestrator and publisher are covered in their own modules.
#[allow(clippy::too_many_arguments)]
pub async fn run_fix_flow(
    llm: &dyn LlmClient,
    mcp: &dyn McpExecutor,
    tool_specs: &[ToolSpec],
    mode: &AgentMode,
    publisher: Option<&Publisher>,
    req: &FixFlowRequest,
    ctx: &DispatchContext,
    max_iterations: usize,
    max_tokens: u32,
    mut shutdown_rx: watch::Receiver<bool>,
) {
    let system_prompt = format!(
        "{SETUP_PROMPT}{}",
        crate::agent::repo_map::repo_hints_prompt()
    );
    let messages = vec![Message {
        role: Role::User,
        content: vec![ContentBlock::Text {
            text: seed_prompt(req),
        }],
    }];

    let run = tokio::time::timeout(
        std::time::Duration::from_secs(FIX_FLOW_TIMEOUT_SECS),
        orchestrator::run_with_messages_in_mode(
            messages,
            &system_prompt,
            llm,
            mcp,
            tool_specs,
            max_iterations,
            max_tokens,
            mode,
            ctx,
        ),
    );

    // Publish runs after this select!, not inside an arm, so a shutdown cancels
    // only the agent run — never the fix_proposed event mid-flight.
    let payload = tokio::select! {
        biased;
        _ = shutdown_rx.changed() => {
            tracing::info!(correlation_id = %ctx.correlation_id, "fix-flow aborted by shutdown");
            return;
        }
        result = run => match result {
            Ok(Ok(outcome)) => outcome_event(&outcome, &req.service, &ctx.correlation_id),
            Ok(Err(e)) => {
                tracing::warn!(correlation_id = %ctx.correlation_id, "fix-flow run failed: {e:#}");
                failed_event(&req.service, &ctx.correlation_id, "error")
            }
            Err(_) => {
                tracing::warn!(correlation_id = %ctx.correlation_id, "fix-flow run timed out");
                failed_event(&req.service, &ctx.correlation_id, "timeout")
            }
        }
    };

    let Some(p) = publisher else {
        return;
    };
    if let Err(e) = p.publish_event("fix_proposed", payload).await {
        tracing::warn!(correlation_id = %ctx.correlation_id, "fix_proposed publish failed: {e:#}");
    }
}

#[cfg(test)]
mod tests;
