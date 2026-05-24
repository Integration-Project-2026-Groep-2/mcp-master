use super::*;
use crate::agent::llm::TokenUsage;
use crate::agent::orchestrator::ToolCallTrace;

fn sample_req() -> FixFlowRequest {
    FixFlowRequest {
        service: "crm".into(),
        suggested_action: "Remove the intentional RuntimeError in heartbeat.py".into(),
        root_cause: Some("test crash".into()),
        correlation_id: Some("corr-1".into()),
    }
}

#[test]
fn seed_prompt_drives_read_before_write_pr() {
    let p = seed_prompt(&sample_req());
    assert!(p.contains("crm"));
    assert!(p.contains("Remove the intentional RuntimeError"));
    assert!(p.contains("fetch_file"));
    assert!(p.contains("request_changes_with_files"));
    assert!(p.contains("READ"));
}

fn trace(status: Option<&str>, action_id: Option<&str>) -> ToolCallTrace {
    ToolCallTrace {
        tool: "request_changes_with_files".into(),
        server: "controlroom".into(),
        ms: 0,
        ok: true,
        error: None,
        args: None,
        status: status.map(Into::into),
        action_id: action_id.map(Into::into),
    }
}

fn outcome(traces: Vec<ToolCallTrace>) -> RunOutcome {
    RunOutcome {
        answer: "done".into(),
        tool_trace: traces,
        tokens: TokenUsage::default(),
        iterations: 1,
    }
}

#[test]
fn outcome_event_proposed_carries_action_id() {
    let o = outcome(vec![trace(Some("pending"), Some("act-1"))]);
    let ev = outcome_event(&o, "crm", "corr-1");
    assert_eq!(ev["status"], "proposed");
    assert_eq!(ev["action_id"], "act-1");
    assert_eq!(ev["service"], "crm");
    assert_eq!(ev["correlation_id"], "corr-1");
}

#[test]
fn outcome_event_no_action_when_nothing_pending() {
    // A normal dispatched read (no pending write) -> no proposal.
    let o = outcome(vec![trace(None, None)]);
    let ev = outcome_event(&o, "crm", "corr-1");
    assert_eq!(ev["status"], "no_action");
    assert!(ev.get("action_id").is_none());
}

#[test]
fn failed_event_carries_reason() {
    let ev = failed_event("crm", "corr-1", "timeout");
    assert_eq!(ev["status"], "failed");
    assert_eq!(ev["reason"], "timeout");
    assert_eq!(ev["service"], "crm");
    assert_eq!(ev["correlation_id"], "corr-1");
}
