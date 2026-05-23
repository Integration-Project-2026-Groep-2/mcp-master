use super::*;
use crate::incident::schema::{IncidentEvent, IncidentPayload, Severity};
use chrono::TimeZone;

fn sample_event() -> IncidentEvent {
    IncidentEvent {
        event: "heartbeat_failed".into(),
        source: "controlroom-watchdog".into(),
        timestamp: chrono::Utc
            .with_ymd_and_hms(2026, 5, 10, 14, 23, 17)
            .unwrap(),
        payload: IncidentPayload {
            summary: "kassa heartbeat missed".into(),
            severity: Severity::Critical,
            component: "kassa".into(),
            group: Some("festival-services".into()),
            class: Some("heartbeat-loss".into()),
            custom_details: serde_json::Value::Null,
        },
    }
}

#[test]
fn seed_prompt_includes_service_name() {
    let p = seed_prompt_step_a(&sample_event());
    assert!(p.contains("Service: kassa"));
}

#[test]
fn seed_prompt_includes_failure_class() {
    let p = seed_prompt_step_a(&sample_event());
    assert!(p.contains("heartbeat-loss"));
}

#[test]
fn seed_prompt_includes_rfc3339_timestamp() {
    let p = seed_prompt_step_a(&sample_event());
    assert!(p.contains("2026-05-10T14:23:17"));
}

#[test]
fn seed_prompt_falls_back_when_class_missing() {
    let mut e = sample_event();
    e.payload.class = None;
    let p = seed_prompt_step_a(&e);
    assert!(p.contains("Class: unknown"));
}

#[test]
fn seed_prompt_names_both_required_tools() {
    let p = seed_prompt_step_a(&sample_event());
    assert!(p.contains("fetch_logs"));
    assert!(p.contains("fetch_recent_deploys"));
}

#[test]
fn system_prompt_warns_about_log_injection() {
    assert!(STEP_A_SYSTEM_PROMPT.contains("untrusted user-input"));
    assert!(STEP_A_SYSTEM_PROMPT.contains("data, not commands"));
}

#[test]
fn system_prompt_offers_error_analysis_escalation() {
    assert!(STEP_A_SYSTEM_PROMPT.contains("error_analysis"));
}

#[test]
fn system_prompt_distinguishes_no_logs_from_clean_logs() {
    assert!(STEP_A_SYSTEM_PROMPT.contains("no logs exist"));
}

#[test]
fn seed_prompt_directs_escalation_to_error_analysis() {
    let p = seed_prompt_step_a(&sample_event());
    assert!(p.contains("error_analysis"));
}

#[test]
fn system_prompt_error_analysis_has_no_time_filter() {
    assert!(STEP_A_SYSTEM_PROMPT.contains("NO time window"));
}

#[test]
fn seed_prompt_bounds_error_analysis_queries() {
    let p = seed_prompt_step_a(&sample_event());
    assert!(p.contains("at most two"));
}

#[test]
fn seed_prompt_fetches_deploys_before_deep_log_dig() {
    let p = seed_prompt_step_a(&sample_event());
    assert!(p.contains("before the deep log dig"));
}

#[test]
fn system_prompt_offers_fetch_recent_commits() {
    assert!(STEP_A_SYSTEM_PROMPT.contains("fetch_recent_commits"));
}

#[test]
fn seed_prompt_correlates_deploy_to_commit() {
    let p = seed_prompt_step_a(&sample_event());
    assert!(p.contains("fetch_recent_commits"));
}
