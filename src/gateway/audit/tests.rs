use super::*;
use crate::gateway::approval::types::ApprovalStatus;
use crate::gateway::auth::AuthScope;
use chrono::Utc;
use uuid::Uuid;

fn fixture() -> PendingAction {
    let now = Utc::now();
    PendingAction {
        action_id: Uuid::nil(),
        correlation_id: "cid-abc".into(),
        user_id: "alice".into(),
        scope: AuthScope::ReadAndAct,
        tool_name: "create_company".into(),
        tool_args: json!({"name": "Acme"}),
        server_label: "crm".into(),
        proposed_at: now,
        expires_at: now + chrono::Duration::minutes(15),
        status: ApprovalStatus::Proposed,
        executed_result: None,
        executed_duration_ms: None,
    }
}

#[test]
fn payload_for_proposed_contains_required_fields() {
    let p = build_proposed(&fixture());
    let obj = p.as_object().unwrap();
    for k in [
        "action_id",
        "correlation_id",
        "user_id",
        "scope",
        "tool",
        "server",
        "expires_at",
    ] {
        assert!(obj.contains_key(k), "missing key: {k}");
    }
    // tool_args and tool_args contents are NOT in the payload —
    // they would echo PII (emails, VAT) into the audit feed.
    assert!(!obj.contains_key("tool_args"));
}

#[test]
fn payload_for_executed_includes_result_and_duration() {
    let p = build_executed(&fixture(), "ok: 1 record affected", 412);
    let obj = p.as_object().unwrap();
    assert_eq!(
        obj.get("result").and_then(Value::as_str),
        Some("ok: 1 record affected")
    );
    assert_eq!(obj.get("duration_ms").and_then(Value::as_u64), Some(412));
}

#[test]
fn payload_for_rejected_includes_reason_when_some() {
    let p = build_rejected(&fixture(), Some("user changed mind"));
    let obj = p.as_object().unwrap();
    assert_eq!(
        obj.get("reason").and_then(Value::as_str),
        Some("user changed mind"),
    );
}

#[test]
fn payload_for_rejected_omits_reason_when_none() {
    let p = build_rejected(&fixture(), None);
    let obj = p.as_object().unwrap();
    assert!(!obj.contains_key("reason"));
}

#[tokio::test]
async fn publish_with_no_inner_publisher_is_a_noop() {
    // Skip-warn path: AuditPublisher::new(None) → every method returns
    // without panicking. Asserts the absence of the inner Arc<Publisher>
    // does NOT take down the agent.
    let audit = AuditPublisher::new(None);
    let action = fixture();
    audit.proposed(&action).await;
    audit.approved(&action).await;
    audit.rejected(&action, Some("test")).await;
    audit.expired(&action).await;
    audit.executed(&action, "ok", 100).await;
}

#[test]
fn scope_serializes_to_human_readable_variant_name() {
    // AuthScope serializes as "Read" / "ReadAndAct" by default —
    // documented here so consumers know what to filter on.
    let p = build_proposed(&fixture());
    let scope = p.as_object().unwrap().get("scope").unwrap();
    assert_eq!(scope.as_str(), Some("ReadAndAct"));
}
