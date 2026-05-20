use super::*;
use serde_json::json;

fn make_action() -> PendingAction {
    let now = Utc::now();
    PendingAction {
        action_id: Uuid::new_v4(),
        correlation_id: "cid-abc".to_string(),
        user_id: "drupal-uid-42".to_string(),
        scope: AuthScope::ReadAndAct,
        tool_name: "create_company".to_string(),
        tool_args: json!({"name": "Acme NV"}),
        server_label: "crm".to_string(),
        proposed_at: now,
        expires_at: now + chrono::Duration::minutes(15),
        status: ApprovalStatus::Proposed,
        executed_result: None,
        executed_duration_ms: None,
    }
}

#[test]
fn approval_status_serializes_to_lowercase() {
    let json = serde_json::to_string(&ApprovalStatus::Proposed).unwrap();
    assert_eq!(json, "\"proposed\"");
    let json = serde_json::to_string(&ApprovalStatus::Executed).unwrap();
    assert_eq!(json, "\"executed\"");
}

#[test]
fn pending_action_round_trips_via_serde() {
    let action = make_action();
    let json = serde_json::to_string(&action).unwrap();
    let back: PendingAction = serde_json::from_str(&json).unwrap();
    assert_eq!(back.action_id, action.action_id);
    assert_eq!(back.tool_name, action.tool_name);
    assert_eq!(back.scope, action.scope);
    assert_eq!(back.status, action.status);
}

#[test]
fn approval_error_display_includes_context() {
    let err = ApprovalError::AlreadyDecided(ApprovalStatus::Approved);
    assert!(format!("{err}").contains("Approved"));
    let err = ApprovalError::WrongUser {
        proposer: "alice".to_string(),
        caller: "mallory".to_string(),
    };
    let s = format!("{err}");
    assert!(s.contains("alice"));
    assert!(s.contains("mallory"));
}
