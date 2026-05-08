//! Audit publisher for approval-flow events.
//!
//! Emits five event types on the existing `ai.events` topic exchange:
//! `action_proposed`, `action_approved`, `action_rejected`,
//! `action_expired`, `action_executed`. Each event rides the standard
//! envelope `{event, source, timestamp, payload}` constructed by
//! [`Publisher::publish_event`] — `AuditPublisher` only contributes the
//! per-event-type payload shape.
//!
//! Skip-warn pattern: `inner: Option<Arc<Publisher>>`. When the broker is
//! unreachable at startup, `AppState.publisher` is `None` and every audit
//! event becomes a no-op + WARN log. Write-tools keep working; consumers
//! just don't hear about it.

#![allow(dead_code)] // Wired into ApprovalFlow + http_api in subsequent PRs.

use std::sync::Arc;

use serde_json::{Value, json};

use crate::gateway::approval::types::PendingAction;
use crate::rabbitmq::publisher::Publisher;

pub struct AuditPublisher {
    inner: Option<Arc<Publisher>>,
}

impl AuditPublisher {
    pub fn new(publisher: Option<Arc<Publisher>>) -> Self {
        Self { inner: publisher }
    }

    pub async fn proposed(&self, action: &PendingAction) {
        self.publish("action_proposed", build_proposed(action))
            .await;
    }

    pub async fn approved(&self, action: &PendingAction) {
        self.publish("action_approved", build_approved(action))
            .await;
    }

    pub async fn rejected(&self, action: &PendingAction, reason: Option<&str>) {
        self.publish("action_rejected", build_rejected(action, reason))
            .await;
    }

    pub async fn expired(&self, action: &PendingAction) {
        self.publish("action_expired", build_expired(action)).await;
    }

    pub async fn executed(&self, action: &PendingAction, result: &str, duration_ms: u64) {
        self.publish(
            "action_executed",
            build_executed(action, result, duration_ms),
        )
        .await;
    }

    async fn publish(&self, event: &str, payload: Value) {
        let Some(publisher) = self.inner.as_ref() else {
            tracing::warn!(event, "audit publisher not bound — skipping");
            return;
        };
        if let Err(e) = publisher.publish_event(event, payload).await {
            tracing::warn!(event, error = %e, "failed to publish audit event");
        }
    }
}

// Pure-function payload builders — tests target these directly so we can
// assert the wire shape without needing a live broker.

fn build_proposed(action: &PendingAction) -> Value {
    json!({
        "action_id": action.action_id,
        "correlation_id": action.correlation_id,
        "user_id": action.user_id,
        "scope": action.scope,
        "tool": action.tool_name,
        "server": action.server_label,
        "expires_at": action.expires_at.to_rfc3339(),
    })
}

fn build_approved(action: &PendingAction) -> Value {
    json!({
        "action_id": action.action_id,
        "correlation_id": action.correlation_id,
        "user_id": action.user_id,
        "tool": action.tool_name,
        "server": action.server_label,
    })
}

fn build_rejected(action: &PendingAction, reason: Option<&str>) -> Value {
    let mut payload = json!({
        "action_id": action.action_id,
        "correlation_id": action.correlation_id,
        "user_id": action.user_id,
        "tool": action.tool_name,
        "server": action.server_label,
    });
    if let Some(r) = reason {
        payload
            .as_object_mut()
            .expect("json! root is object")
            .insert("reason".to_string(), Value::String(r.to_string()));
    }
    payload
}

fn build_expired(action: &PendingAction) -> Value {
    json!({
        "action_id": action.action_id,
        "correlation_id": action.correlation_id,
        "user_id": action.user_id,
        "tool": action.tool_name,
        "server": action.server_label,
        "expired_at": action.expires_at.to_rfc3339(),
    })
}

fn build_executed(action: &PendingAction, result: &str, duration_ms: u64) -> Value {
    json!({
        "action_id": action.action_id,
        "correlation_id": action.correlation_id,
        "user_id": action.user_id,
        "tool": action.tool_name,
        "server": action.server_label,
        "result": result,
        "duration_ms": duration_ms,
    })
}

#[cfg(test)]
mod tests {
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
}
