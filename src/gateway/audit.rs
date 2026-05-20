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
mod tests;
