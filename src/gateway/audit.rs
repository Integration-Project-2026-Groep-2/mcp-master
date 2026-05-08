//! Audit publisher — populated in the next commit.
//!
//! Stub for now so `state::run_cleanup_task` can hold an
//! `Arc<AuditPublisher>` and call its eventual API. Each method here is a
//! placeholder no-op; commit 3 fills them in with the real
//! `Publisher::publish_event` calls.

#![allow(dead_code)]

use std::sync::Arc;

use crate::gateway::approval::types::PendingAction;
use crate::rabbitmq::publisher::Publisher;

pub struct AuditPublisher {
    inner: Option<Arc<Publisher>>,
}

impl AuditPublisher {
    pub fn new(publisher: Option<Arc<Publisher>>) -> Self {
        Self { inner: publisher }
    }

    pub async fn proposed(&self, _action: &PendingAction) {}
    pub async fn approved(&self, _action: &PendingAction) {}
    pub async fn rejected(&self, _action: &PendingAction, _reason: Option<&str>) {}
    pub async fn expired(&self, _action: &PendingAction) {}
    pub async fn executed(&self, _action: &PendingAction, _result: &str, _duration_ms: u64) {}
}
