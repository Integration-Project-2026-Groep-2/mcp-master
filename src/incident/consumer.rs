use std::sync::Arc;
use std::time::{Duration, Instant};

use anyhow::{Context, Result};
use futures_util::StreamExt;
use lapin::{
    Connection, ConnectionProperties, ExchangeKind,
    options::{
        BasicAckOptions, BasicConsumeOptions, BasicNackOptions, ExchangeDeclareOptions,
        QueueBindOptions, QueueDeclareOptions,
    },
    types::FieldTable,
};
use tokio::sync::watch;
use uuid::Uuid;

use super::budget::{Budget, BudgetOutcome};
use super::debounce::Debouncer;
use super::diagnose::DiagnosePipeline;
use super::schema::{IncidentDiagnosis, IncidentEvent};
use crate::rabbitmq::config::RabbitMqConfig;
use crate::rabbitmq::publisher::Publisher;
use crate::retry::backoff_with_jitter;

const QUEUE_NAME: &str = "mcp-master.incidents";
const ROUTING_KEY: &str = "event.heartbeat_failed";
const CONSUMER_TAG: &str = "mcp-master-incident";
const SKIP_EVENT_NAME: &str = "incident_skipped";
const DIAGNOSED_EVENT_NAME: &str = "incident_diagnosed";
const CIRCUIT_OPEN_EVENT_NAME: &str = "incident_circuit_open";

/// Consume `event.heartbeat_failed` until shutdown, reconnecting through
/// transient broker outages. Mirrors `rabbitmq::consumer::run` but with a
/// durable named queue + manual ack so an mcp-master restart during an
/// incident-piek doesn't drop deliveries — the broker buffers until we
/// reconnect.
///
/// `publisher` is optional: when absent (skip-warn path), skip-events are
/// only logged. The Debouncer lives here (not per-session) so it persists
/// across reconnects — a service flapping during a broker-blip shouldn't
/// reset its debounce slot.
///
/// `pipeline` is optional too: when absent, accepted events are logged but
/// no Step A+B run. Tests pass `None` to skip the (heavy to mock) pipeline;
/// production wires `Some(DefaultDiagnosePipeline::new(state))`.
pub async fn run(
    config: RabbitMqConfig,
    publisher: Option<Arc<Publisher>>,
    pipeline: Option<Arc<dyn DiagnosePipeline>>,
    mut shutdown_rx: watch::Receiver<bool>,
) -> Result<()> {
    let debouncer = Arc::new(Debouncer::from_env());
    let budget = Arc::new(Budget::from_env());
    tracing::info!(
        window_s = debouncer.window().as_secs(),
        max_per_hour = budget.max_per_hour(),
        "incident consumer initialised"
    );

    let mut attempt: u32 = 0;
    loop {
        if *shutdown_rx.borrow() {
            return Ok(());
        }

        match consume_session(
            &config,
            &debouncer,
            &budget,
            publisher.as_ref(),
            pipeline.as_ref(),
            &mut shutdown_rx,
        )
        .await
        {
            Ok(()) => return Ok(()),
            Err(e) => {
                attempt = attempt.saturating_add(1);
                let delay = backoff_with_jitter(attempt);
                tracing::warn!(
                    attempt,
                    delay_ms = delay.as_millis() as u64,
                    "incident consumer connection lost: {e:#} — reconnecting after backoff"
                );
                tokio::select! {
                    biased;
                    _ = shutdown_rx.changed() => {
                        if *shutdown_rx.borrow() {
                            tracing::info!("incident consumer shutting down during backoff");
                            return Ok(());
                        }
                    }
                    _ = tokio::time::sleep(delay) => {}
                }
            }
        }
    }
}

async fn consume_session(
    config: &RabbitMqConfig,
    debouncer: &Debouncer,
    budget: &Budget,
    publisher: Option<&Arc<Publisher>>,
    pipeline: Option<&Arc<dyn DiagnosePipeline>>,
    shutdown_rx: &mut watch::Receiver<bool>,
) -> Result<()> {
    let conn = Connection::connect(&config.url, ConnectionProperties::default())
        .await
        .context("AMQP connection")?;
    let channel = conn.create_channel().await.context("AMQP channel")?;

    channel
        .exchange_declare(
            &config.exchange,
            ExchangeKind::Topic,
            ExchangeDeclareOptions {
                durable: true,
                ..Default::default()
            },
            FieldTable::default(),
        )
        .await
        .context("AMQP exchange_declare")?;

    channel
        .queue_declare(
            QUEUE_NAME,
            QueueDeclareOptions {
                durable: true,
                ..Default::default()
            },
            FieldTable::default(),
        )
        .await
        .context("AMQP queue_declare (durable named)")?;

    channel
        .queue_bind(
            QUEUE_NAME,
            &config.exchange,
            ROUTING_KEY,
            QueueBindOptions::default(),
            FieldTable::default(),
        )
        .await
        .context("AMQP queue_bind")?;

    let mut consumer = channel
        .basic_consume(
            QUEUE_NAME,
            CONSUMER_TAG,
            BasicConsumeOptions::default(),
            FieldTable::default(),
        )
        .await
        .context("AMQP basic_consume")?;

    tracing::info!(
        exchange = %config.exchange,
        queue = QUEUE_NAME,
        routing_key = ROUTING_KEY,
        "incident consumer started"
    );

    loop {
        tokio::select! {
            biased;
            _ = shutdown_rx.changed() => {
                if *shutdown_rx.borrow() {
                    tracing::info!("incident consumer shutting down");
                    return Ok(());
                }
            }
            delivery = consumer.next() => match delivery {
                Some(Ok(msg)) => match handle_delivery(&msg.data, debouncer, budget, publisher, pipeline).await {
                    Ok(()) => {
                        if let Err(e) = msg.ack(BasicAckOptions::default()).await {
                            tracing::warn!("incident ack failed: {e:#}");
                        }
                    }
                    Err(e) => {
                        tracing::error!("incident handler failed: {e:#}");
                        if let Err(nack_err) = msg
                            .nack(BasicNackOptions {
                                requeue: false,
                                multiple: false,
                            })
                            .await
                        {
                            tracing::warn!("incident nack failed: {nack_err:#}");
                        }
                    }
                },
                Some(Err(e)) => {
                    return Err(anyhow::Error::from(e).context("incident consumer delivery error"));
                }
                None => {
                    anyhow::bail!("incident consumer stream ended unexpectedly");
                }
            }
        }
    }
}

async fn handle_delivery(
    body: &[u8],
    debouncer: &Debouncer,
    budget: &Budget,
    publisher: Option<&Arc<Publisher>>,
    pipeline: Option<&Arc<dyn DiagnosePipeline>>,
) -> Result<()> {
    let received_at = Instant::now();
    let correlation_id = Uuid::new_v4().to_string();

    let evt: IncidentEvent =
        serde_json::from_slice(body).context("decoding IncidentEvent envelope")?;

    if !evt.payload.severity.is_actionable() {
        tracing::info!(
            correlation_id = %correlation_id,
            service = %evt.payload.component,
            severity = ?evt.payload.severity,
            "incident skipped (severity_too_low)"
        );
        publish_skip(publisher, &evt, &correlation_id, "severity_too_low", None).await;
        return Ok(());
    }

    if let Err(elapsed) = debouncer.check(&evt.payload.component) {
        tracing::info!(
            correlation_id = %correlation_id,
            service = %evt.payload.component,
            elapsed_s = elapsed.as_secs(),
            "incident skipped (debounced)"
        );
        publish_skip(publisher, &evt, &correlation_id, "debounced", Some(elapsed)).await;
        return Ok(());
    }

    if let BudgetOutcome::CircuitOpen { reset_at } = budget.try_consume() {
        let reset_in_s = reset_at.saturating_duration_since(Instant::now()).as_secs();
        tracing::warn!(
            correlation_id = %correlation_id,
            service = %evt.payload.component,
            reset_in_s,
            "incident circuit open — skipping diagnosis (budget exhausted)"
        );
        publish_circuit_open(publisher, &evt, &correlation_id, reset_in_s).await;
        return Ok(());
    }

    tracing::info!(
        correlation_id = %correlation_id,
        service = %evt.payload.component,
        severity = ?evt.payload.severity,
        timestamp = %evt.timestamp,
        "incident accepted for diagnosis"
    );

    let Some(pl) = pipeline else {
        tracing::warn!(
            correlation_id = %correlation_id,
            service = %evt.payload.component,
            "no DiagnosePipeline configured — Step A+B skipped"
        );
        return Ok(());
    };

    let pipeline_start = Instant::now();
    match pl.diagnose(&evt).await {
        Ok(diagnosis) => {
            let pipeline_ms = pipeline_start.elapsed().as_millis() as u64;
            let total_ms = received_at.elapsed().as_millis() as u64;
            tracing::info!(
                correlation_id = %correlation_id,
                service = %evt.payload.component,
                confidence = ?diagnosis.confidence,
                pipeline_ms,
                total_ms,
                "incident diagnosed"
            );
            publish_diagnosis(publisher, &evt, &correlation_id, &diagnosis).await;
        }
        Err(e) => {
            let pipeline_ms = pipeline_start.elapsed().as_millis() as u64;
            tracing::error!(
                correlation_id = %correlation_id,
                service = %evt.payload.component,
                pipeline_ms,
                "diagnose pipeline failed: {e:#}"
            );
        }
    }

    Ok(())
}

async fn publish_skip(
    publisher: Option<&Arc<Publisher>>,
    evt: &IncidentEvent,
    correlation_id: &str,
    reason: &str,
    elapsed: Option<Duration>,
) {
    let Some(p) = publisher else {
        return;
    };
    let mut payload = serde_json::json!({
        "service": evt.payload.component,
        "correlation_id": correlation_id,
        "reason": reason,
        "severity": evt.payload.severity,
        "original_summary": evt.payload.summary,
        "original_timestamp": evt.timestamp.to_rfc3339(),
    });
    if let Some(e) = elapsed {
        payload["since_last_diagnosis_seconds"] = serde_json::json!(e.as_secs());
    }
    if let Err(e) = p.publish_event(SKIP_EVENT_NAME, payload).await {
        tracing::warn!("publish_event({SKIP_EVENT_NAME}) failed: {e:#}");
    }
}

async fn publish_diagnosis(
    publisher: Option<&Arc<Publisher>>,
    event: &IncidentEvent,
    correlation_id: &str,
    diagnosis: &IncidentDiagnosis,
) {
    let Some(p) = publisher else {
        return;
    };
    let payload = serde_json::json!({
        "service": event.payload.component,
        "correlation_id": correlation_id,
        "severity": event.payload.severity,
        "diagnosis": diagnosis,
        "original_summary": event.payload.summary,
        "original_timestamp": event.timestamp.to_rfc3339(),
    });
    if let Err(e) = p.publish_event(DIAGNOSED_EVENT_NAME, payload).await {
        tracing::warn!("publish_event({DIAGNOSED_EVENT_NAME}) failed: {e:#}");
    }
}

async fn publish_circuit_open(
    publisher: Option<&Arc<Publisher>>,
    event: &IncidentEvent,
    correlation_id: &str,
    reset_in_seconds: u64,
) {
    let Some(p) = publisher else {
        return;
    };
    let payload = serde_json::json!({
        "service": event.payload.component,
        "correlation_id": correlation_id,
        "reset_in_seconds": reset_in_seconds,
        "original_summary": event.payload.summary,
        "original_timestamp": event.timestamp.to_rfc3339(),
    });
    if let Err(e) = p.publish_event(CIRCUIT_OPEN_EVENT_NAME, payload).await {
        tracing::warn!("publish_event({CIRCUIT_OPEN_EVENT_NAME}) failed: {e:#}");
    }
}

#[cfg(test)]
mod tests;
