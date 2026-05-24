use super::*;
use chrono::TimeZone;
use serial_test::serial;

#[test]
fn xml_matches_facturatie_heartbeat_contract() {
    let ts = Utc.with_ymd_and_hms(2026, 5, 24, 14, 30, 0).unwrap();
    assert_eq!(
        build_heartbeat_xml("mcp-master", &ts),
        "<?xml version=\"1.0\" encoding=\"UTF-8\"?>\n\
         <Heartbeat><serviceId>mcp-master</serviceId><timestamp>2026-05-24T14:30:00+00:00</timestamp></Heartbeat>\n"
    );
}

#[test]
fn xml_timestamp_is_second_precision_with_numeric_offset() {
    let ts = Utc.with_ymd_and_hms(2026, 1, 2, 3, 4, 5).unwrap();
    let xml = build_heartbeat_xml("svc", &ts);
    assert!(xml.contains("<timestamp>2026-01-02T03:04:05+00:00</timestamp>"));
}

#[test]
#[serial]
fn config_defaults_when_env_unset() {
    unsafe {
        std::env::remove_var("HEARTBEAT_EXCHANGE");
        std::env::remove_var("HEARTBEAT_ROUTING_KEY");
        std::env::remove_var("HEARTBEAT_SERVICE_ID");
        std::env::remove_var("SERVICE_ID");
        std::env::remove_var("HEARTBEAT_INTERVAL_MS");
    }
    let cfg = HeartbeatConfig::from_env();
    assert_eq!(cfg.exchange, "heartbeat.direct");
    assert_eq!(cfg.routing_key, "routing.heartbeat");
    assert_eq!(cfg.service_id, "mcp-master");
    assert_eq!(cfg.interval, Duration::from_millis(1000));
}

#[test]
#[serial]
fn interval_is_clamped_to_minimum() {
    unsafe {
        std::env::set_var("HEARTBEAT_INTERVAL_MS", "50");
    }
    let cfg = HeartbeatConfig::from_env();
    assert_eq!(cfg.interval, Duration::from_millis(100));
    unsafe {
        std::env::remove_var("HEARTBEAT_INTERVAL_MS");
    }
}

#[test]
#[serial]
fn service_id_falls_back_to_generic_service_id_var() {
    unsafe {
        std::env::remove_var("HEARTBEAT_SERVICE_ID");
        std::env::set_var("SERVICE_ID", "mcp-master-canary");
    }
    let cfg = HeartbeatConfig::from_env();
    assert_eq!(cfg.service_id, "mcp-master-canary");
    unsafe {
        std::env::remove_var("SERVICE_ID");
    }
}

#[test]
fn xml_escapes_special_chars_in_service_id() {
    let ts = Utc.with_ymd_and_hms(2026, 5, 24, 14, 30, 0).unwrap();
    let xml = build_heartbeat_xml("a&b<c", &ts);
    assert!(xml.contains("<serviceId>a&amp;b&lt;c</serviceId>"));
}

#[test]
#[serial]
fn interval_is_clamped_to_maximum() {
    unsafe {
        std::env::set_var("HEARTBEAT_INTERVAL_MS", "60000");
    }
    let cfg = HeartbeatConfig::from_env();
    assert_eq!(cfg.interval, Duration::from_millis(1500));
    unsafe {
        std::env::remove_var("HEARTBEAT_INTERVAL_MS");
    }
}

#[tokio::test]
async fn run_exits_on_shutdown_when_broker_unreachable() {
    let (tx, rx) = tokio::sync::watch::channel(false);
    let cfg = HeartbeatConfig {
        exchange: "heartbeat.direct".to_string(),
        routing_key: "routing.heartbeat".to_string(),
        service_id: "mcp-master".to_string(),
        interval: Duration::from_millis(100),
    };
    // Unroutable broker → run sits in the connect-failure backoff loop; the
    // shutdown signal must break it promptly rather than spin or hang.
    let handle = tokio::spawn(run("amqp://127.0.0.1:1/%2f".to_string(), cfg, rx));
    tokio::time::sleep(Duration::from_millis(100)).await;
    tx.send(true).unwrap();
    let joined = tokio::time::timeout(Duration::from_secs(5), handle).await;
    assert!(matches!(joined, Ok(Ok(Ok(())))));
}

// End-to-end against a real broker: drives the production `run` task and
// asserts a heartbeat lands on the wire with the full contract. Run with
// `--ignored`; broker URL via HEARTBEAT_E2E_URL (default localhost:5673).
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
#[ignore = "requires a live RabbitMQ broker"]
async fn e2e_publishes_heartbeat_to_broker() {
    use futures_util::StreamExt;
    use lapin::ExchangeKind;
    use lapin::options::{
        BasicConsumeOptions, ExchangeDeclareOptions, QueueBindOptions, QueueDeclareOptions,
    };
    use lapin::types::FieldTable;

    let url = std::env::var("HEARTBEAT_E2E_URL")
        .unwrap_or_else(|_| "amqp://guest:guest@127.0.0.1:5673/%2f".to_string());

    let conn = crate::rabbitmq::connect_with_timeout(&url)
        .await
        .expect("connect consumer");
    let ch = conn.create_channel().await.expect("channel");
    ch.exchange_declare(
        "heartbeat.direct",
        ExchangeKind::Direct,
        ExchangeDeclareOptions {
            durable: true,
            ..Default::default()
        },
        FieldTable::default(),
    )
    .await
    .expect("exchange_declare");
    let q = ch
        .queue_declare(
            "",
            QueueDeclareOptions {
                exclusive: true,
                auto_delete: true,
                ..Default::default()
            },
            FieldTable::default(),
        )
        .await
        .expect("queue_declare");
    ch.queue_bind(
        q.name().as_str(),
        "heartbeat.direct",
        "routing.heartbeat",
        QueueBindOptions::default(),
        FieldTable::default(),
    )
    .await
    .expect("queue_bind");
    let mut consumer = ch
        .basic_consume(
            q.name().as_str(),
            "hb-e2e",
            BasicConsumeOptions {
                no_ack: true,
                ..Default::default()
            },
            FieldTable::default(),
        )
        .await
        .expect("basic_consume");

    let (tx, rx) = tokio::sync::watch::channel(false);
    let cfg = HeartbeatConfig {
        exchange: "heartbeat.direct".to_string(),
        routing_key: "routing.heartbeat".to_string(),
        service_id: "mcp-master-e2e".to_string(),
        interval: Duration::from_millis(200),
    };
    let handle = tokio::spawn(run(url.clone(), cfg, rx));

    let delivery = tokio::time::timeout(Duration::from_secs(10), consumer.next())
        .await
        .expect("no heartbeat within 10s")
        .expect("consumer stream ended")
        .expect("delivery error");

    let body = String::from_utf8_lossy(&delivery.data);
    assert!(body.contains("<Heartbeat>"), "body: {body}");
    assert!(
        body.contains("<serviceId>mcp-master-e2e</serviceId>"),
        "body: {body}"
    );
    assert!(body.contains("<timestamp>"), "body: {body}");

    let content_type = delivery
        .properties
        .content_type()
        .as_ref()
        .map(|s| s.as_str().to_owned());
    assert_eq!(content_type.as_deref(), Some("application/xml"));
    assert_eq!(delivery.properties.delivery_mode(), &Some(2));

    tx.send(true).unwrap();
    let joined = tokio::time::timeout(Duration::from_secs(5), handle).await;
    assert!(matches!(joined, Ok(Ok(Ok(())))));
}
