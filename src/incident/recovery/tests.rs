    use super::*;
    use std::time::{Duration, Instant};

    const RECOVERY_BODY: &str = r#"{
        "event": "heartbeat_online",
        "source": "controlroom-watchdog",
        "timestamp": "2026-05-12T16:30:00Z",
        "payload": {
            "summary": "KASSA heartbeat is back online%!(EXTRA float64=47)",
            "severity": "critical",
            "component": "kassa",
            "group": "festival-services",
            "class": "heartbeat-loss",
            "custom_details": {
                "heartbeat_count_last_60s": 47,
                "threshold": 30,
                "last_check_at": "2026-05-12T16:30:00Z"
            }
        }
    }"#;

    #[test]
    fn parses_heartbeat_online_body() {
        let evt: IncidentEvent = serde_json::from_str(RECOVERY_BODY).expect("parses");
        assert_eq!(evt.event, "heartbeat_online");
        assert_eq!(evt.payload.component, "kassa");
        assert!(evt.payload.summary.contains("%!(EXTRA"));
    }

    #[test]
    fn scrub_summary_strips_extra_tail() {
        let scrubbed = scrub_summary("KASSA heartbeat is back online%!(EXTRA float64=47)");
        assert_eq!(scrubbed, "KASSA heartbeat is back online");
    }

    #[test]
    fn scrub_summary_no_marker_unchanged() {
        let clean = "CRM heartbeat is back online";
        assert_eq!(scrub_summary(clean), clean);
    }

    #[test]
    fn scrub_summary_handles_empty_and_marker_only() {
        assert_eq!(scrub_summary(""), "");
        assert_eq!(scrub_summary("%!(EXTRA int=0)"), "");
    }

    #[tokio::test]
    async fn handle_recovery_accepts_valid_body_without_publisher() {
        let r = handle_recovery(RECOVERY_BODY.as_bytes(), None).await;
        assert!(r.is_ok(), "skip-warn path must not fail: {r:?}");
    }

    #[tokio::test]
    async fn handle_recovery_rejects_malformed_json() {
        let r = handle_recovery(b"not json", None).await;
        assert!(r.is_err(), "malformed JSON must surface as Err for DLQ");
    }

    #[tokio::test]
    async fn handle_recovery_rejects_unknown_severity() {
        let bad = RECOVERY_BODY.replace("\"critical\"", "\"meltdown\"");
        let r = handle_recovery(bad.as_bytes(), None).await;
        assert!(r.is_err());
    }

    #[tokio::test]
    async fn run_returns_immediately_on_pre_set_shutdown() {
        let (tx, rx) = watch::channel(false);
        tx.send(true).unwrap();

        let cfg = RabbitMqConfig {
            url: "amqp://nobody:nobody@127.0.0.1:1//".to_string(),
            exchange: "test.events".to_string(),
        };

        let started = Instant::now();
        let result = run(cfg, None, rx).await;
        let elapsed = started.elapsed();

        assert!(result.is_ok(), "shutdown-first should succeed: {result:?}");
        assert!(
            elapsed.as_millis() < 100,
            "should return fast (<100ms), took {}ms",
            elapsed.as_millis()
        );
    }

    #[tokio::test]
    async fn run_honors_shutdown_during_backoff() {
        let (tx, rx) = watch::channel(false);

        let cfg = RabbitMqConfig {
            url: "amqp://nobody:nobody@127.0.0.1:1//".to_string(),
            exchange: "test.events".to_string(),
        };

        let handle = tokio::spawn(run(cfg, None, rx));

        tokio::time::sleep(Duration::from_millis(150)).await;
        tx.send(true).unwrap();

        let started = Instant::now();
        let result = tokio::time::timeout(Duration::from_secs(2), handle)
            .await
            .expect("run should not hang past 2s after shutdown signal");

        assert!(started.elapsed().as_millis() < 2000);
        let inner = result.expect("join should succeed");
        assert!(inner.is_ok(), "run should return Ok on shutdown: {inner:?}");
    }
