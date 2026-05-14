    use super::*;
    use std::time::{Duration, Instant};
    use tokio::sync::watch;

    #[tokio::test]
    async fn run_returns_immediately_on_pre_set_shutdown() {
        let (tx, rx) = watch::channel(false);
        tx.send(true).unwrap();

        let cfg = RabbitMqConfig {
            url: "amqp://nobody:nobody@127.0.0.1:1//".to_string(),
            exchange: "test.events".to_string(),
        };

        let started = Instant::now();
        let result = run(cfg, None, None, rx).await;
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

        let handle = tokio::spawn(run(cfg, None, None, rx));

        tokio::time::sleep(Duration::from_millis(150)).await;
        tx.send(true).unwrap();

        let started = Instant::now();
        let result = tokio::time::timeout(Duration::from_secs(2), handle)
            .await
            .expect("run should not hang past 2s after shutdown signal");

        assert!(
            started.elapsed().as_millis() < 2000,
            "shutdown should wake the backoff sleep early"
        );
        assert!(
            result.is_ok(),
            "join should succeed: {:?}",
            result.unwrap_err()
        );
        let inner = result.unwrap();
        assert!(inner.is_ok(), "run should return Ok on shutdown: {inner:?}");
    }

    fn body(severity: &str, component: &str) -> Vec<u8> {
        format!(
            r#"{{
                "event": "heartbeat_failed",
                "source": "controlroom-watchdog",
                "timestamp": "2026-05-10T14:23:17Z",
                "payload": {{
                    "summary": "{component} down",
                    "severity": "{severity}",
                    "component": "{component}"
                }}
            }}"#
        )
        .into_bytes()
    }

    #[tokio::test]
    async fn handle_delivery_accepts_critical_first_time() {
        let d = Debouncer::new(Duration::from_secs(60));
        let b = Budget::new(u32::MAX);
        let r = handle_delivery(&body("critical", "kassa"), &d, &b, None, None).await;
        assert!(r.is_ok());
    }

    #[tokio::test]
    async fn handle_delivery_skips_warning_severity() {
        let d = Debouncer::new(Duration::from_secs(60));
        let b = Budget::new(u32::MAX);
        let r = handle_delivery(&body("warning", "kassa"), &d, &b, None, None).await;
        assert!(r.is_ok(), "skip is not an error path");
    }

    #[tokio::test]
    async fn handle_delivery_skips_info_severity() {
        let d = Debouncer::new(Duration::from_secs(60));
        let b = Budget::new(u32::MAX);
        let r = handle_delivery(&body("info", "kassa"), &d, &b, None, None).await;
        assert!(r.is_ok());
    }

    #[tokio::test]
    async fn handle_delivery_debounces_rapid_repeats() {
        let d = Debouncer::new(Duration::from_secs(60));
        let b = Budget::new(u32::MAX);
        let first = handle_delivery(&body("critical", "kassa"), &d, &b, None, None).await;
        let second = handle_delivery(&body("critical", "kassa"), &d, &b, None, None).await;
        assert!(first.is_ok());
        assert!(second.is_ok(), "second is skipped, not failed");
        assert!(d.check("kassa").is_err(), "first allow consumed the slot");
    }

    #[tokio::test]
    async fn handle_delivery_severity_skip_does_not_consume_debounce_slot() {
        let d = Debouncer::new(Duration::from_secs(60));
        let b = Budget::new(u32::MAX);
        let warn_result = handle_delivery(&body("warning", "kassa"), &d, &b, None, None).await;
        assert!(warn_result.is_ok());
        // Critical should now be allowed — warning didn't burn the slot.
        let crit_result = handle_delivery(&body("critical", "kassa"), &d, &b, None, None).await;
        assert!(crit_result.is_ok());
        assert!(d.check("kassa").is_err());
    }

    #[tokio::test]
    async fn handle_delivery_rejects_malformed_json() {
        let d = Debouncer::new(Duration::from_secs(60));
        let b = Budget::new(u32::MAX);
        let r = handle_delivery(b"not json", &d, &b, None, None).await;
        assert!(r.is_err());
    }

    #[tokio::test]
    async fn handle_delivery_rejects_unknown_severity() {
        let d = Debouncer::new(Duration::from_secs(60));
        let b = Budget::new(u32::MAX);
        let body = br#"{
            "event": "heartbeat_failed",
            "source": "controlroom-watchdog",
            "timestamp": "2026-05-10T14:23:17Z",
            "payload": {
                "summary": "x",
                "severity": "meltdown",
                "component": "x"
            }
        }"#;
        let r = handle_delivery(body, &d, &b, None, None).await;
        assert!(r.is_err());
    }

    use crate::incident::schema::{Confidence, IncidentDiagnosis};
    use async_trait::async_trait;
    use std::collections::VecDeque;
    use std::sync::atomic::{AtomicUsize, Ordering};
    use tokio::sync::Mutex;

    struct MockPipeline {
        queued: Mutex<VecDeque<Result<IncidentDiagnosis, String>>>,
        call_count: AtomicUsize,
    }

    impl MockPipeline {
        fn new() -> Self {
            Self {
                queued: Mutex::new(VecDeque::new()),
                call_count: AtomicUsize::new(0),
            }
        }

        async fn with_ok(self, d: IncidentDiagnosis) -> Self {
            self.queued.lock().await.push_back(Ok(d));
            self
        }

        async fn with_err(self, msg: &str) -> Self {
            self.queued.lock().await.push_back(Err(msg.into()));
            self
        }

        fn call_count(&self) -> usize {
            self.call_count.load(Ordering::SeqCst)
        }
    }

    #[async_trait]
    impl DiagnosePipeline for MockPipeline {
        async fn diagnose(&self, _event: &IncidentEvent) -> Result<IncidentDiagnosis> {
            self.call_count.fetch_add(1, Ordering::SeqCst);
            match self.queued.lock().await.pop_front() {
                Some(Ok(d)) => Ok(d),
                Some(Err(e)) => Err(anyhow::anyhow!("{e}")),
                None => Err(anyhow::anyhow!("MockPipeline: no canned response")),
            }
        }
    }

    fn diag(confidence: Confidence) -> IncidentDiagnosis {
        IncidentDiagnosis {
            root_cause: "test cause".into(),
            critical_failure: "test failure".into(),
            impact: "test impact".into(),
            confidence,
            suggested_action: None,
            evidence_summary: "test evidence".into(),
        }
    }

    #[tokio::test]
    async fn handle_delivery_calls_pipeline_when_configured_and_accepted() {
        let d = Debouncer::new(Duration::from_secs(60));
        let b = Budget::new(u32::MAX);
        let mock = Arc::new(MockPipeline::new().with_ok(diag(Confidence::High)).await);
        let pipeline: Arc<dyn DiagnosePipeline> = mock.clone();
        let r = handle_delivery(&body("critical", "kassa"), &d, &b, None, Some(&pipeline)).await;
        assert!(r.is_ok());
        assert_eq!(mock.call_count(), 1);
    }

    #[tokio::test]
    async fn handle_delivery_skips_pipeline_when_severity_filtered() {
        let d = Debouncer::new(Duration::from_secs(60));
        let b = Budget::new(u32::MAX);
        let mock = Arc::new(MockPipeline::new());
        let pipeline: Arc<dyn DiagnosePipeline> = mock.clone();
        let r = handle_delivery(&body("warning", "kassa"), &d, &b, None, Some(&pipeline)).await;
        assert!(r.is_ok());
        assert_eq!(mock.call_count(), 0);
    }

    #[tokio::test]
    async fn handle_delivery_skips_pipeline_when_debounced() {
        let d = Debouncer::new(Duration::from_secs(60));
        let b = Budget::new(u32::MAX);
        let mock = Arc::new(MockPipeline::new().with_ok(diag(Confidence::High)).await);
        let pipeline: Arc<dyn DiagnosePipeline> = mock.clone();

        let first =
            handle_delivery(&body("critical", "kassa"), &d, &b, None, Some(&pipeline)).await;
        let second =
            handle_delivery(&body("critical", "kassa"), &d, &b, None, Some(&pipeline)).await;
        assert!(first.is_ok());
        assert!(second.is_ok());
        assert_eq!(mock.call_count(), 1, "second event was debounced");
    }

    #[tokio::test]
    async fn handle_delivery_swallows_pipeline_errors_so_message_is_acked() {
        let d = Debouncer::new(Duration::from_secs(60));
        let b = Budget::new(u32::MAX);
        let mock = Arc::new(MockPipeline::new().with_err("LLM timeout").await);
        let pipeline: Arc<dyn DiagnosePipeline> = mock.clone();
        let r = handle_delivery(&body("critical", "kassa"), &d, &b, None, Some(&pipeline)).await;
        assert!(
            r.is_ok(),
            "pipeline failure must not propagate (would trigger DLQ)"
        );
        assert_eq!(mock.call_count(), 1);
    }

    #[tokio::test]
    async fn handle_delivery_accepts_when_no_pipeline_configured() {
        let d = Debouncer::new(Duration::from_secs(60));
        let b = Budget::new(u32::MAX);
        let r = handle_delivery(&body("critical", "kassa"), &d, &b, None, None).await;
        assert!(r.is_ok());
    }

    #[tokio::test]
    async fn handle_delivery_skips_pipeline_when_budget_exhausted() {
        let d = Debouncer::new(Duration::from_secs(60));
        let b = Budget::new(0);
        let mock = Arc::new(MockPipeline::new());
        let pipeline: Arc<dyn DiagnosePipeline> = mock.clone();
        let r = handle_delivery(&body("critical", "kassa"), &d, &b, None, Some(&pipeline)).await;
        assert!(r.is_ok());
        assert_eq!(
            mock.call_count(),
            0,
            "pipeline must not run when circuit is open"
        );
    }

    #[tokio::test]
    async fn handle_delivery_budget_caps_total_pipeline_calls() {
        let d = Debouncer::new(Duration::from_secs(60));
        let b = Budget::new(2);
        let mock = Arc::new(
            MockPipeline::new()
                .with_ok(diag(Confidence::High))
                .await
                .with_ok(diag(Confidence::Medium))
                .await
                .with_ok(diag(Confidence::Low))
                .await,
        );
        let pipeline: Arc<dyn DiagnosePipeline> = mock.clone();

        let r1 = handle_delivery(&body("critical", "kassa"), &d, &b, None, Some(&pipeline)).await;
        let r2 = handle_delivery(&body("critical", "crm"), &d, &b, None, Some(&pipeline)).await;
        let r3 = handle_delivery(
            &body("critical", "controlroom"),
            &d,
            &b,
            None,
            Some(&pipeline),
        )
        .await;

        assert!(r1.is_ok());
        assert!(r2.is_ok());
        assert!(r3.is_ok());
        assert_eq!(mock.call_count(), 2, "third event must hit circuit-open");
    }

    #[tokio::test]
    async fn handle_delivery_severity_skip_does_not_consume_budget() {
        let d = Debouncer::new(Duration::from_secs(60));
        let b = Budget::new(1);
        let mock = Arc::new(MockPipeline::new().with_ok(diag(Confidence::High)).await);
        let pipeline: Arc<dyn DiagnosePipeline> = mock.clone();

        let _ = handle_delivery(&body("warning", "kassa"), &d, &b, None, Some(&pipeline)).await;
        let r = handle_delivery(&body("critical", "kassa"), &d, &b, None, Some(&pipeline)).await;

        assert!(r.is_ok());
        assert_eq!(
            mock.call_count(),
            1,
            "warning skip didn't burn the budget slot"
        );
    }
