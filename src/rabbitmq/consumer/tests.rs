    use super::*;
    use std::time::Instant;
    use tokio::sync::watch;

    /// Fast-path: when shutdown is set BEFORE `run` is called, return Ok
    /// immediately without trying to connect (which would block forever
    /// against an unreachable broker).
    #[tokio::test]
    async fn run_returns_immediately_on_pre_set_shutdown() {
        let (tx, rx) = watch::channel(false);
        tx.send(true).unwrap();

        let cfg = RabbitMqConfig {
            // Definitely-unreachable address. If we ever try to connect,
            // it would hang or fail — but the pre-set shutdown should
            // short-circuit before we get there.
            url: "amqp://nobody:nobody@127.0.0.1:1//".to_string(),
            exchange: "test.events".to_string(),
        };

        let started = Instant::now();
        let result = run(cfg, rx).await;
        let elapsed = started.elapsed();

        assert!(result.is_ok(), "shutdown-first should succeed: {result:?}");
        assert!(
            elapsed.as_millis() < 100,
            "should return fast (<100ms), took {}ms",
            elapsed.as_millis()
        );
    }

    /// On a connect failure to an unreachable broker, the outer loop must
    /// honor a shutdown signal during the backoff sleep. Validates that we
    /// don't get stuck in a multi-second sleep blocking graceful shutdown.
    #[tokio::test]
    async fn run_honors_shutdown_during_backoff() {
        let (tx, rx) = watch::channel(false);

        let cfg = RabbitMqConfig {
            // Port 1 is reserved/unbindable on most systems → connect fails
            // fast. The test relies on connect failing within ~ms, then
            // entering the backoff sleep where we send the shutdown signal.
            url: "amqp://nobody:nobody@127.0.0.1:1//".to_string(),
            exchange: "test.events".to_string(),
        };

        let handle = tokio::spawn(run(cfg, rx));

        // Give the run a moment to fail-connect once and enter backoff.
        tokio::time::sleep(std::time::Duration::from_millis(150)).await;
        tx.send(true).unwrap();

        let started = Instant::now();
        let result = tokio::time::timeout(std::time::Duration::from_secs(2), handle)
            .await
            .expect("run should not hang past 2s after shutdown signal");

        // Total time under 2s — proves we didn't sleep through the full
        // ~1s backoff before noticing shutdown.
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
