    use super::*;

    #[test]
    fn is_likely_disconnect_catches_io_errors() {
        let err = anyhow::anyhow!("AMQP basic_publish: IO error: connection reset");
        assert!(is_likely_disconnect(&err));
    }

    #[test]
    fn is_likely_disconnect_catches_channel_closed() {
        let err = anyhow::anyhow!("Channel closed");
        assert!(is_likely_disconnect(&err));
    }

    #[test]
    fn is_likely_disconnect_catches_broken_pipe() {
        let err = anyhow::anyhow!("write error: Broken pipe");
        assert!(is_likely_disconnect(&err));
    }

    #[test]
    fn is_likely_disconnect_rejects_deterministic_errors() {
        // These are NOT disconnects — they're config / auth bugs that
        // a reconnect won't fix. Retrying would just waste effort.
        let exchange_not_found = anyhow::anyhow!("AMQP exchange not found");
        assert!(!is_likely_disconnect(&exchange_not_found));

        let access_refused = anyhow::anyhow!("ACCESS_REFUSED login refused");
        assert!(!is_likely_disconnect(&access_refused));

        let not_allowed = anyhow::anyhow!("NOT_ALLOWED vhost / not found");
        assert!(!is_likely_disconnect(&not_allowed));
    }

    #[test]
    fn is_likely_disconnect_is_case_insensitive() {
        let upper = anyhow::anyhow!("CONNECTION RESET by peer");
        assert!(is_likely_disconnect(&upper));
    }
