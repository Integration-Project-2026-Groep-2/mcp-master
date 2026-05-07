use anyhow::{Context, Result};

#[derive(Clone, Debug)]
pub struct RabbitMqConfig {
    pub url: String,
    pub exchange: String,
}

impl RabbitMqConfig {
    pub fn from_env() -> Result<Self> {
        Ok(Self {
            url: std::env::var("RABBITMQ_URL").context("RABBITMQ_URL not set")?,
            exchange: std::env::var("RABBITMQ_EXCHANGE").unwrap_or_else(|_| "ai.events".into()),
        })
    }

    /// Redacted form of `url` safe for stdout logs — strips userinfo
    /// (`user:password@`) so the broker password never leaks via tracing.
    /// Returns `"<invalid URL>"` if parsing fails; lapin will surface the
    /// real error on connect anyway.
    pub fn host_for_logging(&self) -> String {
        match url::Url::parse(&self.url) {
            Ok(u) => {
                let host = u.host_str().unwrap_or("?");
                let port = u.port().map(|p| format!(":{p}")).unwrap_or_default();
                format!("{}://{}{}{}", u.scheme(), host, port, u.path())
            }
            Err(_) => "<invalid URL>".to_string(),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn host_for_logging_strips_userinfo() {
        let cfg = RabbitMqConfig {
            url: "amqp://lapin:supersecret@rabbitmq_management:5672//".to_string(),
            exchange: "ai.events".to_string(),
        };
        let s = cfg.host_for_logging();
        assert!(!s.contains("supersecret"), "must not leak password: {s}");
        assert!(!s.contains("lapin"), "must not leak username: {s}");
        assert!(
            s.contains("rabbitmq_management:5672"),
            "must keep host:port: {s}"
        );
        assert!(s.starts_with("amqp://"), "must keep scheme: {s}");
    }

    #[test]
    fn host_for_logging_handles_no_port() {
        let cfg = RabbitMqConfig {
            url: "amqp://guest:guest@localhost//".to_string(),
            exchange: "x".to_string(),
        };
        let s = cfg.host_for_logging();
        assert!(!s.contains("guest"), "must not leak guest credentials: {s}");
        assert!(s.contains("localhost"));
    }

    #[test]
    fn host_for_logging_invalid_url_returns_placeholder() {
        let cfg = RabbitMqConfig {
            url: "not a url".to_string(),
            exchange: "x".to_string(),
        };
        assert_eq!(cfg.host_for_logging(), "<invalid URL>");
    }
}
