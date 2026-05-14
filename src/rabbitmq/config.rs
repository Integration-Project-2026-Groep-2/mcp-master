use anyhow::{Context, Result};

#[derive(Clone, Debug)]
pub struct RabbitMqConfig {
    pub url: String,
    pub exchange: String,
}

impl RabbitMqConfig {
    pub fn from_env() -> Result<Self> {
        let raw = std::env::var("RABBITMQ_URL").context("RABBITMQ_URL not set")?;
        Ok(Self {
            url: normalize_amqp_url(&raw),
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

/// Normalize trailing single `/` to spec-conforming `/%2F` so strict-spec
/// AMQP libraries (e.g. lapin via amq-protocol) connect to the default
/// vhost `/` instead of failing with empty-vhost lookup. Matches the
/// permissive behavior of pika/Python — but explicit at parse time so
/// lapin's `url.path().get(1..)` yields `"%2F"` → vhost `/` after decode.
///
/// Per AMQP-URI spec (rabbitmq.com/docs/uri-spec):
/// - `amqp://h/`   → vhost `""` (empty)         — broken for default vhost
/// - `amqp://h//`  → vhost `/` (multi-segment)  — non-conforming but works
/// - `amqp://h/%2F`→ vhost `/`                  — spec-conforming
/// - `amqp://h`    → vhost absent → lapin defaults to `/`
fn normalize_amqp_url(raw: &str) -> String {
    let trimmed = raw.trim();

    if trimmed.ends_with("//") || trimmed.to_ascii_lowercase().ends_with("/%2f") {
        return trimmed.to_string();
    }

    let Some(scheme_end) = trimmed.find("://").map(|i| i + 3) else {
        return trimmed.to_string();
    };
    let Some(rel_path_start) = trimmed[scheme_end..].find('/') else {
        return trimmed.to_string();
    };
    let path_start = scheme_end + rel_path_start;
    let path = &trimmed[path_start..];

    if path == "/" {
        tracing::warn!(
            "RABBITMQ_URL ends in single '/' (empty vhost per AMQP-URI spec); \
             auto-normalizing to '/%2F' for default vhost. \
             Fix at source: use '//' or '/%2F'."
        );
        return format!("{trimmed}%2F");
    }

    trimmed.to_string()
}

#[cfg(test)]
mod tests;
