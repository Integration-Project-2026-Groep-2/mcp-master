//! RabbitMQ management HTTP API client for the `/architecture` endpoint.
//!
//! Talks to the management plugin on port 15672 to enumerate exchanges,
//! queues, and bindings — completely separate protocol from the `lapin`
//! AMQP client we use for pub/sub on port 5672. Credentials are reused
//! from `RABBITMQ_URL` since the management plugin shares the broker's
//! internal user list.

use std::time::Duration;

use anyhow::{Context, Result};
use serde::Deserialize;
use url::Url;

const DEFAULT_MGMT_PORT: u16 = 15672;
const REQUEST_TIMEOUT_SECONDS: u64 = 10;

/// One exchange as reported by `GET /api/exchanges/<vhost>`.
/// RabbitMQ returns many fields; we only deserialize what we need.
#[derive(Debug, Clone, Deserialize, PartialEq)]
pub struct ExchangeInfo {
    pub name: String,
    /// `topic` | `direct` | `fanout` | `headers`. RabbitMQ never includes
    /// a discriminator, just the kind-name string.
    #[serde(rename = "type")]
    pub kind: String,
}

/// One queue as reported by `GET /api/queues/<vhost>`.
#[derive(Debug, Clone, Deserialize, PartialEq)]
pub struct QueueInfo {
    pub name: String,
}

/// One binding as reported by `GET /api/bindings/<vhost>`. RabbitMQ uses
/// `source` for the exchange, `destination` for queue-or-exchange name,
/// `destination_type` to distinguish them.
#[derive(Debug, Clone, Deserialize, PartialEq)]
pub struct BindingInfo {
    pub source: String,
    pub destination: String,
    pub destination_type: String,
    pub routing_key: String,
}

/// Aggregated topology snapshot — what the builder needs to draw the
/// RabbitMQ-side of the graph.
#[derive(Debug, Clone, PartialEq)]
pub struct RabbitMqTopology {
    pub exchanges: Vec<ExchangeInfo>,
    pub queues: Vec<QueueInfo>,
    pub bindings: Vec<BindingInfo>,
}

/// HTTP client for the RabbitMQ management plugin.
pub struct ManagementClient {
    http: reqwest::Client,
    base_url: String,
    user: String,
    pass: String,
    /// URL-encoded vhost (default vhost `/` is `%2F`).
    vhost_encoded: String,
}

impl ManagementClient {
    /// Test-only constructor: build a client pointing at an arbitrary
    /// `base_url` (wiremock, localhost dev broker, etc.) without going
    /// through env-var parsing.
    #[cfg(test)]
    pub fn new_for_test(base_url: impl Into<String>) -> Self {
        Self {
            http: reqwest::Client::builder()
                .timeout(Duration::from_secs(REQUEST_TIMEOUT_SECONDS))
                .build()
                .expect("test reqwest client builds"),
            base_url: base_url.into(),
            user: "guest".to_string(),
            pass: "guest".to_string(),
            vhost_encoded: "%2F".to_string(),
        }
    }

    /// Construct from env. Required: `RABBITMQ_URL` (same var the AMQP
    /// publisher reads). Optional: `RABBITMQ_MGMT_URL` to override the
    /// derived base URL (e.g. when the management plugin lives on a
    /// different host than the AMQP listener).
    pub fn from_env() -> Result<Self> {
        let amqp_url =
            std::env::var("RABBITMQ_URL").context("RABBITMQ_URL not set for management client")?;
        let parsed = Url::parse(&amqp_url)
            .context("RABBITMQ_URL is not a valid URL for management client")?;

        let user = parsed.username().to_string();
        let pass = parsed.password().unwrap_or("").to_string();
        let vhost_raw = vhost_from_amqp_url(&parsed);
        let vhost_encoded = urlencoding_encode(&vhost_raw);

        let base_url = match std::env::var("RABBITMQ_MGMT_URL") {
            Ok(v) if !v.trim().is_empty() => v.trim_end_matches('/').to_string(),
            _ => derive_mgmt_url(&parsed)?,
        };

        let http = reqwest::Client::builder()
            .timeout(Duration::from_secs(REQUEST_TIMEOUT_SECONDS))
            .build()
            .context("building reqwest client for RabbitMQ management API")?;

        Ok(Self {
            http,
            base_url,
            user,
            pass,
            vhost_encoded,
        })
    }

    /// Fetch the three resource-collections in parallel — they're
    /// independent on the broker side.
    pub async fn fetch_topology(&self) -> Result<RabbitMqTopology> {
        let (exchanges, queues, bindings) = tokio::try_join!(
            self.fetch_exchanges(),
            self.fetch_queues(),
            self.fetch_bindings(),
        )?;
        Ok(RabbitMqTopology {
            exchanges,
            queues,
            bindings,
        })
    }

    async fn fetch_exchanges(&self) -> Result<Vec<ExchangeInfo>> {
        let path = format!("/api/exchanges/{}", self.vhost_encoded);
        let body = self.get(&path).await?;
        serde_json::from_str(&body).context("decoding /api/exchanges response")
    }

    async fn fetch_queues(&self) -> Result<Vec<QueueInfo>> {
        let path = format!("/api/queues/{}", self.vhost_encoded);
        let body = self.get(&path).await?;
        serde_json::from_str(&body).context("decoding /api/queues response")
    }

    async fn fetch_bindings(&self) -> Result<Vec<BindingInfo>> {
        let path = format!("/api/bindings/{}", self.vhost_encoded);
        let body = self.get(&path).await?;
        serde_json::from_str(&body).context("decoding /api/bindings response")
    }

    async fn get(&self, path: &str) -> Result<String> {
        let url = format!("{}{}", self.base_url, path);
        let resp = self
            .http
            .get(&url)
            .basic_auth(&self.user, Some(&self.pass))
            .send()
            .await
            .with_context(|| format!("GET {url}"))?;
        let status = resp.status();
        let text = resp.text().await.context("reading response body")?;
        if !status.is_success() {
            anyhow::bail!("RabbitMQ management API returned {status}: {text}");
        }
        Ok(text)
    }
}

fn derive_mgmt_url(amqp: &Url) -> Result<String> {
    let host = amqp
        .host_str()
        .context("RABBITMQ_URL has no host — cannot derive management URL")?;
    Ok(format!("http://{host}:{DEFAULT_MGMT_PORT}"))
}

fn vhost_from_amqp_url(amqp: &Url) -> String {
    // AMQP URI spec: path component is the vhost name. `/` → vhost ""
    // (empty); `//` (or `/%2F`) → vhost `/`. Normalize for the mgmt API
    // which always wants the actual vhost name. url::Url::path() returns
    // the raw (still percent-encoded) path, so we decode after trimming.
    let path = amqp.path();
    if path.is_empty() || path == "/" {
        return "/".to_string();
    }
    percent_decode(path.trim_start_matches('/'))
}

/// Minimal `%XX` → byte decoder. Vhosts are short ASCII strings so we
/// avoid pulling in `percent_encoding` as a direct dep.
fn percent_decode(s: &str) -> String {
    let bytes = s.as_bytes();
    let mut out = Vec::with_capacity(bytes.len());
    let mut i = 0;
    while i < bytes.len() {
        if bytes[i] == b'%' && i + 2 < bytes.len() {
            let hi = (bytes[i + 1] as char).to_digit(16);
            let lo = (bytes[i + 2] as char).to_digit(16);
            if let (Some(h), Some(l)) = (hi, lo) {
                out.push(((h << 4) | l) as u8);
                i += 3;
                continue;
            }
        }
        out.push(bytes[i]);
        i += 1;
    }
    String::from_utf8_lossy(&out).into_owned()
}

/// Tiny URL-encoder — RabbitMQ vhosts are short strings, so we don't
/// pull in a full crate. Encodes `/` (the only path-meaningful char a
/// vhost realistically contains) plus the basics.
fn urlencoding_encode(s: &str) -> String {
    let mut out = String::with_capacity(s.len() * 3);
    for ch in s.chars() {
        match ch {
            'A'..='Z' | 'a'..='z' | '0'..='9' | '-' | '_' | '.' | '~' => out.push(ch),
            _ => {
                for byte in ch.to_string().as_bytes() {
                    out.push_str(&format!("%{byte:02X}"));
                }
            }
        }
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn derive_mgmt_url_swaps_to_http_15672() {
        let amqp = Url::parse("amqp://lapin:secret@rabbitmq_management:5672/%2F").unwrap();
        let url = derive_mgmt_url(&amqp).unwrap();
        assert_eq!(url, "http://rabbitmq_management:15672");
    }

    #[test]
    fn derive_mgmt_url_works_with_amqps_scheme() {
        let amqp = Url::parse("amqps://lapin:secret@cluster.example:5671/%2F").unwrap();
        let url = derive_mgmt_url(&amqp).unwrap();
        // Management is always HTTP in our setup; production-strict TLS
        // can be a follow-up env-var if needed.
        assert_eq!(url, "http://cluster.example:15672");
    }

    #[test]
    fn vhost_from_amqp_url_handles_default() {
        let amqp = Url::parse("amqp://lapin:secret@host:5672/%2F").unwrap();
        assert_eq!(vhost_from_amqp_url(&amqp), "/");
    }

    #[test]
    fn vhost_from_amqp_url_handles_named() {
        let amqp = Url::parse("amqp://lapin:secret@host:5672/prod").unwrap();
        assert_eq!(vhost_from_amqp_url(&amqp), "prod");
    }

    #[test]
    fn vhost_from_amqp_url_handles_empty_path() {
        // Bare URL without trailing slash — broker defaults to vhost `/`.
        let amqp = Url::parse("amqp://lapin:secret@host:5672").unwrap();
        assert_eq!(vhost_from_amqp_url(&amqp), "/");
    }

    #[test]
    fn urlencoding_handles_default_vhost() {
        assert_eq!(urlencoding_encode("/"), "%2F");
    }

    #[test]
    fn urlencoding_passes_through_alphanumerics() {
        assert_eq!(urlencoding_encode("prod-shift_v2"), "prod-shift_v2");
    }

    #[test]
    fn parse_exchanges_response_extracts_name_and_kind() {
        let body = r#"[
            {"name":"","type":"direct","durable":true},
            {"name":"ai.events","type":"topic","durable":true,"auto_delete":false},
            {"name":"amq.direct","type":"direct","durable":true}
        ]"#;
        let exchanges: Vec<ExchangeInfo> = serde_json::from_str(body).unwrap();
        assert_eq!(exchanges.len(), 3);
        assert_eq!(exchanges[1].name, "ai.events");
        assert_eq!(exchanges[1].kind, "topic");
    }

    #[test]
    fn parse_queues_response_extracts_name() {
        let body = r#"[
            {"name":"mcp-master.incidents","durable":true,"messages":0},
            {"name":"frontend.ai_incidents","durable":true,"messages":3}
        ]"#;
        let queues: Vec<QueueInfo> = serde_json::from_str(body).unwrap();
        assert_eq!(queues.len(), 2);
        assert_eq!(queues[0].name, "mcp-master.incidents");
    }

    #[test]
    fn parse_bindings_response_extracts_source_destination_routing_key() {
        let body = r#"[
            {
                "source":"ai.events",
                "destination":"frontend.ai_incidents",
                "destination_type":"queue",
                "routing_key":"event.incident_diagnosed",
                "arguments":{}
            },
            {
                "source":"",
                "destination":"mcp-master.incidents",
                "destination_type":"queue",
                "routing_key":"mcp-master.incidents",
                "arguments":{}
            }
        ]"#;
        let bindings: Vec<BindingInfo> = serde_json::from_str(body).unwrap();
        assert_eq!(bindings.len(), 2);
        assert_eq!(bindings[0].source, "ai.events");
        assert_eq!(bindings[0].destination, "frontend.ai_incidents");
        assert_eq!(bindings[0].routing_key, "event.incident_diagnosed");
    }
}
