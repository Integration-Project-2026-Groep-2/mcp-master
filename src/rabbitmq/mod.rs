use std::time::Duration;

use anyhow::{Context, Result};
use lapin::{Connection, ConnectionProperties};

pub mod config;
pub mod consumer;
pub mod publisher;

/// Lenient deadline on AMQP connection establishment. A half-up broker (TCP
/// accepts but never finishes the AMQP handshake) would otherwise block the
/// caller — and on the publisher path, the connection mutex — indefinitely.
const CONNECT_TIMEOUT: Duration = Duration::from_secs(30);

/// Establish an AMQP connection with a bounded handshake deadline. All connect
/// paths route through here so a hung broker surfaces as an error the reconnect
/// loops can act on.
pub async fn connect_with_timeout(url: &str) -> Result<Connection> {
    match tokio::time::timeout(
        CONNECT_TIMEOUT,
        Connection::connect(url, ConnectionProperties::default()),
    )
    .await
    {
        Ok(res) => res.context("AMQP connection"),
        Err(_elapsed) => anyhow::bail!("AMQP connect timed out after {CONNECT_TIMEOUT:?}"),
    }
}
