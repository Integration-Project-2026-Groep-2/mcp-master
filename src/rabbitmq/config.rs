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
}
