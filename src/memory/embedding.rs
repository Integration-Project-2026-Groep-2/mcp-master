use std::time::Duration;

use anyhow::{Context, Result};
use async_trait::async_trait;
use serde::{Deserialize, Serialize};

use super::config::EmbeddingConfig;

#[async_trait]
pub trait EmbeddingClient: Send + Sync {
    async fn embed_texts(&self, texts: &[String]) -> Result<Vec<Vec<f32>>>;
}

#[derive(Clone)]
pub struct HttpEmbeddingClient {
    http: reqwest::Client,
    base_url: String,
    model: String,
    api_key: Option<String>,
    endpoint_path: String,
    dimension: usize,
}

impl HttpEmbeddingClient {
    pub fn new(config: &EmbeddingConfig) -> Result<Self> {
        let http = reqwest::Client::builder()
            .timeout(Duration::from_secs(config.request_timeout_secs))
            .build()
            .context("building embedding HTTP client")?;

        Ok(Self {
            http,
            base_url: config.base_url.clone(),
            model: config.model.clone(),
            api_key: config.api_key.clone(),
            endpoint_path: config.endpoint_path.clone(),
            dimension: config.dimension,
        })
    }
}

#[async_trait]
impl EmbeddingClient for HttpEmbeddingClient {
    async fn embed_texts(&self, texts: &[String]) -> Result<Vec<Vec<f32>>> {
        if texts.is_empty() {
            return Ok(Vec::new());
        }

        let url = format!("{}{}", self.base_url.trim_end_matches('/'), self.endpoint_path.as_str());
        let body = EmbeddingsRequest {
            model: &self.model,
            input: texts,
        };

        let mut request = self.http.post(&url).json(&body);
        if let Some(api_key) = &self.api_key {
            request = request.bearer_auth(api_key);
        }

        let response = request
            .send()
            .await
            .with_context(|| format!("embedding request failed for {url}"))?
            .error_for_status()
            .with_context(|| format!("embedding endpoint returned error status for {url}"))?;

        let parsed: EmbeddingsResponse = response
            .json()
            .await
            .context("decoding embeddings response JSON")?;

        let mut vectors = parsed.data;
        vectors.sort_by_key(|item| item.index);
        let vectors: Vec<Vec<f32>> = vectors.into_iter().map(|item| item.embedding).collect();

        if vectors.iter().any(|embedding| embedding.len() != self.dimension) {
            anyhow::bail!(
                "embedding dimension mismatch: expected {}, got {:?}",
                self.dimension,
                vectors.iter().map(Vec::len).collect::<Vec<_>>()
            );
        }

        Ok(vectors)
    }
}

#[derive(Debug, Serialize)]
struct EmbeddingsRequest<'a> {
    model: &'a str,
    input: &'a [String],
}

#[derive(Debug, Deserialize)]
struct EmbeddingsResponse {
    data: Vec<EmbeddingDatum>,
}

#[derive(Debug, Deserialize)]
struct EmbeddingDatum {
    index: usize,
    embedding: Vec<f32>,
}
