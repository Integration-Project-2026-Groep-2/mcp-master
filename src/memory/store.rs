use std::collections::BTreeMap;

use anyhow::{Context, Result};
use async_trait::async_trait;
use serde::{Deserialize, Serialize};
use serde_json::{Map, Value};

use super::config::QdrantConfig;
use super::types::{MemoryHit, MemorySource};

#[async_trait]
pub trait VectorStore: Send + Sync {
    async fn ensure_collection(&self, collection: &str, dimension: usize) -> Result<()>;
    async fn upsert_points(&self, collection: &str, points: Vec<VectorPoint>) -> Result<()>;
    async fn search_points(
        &self,
        collection: &str,
        namespace: &str,
        user_id: Option<&str>,
        vector: &[f32],
        top_k: usize,
    ) -> Result<Vec<MemoryHit>>;
    async fn delete_by_user(&self, collection: &str, user_id: &str) -> Result<()>;
}

#[derive(Debug, Clone)]
pub struct VectorPoint {
    pub id: String,
    pub vector: Vec<f32>,
    pub payload: MemoryPayload,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MemoryPayload {
    pub namespace: String,
    pub source: MemorySource,
    pub correlation_id: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub user_id: Option<String>,
    pub text: String,
    pub chunk_index: u32,
    pub chunk_count: u32,
    pub created_at_unix_ms: i64,
}

/// Lenient deadline on Qdrant calls — mirrors the embedding client; only trips
/// on a hung connection, not on a slow-but-legitimate query.
const QDRANT_REQUEST_TIMEOUT_SECS: u64 = 60;

#[derive(Clone)]
pub struct QdrantVectorStore {
    http: reqwest::Client,
    base_url: String,
    api_key: Option<String>,
}

impl QdrantVectorStore {
    pub fn new(config: &QdrantConfig) -> Result<Self> {
        Ok(Self {
            http: reqwest::Client::builder()
                .timeout(std::time::Duration::from_secs(QDRANT_REQUEST_TIMEOUT_SECS))
                .build()
                .context("building qdrant HTTP client")?,
            base_url: config.base_url.clone(),
            api_key: config.collection_api_key.clone(),
        })
    }

    fn collection_url(&self, collection: &str) -> String {
        format!(
            "{}/collections/{}",
            self.base_url.trim_end_matches('/'),
            collection
        )
    }

    fn points_url(&self, collection: &str) -> String {
        format!(
            "{}/collections/{}/points",
            self.base_url.trim_end_matches('/'),
            collection
        )
    }

    fn request(&self, method: reqwest::Method, url: &str) -> reqwest::RequestBuilder {
        let request = self.http.request(method, url);
        if let Some(api_key) = &self.api_key {
            request.header("api-key", api_key)
        } else {
            request
        }
    }
}

#[async_trait]
impl VectorStore for QdrantVectorStore {
    async fn ensure_collection(&self, collection: &str, dimension: usize) -> Result<()> {
        let url = self.collection_url(collection);
        let body = serde_json::json!({
            "vectors": {
                "size": dimension,
                "distance": "Cosine"
            }
        });

        self.request(reqwest::Method::PUT, &url)
            .json(&body)
            .send()
            .await
            .with_context(|| format!("creating qdrant collection {collection}"))?
            .error_for_status()
            .with_context(|| format!("qdrant collection creation failed for {collection}"))?;

        Ok(())
    }

    async fn upsert_points(&self, collection: &str, points: Vec<VectorPoint>) -> Result<()> {
        if points.is_empty() {
            return Ok(());
        }

        let url = self.points_url(collection);
        let payload_points: Vec<Value> = points
            .into_iter()
            .map(|point| {
                serde_json::json!({
                    "id": point.id,
                    "vector": point.vector,
                    "payload": point_payload_to_map(point.payload),
                })
            })
            .collect();

        self.request(reqwest::Method::PUT, &format!("{url}?wait=true"))
            .json(&serde_json::json!({ "points": payload_points }))
            .send()
            .await
            .with_context(|| format!("upserting qdrant points into {collection}"))?
            .error_for_status()
            .with_context(|| format!("qdrant upsert failed for {collection}"))?;

        Ok(())
    }

    async fn search_points(
        &self,
        collection: &str,
        namespace: &str,
        user_id: Option<&str>,
        vector: &[f32],
        top_k: usize,
    ) -> Result<Vec<MemoryHit>> {
        let url = format!("{}/points/search", self.points_url(collection));
        let mut must = vec![serde_json::json!({
            "key": "namespace",
            "match": { "value": namespace }
        })];
        if let Some(user_id) = user_id.filter(|value| !value.is_empty()) {
            must.push(serde_json::json!({
                "key": "user_id",
                "match": { "value": user_id }
            }));
        }

        let body = serde_json::json!({
            "vector": vector,
            "limit": top_k,
            "with_payload": true,
            "with_vector": false,
            "filter": { "must": must },
        });

        let response = self
            .request(reqwest::Method::POST, &url)
            .json(&body)
            .send()
            .await
            .with_context(|| format!("qdrant search request failed for {collection}"))?
            .error_for_status()
            .with_context(|| format!("qdrant search failed for {collection}"))?;

        let parsed: QdrantSearchResponse = response
            .json()
            .await
            .context("decoding qdrant search response JSON")?;

        Ok(parsed
            .result
            .into_iter()
            .filter_map(|hit| hit.into_memory_hit())
            .collect())
    }

    async fn delete_by_user(&self, collection: &str, user_id: &str) -> Result<()> {
        let url = format!("{}/delete?wait=true", self.points_url(collection));
        let body = serde_json::json!({
            "filter": { "must": [ { "key": "user_id", "match": { "value": user_id } } ] }
        });
        self.request(reqwest::Method::POST, &url)
            .json(&body)
            .send()
            .await
            .with_context(|| format!("qdrant delete-by-user request failed for {collection}"))?
            .error_for_status()
            .with_context(|| format!("qdrant delete-by-user failed for {collection}"))?;
        Ok(())
    }
}

fn point_payload_to_map(payload: MemoryPayload) -> Map<String, Value> {
    let mut map = Map::new();
    map.insert("namespace".into(), Value::String(payload.namespace));
    map.insert(
        "source".into(),
        Value::String(payload.source.as_str().into()),
    );
    map.insert(
        "correlation_id".into(),
        Value::String(payload.correlation_id),
    );
    if let Some(user_id) = payload.user_id {
        map.insert("user_id".into(), Value::String(user_id));
    }
    map.insert("text".into(), Value::String(payload.text));
    map.insert(
        "chunk_index".into(),
        Value::Number(payload.chunk_index.into()),
    );
    map.insert(
        "chunk_count".into(),
        Value::Number(payload.chunk_count.into()),
    );
    map.insert(
        "created_at_unix_ms".into(),
        Value::Number(payload.created_at_unix_ms.into()),
    );
    map
}

#[derive(Debug, Deserialize)]
struct QdrantSearchResponse {
    result: Vec<QdrantPoint>,
}

#[derive(Debug, Deserialize)]
struct QdrantPoint {
    score: f32,
    payload: Option<BTreeMap<String, Value>>,
}

impl QdrantPoint {
    fn into_memory_hit(self) -> Option<MemoryHit> {
        let payload = self.payload?;
        let namespace = payload.get("namespace")?.as_str()?.to_string();
        let source = match payload.get("source")?.as_str()? {
            "chat" => MemorySource::Chat,
            "scheduled_summary" => MemorySource::ScheduledSummary,
            "incident_evidence" => MemorySource::IncidentEvidence,
            "incident_diagnosis" => MemorySource::IncidentDiagnosis,
            _ => return None,
        };
        let correlation_id = payload.get("correlation_id")?.as_str()?.to_string();
        let user_id = payload
            .get("user_id")
            .and_then(|value| value.as_str())
            .map(ToString::to_string);
        let text = payload.get("text")?.as_str()?.to_string();
        let chunk_index = payload.get("chunk_index")?.as_u64()? as u32;
        let chunk_count = payload.get("chunk_count")?.as_u64()? as u32;
        let created_at_unix_ms = payload.get("created_at_unix_ms")?.as_i64()?;

        Some(MemoryHit {
            score: self.score,
            namespace,
            source,
            correlation_id,
            user_id,
            text,
            chunk_index,
            chunk_count,
            created_at_unix_ms,
        })
    }
}
