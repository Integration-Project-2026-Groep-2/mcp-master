use anyhow::{Context, Result};

#[derive(Debug, Clone)]
pub struct MemoryConfig {
    pub namespace: String,
    pub collection: String,
    pub retrieval_top_k: usize,
    pub chunk_chars: usize,
    pub chunk_overlap_chars: usize,
    pub ingest_queue_capacity: usize,
    pub max_query_chars: usize,
    pub embedding_batch_size: usize,
    pub embedding: EmbeddingConfig,
    pub qdrant: QdrantConfig,
}

#[derive(Debug, Clone)]
pub struct EmbeddingConfig {
    pub base_url: String,
    pub model: String,
    pub api_key: Option<String>,
    pub endpoint_path: String,
    pub dimension: usize,
    pub request_timeout_secs: u64,
}

#[derive(Debug, Clone)]
pub struct QdrantConfig {
    pub base_url: String,
    pub collection_api_key: Option<String>,
}

impl MemoryConfig {
    pub fn from_env() -> Result<Option<Self>> {
        if !env_bool("MEMORY_ENABLED").unwrap_or(false) {
            return Ok(None);
        }

        let namespace = env_string("MEMORY_NAMESPACE").unwrap_or_else(|_| "default".to_string());
        let collection =
            env_string("MEMORY_COLLECTION").unwrap_or_else(|_| "mcp_master_memory".to_string());

        let retrieval_top_k = env_usize("MEMORY_TOP_K", 5)?;
        let chunk_chars = env_usize("MEMORY_CHUNK_CHARS", 1200)?;
        let chunk_overlap_chars = env_usize("MEMORY_CHUNK_OVERLAP_CHARS", 120)?;
        let ingest_queue_capacity = env_usize("MEMORY_INGEST_QUEUE_CAPACITY", 64)?;
        let max_query_chars = env_usize("MEMORY_MAX_QUERY_CHARS", 4096)?;
        let embedding_batch_size = env_usize("MEMORY_EMBEDDING_BATCH_SIZE", 16)?;

        let embedding = EmbeddingConfig::from_env()?;
        let qdrant = QdrantConfig::from_env()?;

        if chunk_overlap_chars >= chunk_chars {
            anyhow::bail!("MEMORY_CHUNK_OVERLAP_CHARS must be smaller than MEMORY_CHUNK_CHARS");
        }

        Ok(Some(Self {
            namespace,
            collection,
            retrieval_top_k,
            chunk_chars,
            chunk_overlap_chars,
            ingest_queue_capacity,
            max_query_chars,
            embedding_batch_size,
            embedding,
            qdrant,
        }))
    }
}

impl EmbeddingConfig {
    pub fn from_env() -> Result<Self> {
        Ok(Self {
            base_url: env_string("MEMORY_EMBEDDING_URL").context("MEMORY_EMBEDDING_URL not set")?,
            model: env_string("MEMORY_EMBEDDING_MODEL")
                .context("MEMORY_EMBEDDING_MODEL not set")?,
            api_key: env_optional_string("MEMORY_EMBEDDING_API_KEY"),
            endpoint_path: env_string("MEMORY_EMBEDDING_PATH")
                .unwrap_or_else(|_| "/v1/embeddings".to_string()),
            dimension: env_usize("MEMORY_EMBEDDING_DIMENSION", 1536)?,
            request_timeout_secs: env_usize("MEMORY_EMBEDDING_TIMEOUT_SECS", 30)? as u64,
        })
    }
}

impl QdrantConfig {
    pub fn from_env() -> Result<Self> {
        Ok(Self {
            base_url: env_string("MEMORY_QDRANT_URL").context("MEMORY_QDRANT_URL not set")?,
            collection_api_key: env_optional_string("MEMORY_QDRANT_API_KEY"),
        })
    }
}

fn env_string(name: &str) -> Result<String> {
    std::env::var(name).with_context(|| format!("{name} env var is required"))
}

fn env_optional_string(name: &str) -> Option<String> {
    std::env::var(name)
        .ok()
        .filter(|value| !value.trim().is_empty())
}

fn env_usize(name: &str, default: usize) -> Result<usize> {
    match std::env::var(name) {
        Ok(value) => value
            .trim()
            .parse::<usize>()
            .with_context(|| format!("{name} must be a positive integer")),
        Err(_) => Ok(default),
    }
}

fn env_bool(name: &str) -> Option<bool> {
    let raw = std::env::var(name).ok()?;
    let normalized = raw.trim().to_ascii_lowercase();
    match normalized.as_str() {
        "1" | "true" | "yes" | "on" => Some(true),
        "0" | "false" | "no" | "off" => Some(false),
        _ => None,
    }
}
