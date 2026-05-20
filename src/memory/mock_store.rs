use std::sync::Arc;

use anyhow::Result;
use async_trait::async_trait;
use parking_lot::RwLock;

use super::store::{MemoryPayload, VectorPoint, VectorStore};
use super::types::MemoryHit;

/// In-memory vector store for testing. Stores all points in a Vec and
/// does naive linear search. Not for production use — trades durability
/// and efficiency for convenience in dev/test.
#[derive(Clone)]
pub struct InMemoryVectorStore {
    points: Arc<RwLock<Vec<StoredPoint>>>,
}

struct StoredPoint {
    vector: Vec<f32>,
    payload: MemoryPayload,
}

impl InMemoryVectorStore {
    pub fn new() -> Self {
        Self {
            points: Arc::new(RwLock::new(Vec::new())),
        }
    }

    #[cfg(test)]
    fn point_count(&self) -> usize {
        self.points.read().len()
    }

    /// Linear search: compute cosine similarity to every point and return top-k.
    fn cosine_sim(a: &[f32], b: &[f32]) -> f32 {
        if a.is_empty() || b.is_empty() {
            return 0.0;
        }
        let dot: f32 = a.iter().zip(b).map(|(x, y)| x * y).sum();
        let a_norm: f32 = a.iter().map(|x| x * x).sum::<f32>().sqrt();
        let b_norm: f32 = b.iter().map(|x| x * x).sum::<f32>().sqrt();
        if a_norm == 0.0 || b_norm == 0.0 {
            0.0
        } else {
            dot / (a_norm * b_norm)
        }
    }
}

#[async_trait]
impl VectorStore for InMemoryVectorStore {
    async fn ensure_collection(&self, _collection: &str, _dimension: usize) -> Result<()> {
        // No-op for in-memory store — collection always exists.
        Ok(())
    }

    async fn upsert_points(&self, _collection: &str, points: Vec<VectorPoint>) -> Result<()> {
        let mut store = self.points.write();
        for point in points {
            store.push(StoredPoint {
                vector: point.vector,
                payload: point.payload,
            });
        }
        Ok(())
    }

    async fn search_points(
        &self,
        _collection: &str,
        namespace: &str,
        user_id: Option<&str>,
        vector: &[f32],
        top_k: usize,
    ) -> Result<Vec<MemoryHit>> {
        let store = self.points.read();
        let mut hits: Vec<_> = store
            .iter()
            .filter(|p| {
                p.payload.namespace == namespace
                    && (user_id.is_none()
                        || user_id.is_some_and(|uid| {
                            p.payload.user_id.as_deref() == Some(uid) || uid.is_empty()
                        }))
            })
            .map(|p| {
                let score = Self::cosine_sim(vector, &p.vector);
                MemoryHit {
                    score,
                    namespace: p.payload.namespace.clone(),
                    source: p.payload.source,
                    correlation_id: p.payload.correlation_id.clone(),
                    user_id: p.payload.user_id.clone(),
                    text: p.payload.text.clone(),
                    chunk_index: p.payload.chunk_index,
                    chunk_count: p.payload.chunk_count,
                    created_at_unix_ms: p.payload.created_at_unix_ms,
                }
            })
            .collect();

        hits.sort_by(|a, b| {
            b.score
                .partial_cmp(&a.score)
                .unwrap_or(std::cmp::Ordering::Equal)
        });
        hits.truncate(top_k);
        Ok(hits)
    }

    async fn delete_by_user(&self, _collection: &str, user_id: &str) -> Result<()> {
        self.points
            .write()
            .retain(|p| p.payload.user_id.as_deref() != Some(user_id));
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::memory::types::MemorySource;

    #[tokio::test]
    async fn in_memory_store_upserts_and_searches() {
        let store = InMemoryVectorStore::new();
        let points = vec![
            VectorPoint {
                id: "1".into(),
                vector: vec![1.0, 0.0, 0.0],
                payload: MemoryPayload {
                    namespace: "default".into(),
                    source: MemorySource::Chat,
                    correlation_id: "cid-1".into(),
                    user_id: None,
                    text: "hello world".into(),
                    chunk_index: 0,
                    chunk_count: 1,
                    created_at_unix_ms: 1,
                },
            },
            VectorPoint {
                id: "2".into(),
                vector: vec![0.0, 1.0, 0.0],
                payload: MemoryPayload {
                    namespace: "default".into(),
                    source: MemorySource::Chat,
                    correlation_id: "cid-2".into(),
                    user_id: None,
                    text: "goodbye world".into(),
                    chunk_index: 0,
                    chunk_count: 1,
                    created_at_unix_ms: 2,
                },
            },
        ];
        store.upsert_points("test", points).await.unwrap();
        assert_eq!(store.point_count(), 2);

        let results = store
            .search_points("test", "default", None, &[1.0, 0.0, 0.0], 1)
            .await
            .unwrap();
        assert_eq!(results.len(), 1);
        assert_eq!(results[0].text, "hello world");
        assert!(results[0].score > 0.99);
    }

    #[tokio::test]
    async fn delete_by_user_removes_only_that_user() {
        let store = InMemoryVectorStore::new();
        let point = |id: &str, user: &str| VectorPoint {
            id: id.into(),
            vector: vec![1.0, 0.0, 0.0],
            payload: MemoryPayload {
                namespace: "default".into(),
                source: MemorySource::Chat,
                correlation_id: "cid".into(),
                user_id: Some(user.into()),
                text: "t".into(),
                chunk_index: 0,
                chunk_count: 1,
                created_at_unix_ms: 1,
            },
        };
        store
            .upsert_points("c", vec![point("1", "alice"), point("2", "bob")])
            .await
            .unwrap();

        store.delete_by_user("c", "alice").await.unwrap();

        let alice = store
            .search_points("c", "default", Some("alice"), &[1.0, 0.0, 0.0], 10)
            .await
            .unwrap();
        let bob = store
            .search_points("c", "default", Some("bob"), &[1.0, 0.0, 0.0], 10)
            .await
            .unwrap();
        assert!(alice.is_empty());
        assert_eq!(bob.len(), 1);
    }

    #[tokio::test]
    async fn search_scopes_results_to_user_id() {
        let store = InMemoryVectorStore::new();
        let point = |id: &str, user: &str| VectorPoint {
            id: id.into(),
            vector: vec![1.0, 0.0, 0.0],
            payload: MemoryPayload {
                namespace: "default".into(),
                source: MemorySource::Chat,
                correlation_id: "cid".into(),
                user_id: Some(user.into()),
                text: format!("secret of {user}"),
                chunk_index: 0,
                chunk_count: 1,
                created_at_unix_ms: 1,
            },
        };
        store
            .upsert_points("c", vec![point("1", "alice"), point("2", "bob")])
            .await
            .unwrap();

        let bob = store
            .search_points("c", "default", Some("bob"), &[1.0, 0.0, 0.0], 10)
            .await
            .unwrap();
        assert_eq!(bob.len(), 1);
        assert_eq!(bob[0].user_id.as_deref(), Some("bob"));
    }
}
