use std::collections::HashMap;

use crate::bm25::Bm25;
use crate::corpus::Chunk;
use crate::embed::Embedder;

const RRF_K: f64 = 60.0;

#[derive(Debug, Clone)]
pub struct Hit {
    pub text: String,
    pub source: String,
}

/// Hybrid retriever: dense cosine (e5) + BM25, fused with Reciprocal Rank Fusion.
pub struct HybridIndex {
    chunks: Vec<Chunk>,
    vectors: Vec<Vec<f32>>,
    bm25: Bm25,
    embedder: Embedder,
}

impl HybridIndex {
    pub fn build(chunks: Vec<Chunk>, embedder: Embedder) -> anyhow::Result<Self> {
        let texts: Vec<String> = chunks.iter().map(|c| c.text.clone()).collect();
        let bm25 = Bm25::build(&texts);
        let mut vectors = embedder.embed_passages(&texts)?;
        for v in &mut vectors {
            normalize(v);
        }
        Ok(Self {
            chunks,
            vectors,
            bm25,
            embedder,
        })
    }

    pub fn search(&self, query: &str, k: usize) -> anyhow::Result<Vec<Hit>> {
        let mut qv = self.embedder.embed_query(query)?;
        normalize(&mut qv);
        let pool = (k * 4).max(10);
        let dense = dense_search(&qv, &self.vectors, pool);
        let lexical = self.bm25.search(query, pool);
        Ok(rrf_fuse(&dense, &lexical, RRF_K, k)
            .into_iter()
            .map(|(id, _score)| Hit {
                text: self.chunks[id].text.clone(),
                source: self.chunks[id].source.clone(),
            })
            .collect())
    }
}

fn normalize(v: &mut [f32]) {
    let norm = v.iter().map(|x| x * x).sum::<f32>().sqrt();
    if norm > 0.0 {
        for x in v.iter_mut() {
            *x /= norm;
        }
    }
}

/// Dot product — valid as cosine because both inputs are normalized.
fn cosine(a: &[f32], b: &[f32]) -> f32 {
    a.iter().zip(b).map(|(x, y)| x * y).sum()
}

fn dense_search(query_vec: &[f32], vectors: &[Vec<f32>], k: usize) -> Vec<(usize, f64)> {
    let mut scored: Vec<(usize, f64)> = vectors
        .iter()
        .enumerate()
        .map(|(i, v)| (i, f64::from(cosine(query_vec, v))))
        .collect();
    scored.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
    scored.truncate(k);
    scored
}

/// Reciprocal Rank Fusion: score(d) = Σ 1/(rrf_k + rank(d)) over both ranked lists.
fn rrf_fuse(
    dense: &[(usize, f64)],
    lexical: &[(usize, f64)],
    rrf_k: f64,
    k: usize,
) -> Vec<(usize, f64)> {
    let mut fused: HashMap<usize, f64> = HashMap::new();
    for list in [dense, lexical] {
        for (rank, (id, _)) in list.iter().enumerate() {
            *fused.entry(*id).or_default() += 1.0 / (rrf_k + (rank + 1) as f64);
        }
    }
    let mut out: Vec<(usize, f64)> = fused.into_iter().collect();
    out.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
    out.truncate(k);
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn normalize_makes_unit_vector() {
        let mut v = vec![3.0f32, 4.0];
        normalize(&mut v);
        let n = (v[0] * v[0] + v[1] * v[1]).sqrt();
        assert!((n - 1.0).abs() < 1e-6);
    }

    #[test]
    fn dense_search_ranks_nearest_first() {
        let vectors = vec![vec![1.0, 0.0], vec![0.0, 1.0], vec![0.7, 0.7]];
        let hits = dense_search(&[1.0, 0.0], &vectors, 3);
        assert_eq!(hits[0].0, 0);
        assert_eq!(hits[2].0, 1);
    }

    #[test]
    fn rrf_prefers_doc_high_in_both_lists() {
        let dense = vec![(5, 0.9), (1, 0.8), (2, 0.7)];
        let lexical = vec![(5, 10.0), (3, 8.0), (1, 5.0)];
        let fused = rrf_fuse(&dense, &lexical, RRF_K, 3);
        assert_eq!(fused[0].0, 5, "doc ranked high in both must win");
    }

    #[test]
    fn rrf_respects_top_k() {
        let dense = vec![(0, 1.0), (1, 1.0), (2, 1.0), (3, 1.0)];
        let lexical = vec![(4, 1.0), (5, 1.0)];
        assert_eq!(rrf_fuse(&dense, &lexical, RRF_K, 3).len(), 3);
    }

    #[test]
    #[ignore = "loads the e5 ONNX model"]
    fn hybrid_surfaces_relevant_chunk() {
        let chunks = vec![
            Chunk {
                text: "Contract 7 is the heartbeat exchange with routing key routing.heartbeat"
                    .into(),
                source: "a.md".into(),
            },
            Chunk {
                text: "Company deduplication uses the VAT number".into(),
                source: "b.md".into(),
            },
            Chunk {
                text: "Coffee brewing methods and espresso".into(),
                source: "c.md".into(),
            },
        ];
        let index = HybridIndex::build(chunks, Embedder::new().unwrap()).unwrap();
        let hits = index.search("how does contract 7 work", 2).unwrap();
        assert!(!hits.is_empty());
        assert_eq!(hits[0].source, "a.md");
    }
}
