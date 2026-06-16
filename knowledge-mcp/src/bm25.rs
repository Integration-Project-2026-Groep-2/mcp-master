use std::collections::HashMap;

const K1: f64 = 1.2;
const B: f64 = 0.75;

/// In-memory BM25 ranker over a fixed set of documents (chunk texts).
pub struct Bm25 {
    docs: Vec<DocStats>,
    df: HashMap<String, usize>,
    avgdl: f64,
}

struct DocStats {
    len: usize,
    tf: HashMap<String, usize>,
}

impl Bm25 {
    pub fn build(docs: &[String]) -> Self {
        let mut stats = Vec::with_capacity(docs.len());
        let mut df: HashMap<String, usize> = HashMap::new();
        let mut total_len = 0usize;
        for doc in docs {
            let tokens = tokenize(doc);
            total_len += tokens.len();
            let mut tf: HashMap<String, usize> = HashMap::new();
            for t in tokens {
                *tf.entry(t).or_default() += 1;
            }
            for term in tf.keys() {
                *df.entry(term.clone()).or_default() += 1;
            }
            stats.push(DocStats {
                len: tf.values().sum(),
                tf,
            });
        }
        let avgdl = if docs.is_empty() {
            0.0
        } else {
            total_len as f64 / docs.len() as f64
        };
        Self {
            docs: stats,
            df,
            avgdl,
        }
    }

    /// Top-`k` documents by BM25 score (descending). Only positive scores are returned.
    pub fn search(&self, query: &str, k: usize) -> Vec<(usize, f64)> {
        let q_terms = tokenize(query);
        let n = self.docs.len() as f64;
        let mut scored: Vec<(usize, f64)> = self
            .docs
            .iter()
            .enumerate()
            .filter_map(|(id, doc)| {
                let mut score = 0.0;
                for term in &q_terms {
                    let Some(&tf_raw) = doc.tf.get(term) else {
                        continue;
                    };
                    let tf = tf_raw as f64;
                    let df = *self.df.get(term).unwrap_or(&0) as f64;
                    let idf = ((n - df + 0.5) / (df + 0.5) + 1.0).ln();
                    let denom = tf + K1 * (1.0 - B + B * (doc.len as f64 / self.avgdl.max(1.0)));
                    score += idf * (tf * (K1 + 1.0)) / denom;
                }
                (score > 0.0).then_some((id, score))
            })
            .collect();
        scored.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
        scored.truncate(k);
        scored
    }
}

fn tokenize(text: &str) -> Vec<String> {
    text.to_lowercase()
        .split(|c: char| !c.is_alphanumeric())
        .filter(|s| !s.is_empty())
        .map(String::from)
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn tokenize_splits_ids_into_terms() {
        assert_eq!(tokenize("Contract 7"), vec!["contract", "7"]);
        assert_eq!(tokenize("contract-7"), vec!["contract", "7"]);
        assert_eq!(tokenize("Contract_7!"), vec!["contract", "7"]);
    }

    #[test]
    fn search_ranks_matching_doc_first() {
        let docs = vec![
            "the heartbeat contract 7 defines the routing key".to_string(),
            "company deduplication uses the vat number".to_string(),
            "totally unrelated text about coffee".to_string(),
        ];
        let bm25 = Bm25::build(&docs);
        let hits = bm25.search("how does contract 7 work", 3);
        assert!(!hits.is_empty());
        assert_eq!(hits[0].0, 0, "doc 0 should rank first");
    }

    #[test]
    fn search_returns_empty_on_no_match() {
        let bm25 = Bm25::build(&["alpha beta".to_string()]);
        assert!(bm25.search("gamma delta", 5).is_empty());
    }

    #[test]
    fn search_respects_top_k() {
        let docs: Vec<String> = (0..10)
            .map(|_| "contract seven heartbeat".to_string())
            .collect();
        let bm25 = Bm25::build(&docs);
        assert_eq!(bm25.search("contract", 3).len(), 3);
    }
}
