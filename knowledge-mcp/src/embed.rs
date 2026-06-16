use std::sync::Mutex;

use anyhow::Context;
use fastembed::{EmbeddingModel, InitOptions, TextEmbedding};

/// Local ONNX embedder (multilingual-e5-small) — handles NL + EN docs without an API key.
/// `embed` needs `&mut`, so the model lives behind a `Mutex` to keep the embedder
/// shareable via `&self` (the index/server hold it in an `Arc`). The guard never
/// crosses an `.await` — `embed` is synchronous CPU work.
pub struct Embedder {
    model: Mutex<TextEmbedding>,
}

impl Embedder {
    pub fn new() -> anyhow::Result<Self> {
        let model = TextEmbedding::try_new(
            InitOptions::new(EmbeddingModel::MultilingualE5Small)
                .with_show_download_progress(false),
        )
        .context("init multilingual-e5-small embedder")?;
        Ok(Self {
            model: Mutex::new(model),
        })
    }

    /// Embed corpus chunks. e5 expects the `passage:` prefix for documents.
    pub fn embed_passages(&self, texts: &[String]) -> anyhow::Result<Vec<Vec<f32>>> {
        let prefixed: Vec<String> = texts.iter().map(|t| format!("passage: {t}")).collect();
        let mut model = self.model.lock().expect("embedder mutex poisoned");
        model.embed(prefixed, None).context("embed passages")
    }

    /// Embed a search query. e5 expects the `query:` prefix.
    pub fn embed_query(&self, query: &str) -> anyhow::Result<Vec<f32>> {
        let mut model = self.model.lock().expect("embedder mutex poisoned");
        let mut out = model
            .embed(vec![format!("query: {query}")], None)
            .context("embed query")?;
        out.pop().context("embedder returned no vector")
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    #[ignore = "downloads the e5 ONNX model on first run"]
    fn passage_and_query_share_dimension() {
        let e = Embedder::new().unwrap();
        let passages = e
            .embed_passages(&["heartbeat contract".to_string()])
            .unwrap();
        let q = e.embed_query("contract").unwrap();
        assert_eq!(passages.len(), 1);
        assert!(!q.is_empty());
        assert_eq!(
            passages[0].len(),
            q.len(),
            "passage and query embeddings must share dimension"
        );
    }
}
