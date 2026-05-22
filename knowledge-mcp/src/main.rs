mod bm25;
mod corpus;
mod embed;
mod index;
mod server;

use std::path::PathBuf;
use std::sync::Arc;

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    let corpus_dir = std::env::var("CORPUS_DIR").unwrap_or_else(|_| "corpus".to_string());
    let port: u16 = std::env::var("PORT")
        .ok()
        .and_then(|p| p.parse().ok())
        .unwrap_or(7099);

    let chunks = corpus::load_corpus(&PathBuf::from(&corpus_dir))?;
    eprintln!(
        "knowledge-mcp: loaded {} chunks from {corpus_dir}",
        chunks.len()
    );

    let embedder = embed::Embedder::new()?;
    let index = Arc::new(index::HybridIndex::build(chunks, embedder)?);
    eprintln!("knowledge-mcp: hybrid index ready");

    server::serve(index, port).await
}
