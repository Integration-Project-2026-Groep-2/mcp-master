use std::path::Path;

use anyhow::Context;
use text_splitter::{ChunkConfig, MarkdownSplitter};
use walkdir::WalkDir;

const CHUNK_CAPACITY: usize = 1500;
const CHUNK_OVERLAP: usize = 200;

#[derive(Debug, Clone)]
pub struct Chunk {
    pub text: String,
    pub source: String,
}

/// Load every `.md` file under `dir` (recursively) and split it into chunks.
/// `source` is the file path relative to `dir`, slash-normalized for stable citations.
pub fn load_corpus(dir: &Path) -> anyhow::Result<Vec<Chunk>> {
    let mut chunks = Vec::new();
    for entry in WalkDir::new(dir).into_iter().filter_map(Result::ok) {
        let path = entry.path();
        if path.extension().and_then(|e| e.to_str()) != Some("md") {
            continue;
        }
        let text =
            std::fs::read_to_string(path).with_context(|| format!("reading {}", path.display()))?;
        let source = path
            .strip_prefix(dir)
            .unwrap_or(path)
            .to_string_lossy()
            .replace('\\', "/");
        chunks.extend(chunk_markdown(&text, &source)?);
    }
    Ok(chunks)
}

fn chunk_markdown(text: &str, source: &str) -> anyhow::Result<Vec<Chunk>> {
    let splitter =
        MarkdownSplitter::new(ChunkConfig::new(CHUNK_CAPACITY).with_overlap(CHUNK_OVERLAP)?);
    Ok(splitter
        .chunks(text)
        .filter(|c| !c.trim().is_empty())
        .map(|c| Chunk {
            text: c.to_string(),
            source: source.to_string(),
        })
        .collect())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn chunk_markdown_splits_long_doc() {
        let doc = format!("# Title\n\n{}", "lorem ipsum dolor sit amet. ".repeat(200));
        let chunks = chunk_markdown(&doc, "x.md").unwrap();
        assert!(
            chunks.len() > 1,
            "expected multiple chunks, got {}",
            chunks.len()
        );
        assert!(chunks.iter().all(|c| !c.text.trim().is_empty()));
        assert!(chunks.iter().all(|c| c.source == "x.md"));
    }

    #[test]
    fn chunk_markdown_short_doc_single_chunk() {
        let chunks = chunk_markdown("# Hi\n\nshort.", "y.md").unwrap();
        assert_eq!(chunks.len(), 1);
        assert!(chunks[0].text.contains("short"));
    }

    #[test]
    fn load_corpus_reads_only_markdown_recursively() {
        let dir = tempfile::tempdir().unwrap();
        std::fs::write(dir.path().join("a.md"), "# A\n\nalpha content here.").unwrap();
        std::fs::write(dir.path().join("b.txt"), "ignored").unwrap();
        let sub = dir.path().join("sub");
        std::fs::create_dir(&sub).unwrap();
        std::fs::write(sub.join("c.md"), "# C\n\ngamma content.").unwrap();

        let chunks = load_corpus(dir.path()).unwrap();
        let sources: std::collections::HashSet<_> =
            chunks.iter().map(|c| c.source.as_str()).collect();
        assert!(sources.contains("a.md"));
        assert!(sources.contains("sub/c.md"));
        assert!(!sources.iter().any(|s| s.ends_with(".txt")));
    }
}
