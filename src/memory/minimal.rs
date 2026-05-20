use anyhow::{Context, Result};
use blake3::Hasher;
use rusqlite::{params, Connection, OptionalExtension};
use std::sync::Mutex;
use std::time::{SystemTime, UNIX_EPOCH};

const RESPONSE_CACHE_TTL_MS: i64 = 60_000;

/// Very small memory representation.
pub struct Memory {
    pub id: i64,
    pub text: String,
    pub embedding: Vec<f32>,
    pub created_at_unix_ms: i64,
}

/// Compact sqlite-backed cache + semantic memory.
pub struct SqliteMemory {
    conn: Mutex<Connection>,
}

fn now_unix_ms() -> i64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default()
        .as_millis() as i64
}

impl SqliteMemory {
    /// Open a sqlite file and ensure tables exist.
    pub fn open(path: &str) -> Result<Self> {
        let conn = Connection::open(path).context("opening sqlite file")?;
        conn.pragma_update(None, "journal_mode", &"wal")?;

        conn.execute_batch(
            "BEGIN;
             CREATE TABLE IF NOT EXISTS responses (
               prompt_hash TEXT PRIMARY KEY,
               prompt_norm TEXT NOT NULL,
               response TEXT NOT NULL,
               created_at INTEGER NOT NULL
             );

             CREATE TABLE IF NOT EXISTS memories (
               id INTEGER PRIMARY KEY AUTOINCREMENT,
               text TEXT NOT NULL,
               embedding BLOB NOT NULL,
               created_at INTEGER NOT NULL
             );
             COMMIT;",
        )?;

        Ok(Self {
            conn: Mutex::new(conn),
        })
    }

    /// Normalize prompt: trim and collapse whitespace.
    pub fn normalize_prompt(prompt: &str) -> String {
        let mut out = String::with_capacity(prompt.len());
        let mut last_was_space = false;
        for ch in prompt.chars() {
            if ch.is_whitespace() {
                if !last_was_space {
                    out.push(' ');
                    last_was_space = true;
                }
            } else {
                out.push(ch);
                last_was_space = false;
            }
        }
        out.trim().to_string()
    }

    /// Hash normalized prompt with blake3 hex.
    pub fn hash_prompt(norm: &str) -> String {
        let mut h = Hasher::new();
        h.update(norm.as_bytes());
        h.finalize().to_hex().to_string()
    }

    /// Lookup exact cached response for a prompt. Returns None if not present.
    pub fn lookup_response(&self, prompt: &str) -> Result<Option<String>> {
        let norm = Self::normalize_prompt(prompt);
        let key = Self::hash_prompt(&norm);
        let now = now_unix_ms();
        let conn = self.conn.lock().expect("sqlite cache mutex poisoned");
        let row: Option<(String, i64)> = conn
            .query_row(
                "SELECT response, created_at FROM responses WHERE prompt_hash = ?1",
                params![key],
                |r| Ok((r.get(0)?, r.get(1)?)),
            )
            .optional()
            .context("querying response cache")?;

        match row {
            Some((resp, created_at)) if now - created_at <= RESPONSE_CACHE_TTL_MS => Ok(Some(resp)),
            Some((_resp, _created_at)) => {
                conn.execute("DELETE FROM responses WHERE prompt_hash = ?1", params![key])
                    .context("evicting stale response cache entry")?;
                Ok(None)
            }
            None => Ok(None),
        }
    }

    /// Store exact response mapping for a prompt.
    pub fn store_response(&self, prompt: &str, response: &str) -> Result<()> {
        let norm = Self::normalize_prompt(prompt);
        let key = Self::hash_prompt(&norm);
        let now = now_unix_ms();
        let conn = self.conn.lock().expect("sqlite cache mutex poisoned");
        conn.execute(
            "INSERT OR REPLACE INTO responses (prompt_hash, prompt_norm, response, created_at) VALUES (?1, ?2, ?3, ?4)",
            params![key, norm, response, now],
        )?;
        Ok(())
    }

    /// Add a semantic memory entry. Embedding is stored as little-endian f32 bytes.
    pub fn add_memory(&self, text: &str, embedding: &[f32]) -> Result<i64> {
        let blob = f32_slice_to_bytes(embedding);
        let now = now_unix_ms();
        let conn = self.conn.lock().expect("sqlite cache mutex poisoned");
        conn.execute(
            "INSERT INTO memories (text, embedding, created_at) VALUES (?1, ?2, ?3)",
            params![text, blob, now],
        )?;
        Ok(conn.last_insert_rowid())
    }

    /// Return top-k memories by cosine similarity with the provided embedding.
    pub fn query_similar(&self, embedding: &[f32], top_k: usize) -> Result<Vec<(Memory, f32)>> {
        let conn = self.conn.lock().expect("sqlite cache mutex poisoned");
        let mut stmt = conn
            .prepare("SELECT id, text, embedding, created_at FROM memories")?;
        let rows = stmt.query_map([], |row| {
            let id: i64 = row.get(0)?;
            let text: String = row.get(1)?;
            let blob: Vec<u8> = row.get(2)?;
            let created: i64 = row.get(3)?;
            let emb = bytes_to_f32_vec(&blob);
            Ok((id, text, emb, created))
        })?;

        let mut scored = Vec::new();
        for r in rows {
            let (id, text, emb, created) = r?;
            if emb.len() != embedding.len() {
                continue;
            }
            let score = cosine_similarity(embedding, &emb);
            scored.push((Memory { id, text, embedding: emb, created_at_unix_ms: created }, score));
        }

        scored.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
        if scored.len() > top_k {
            scored.truncate(top_k);
        }
        Ok(scored)
    }
}

fn f32_slice_to_bytes(slice: &[f32]) -> Vec<u8> {
    let mut v = Vec::with_capacity(slice.len() * 4);
    for &f in slice {
        v.extend_from_slice(&f.to_le_bytes());
    }
    v
}

fn bytes_to_f32_vec(bytes: &[u8]) -> Vec<f32> {
    let mut v = Vec::with_capacity(bytes.len() / 4);
    let mut i = 0;
    while i + 4 <= bytes.len() {
        let b = [bytes[i], bytes[i + 1], bytes[i + 2], bytes[i + 3]];
        v.push(f32::from_le_bytes(b));
        i += 4;
    }
    v
}

fn dot(a: &[f32], b: &[f32]) -> f32 {
    a.iter().zip(b).map(|(x, y)| x * y).sum()
}

fn norm(a: &[f32]) -> f32 {
    dot(a, a).sqrt()
}

fn cosine_similarity(a: &[f32], b: &[f32]) -> f32 {
    let na = norm(a);
    let nb = norm(b);
    if na == 0.0 || nb == 0.0 {
        0.0
    } else {
        dot(a, b) / (na * nb)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::NamedTempFile;

    #[test]
    fn cache_roundtrip() {
        let tmp = NamedTempFile::new().unwrap();
        let db = SqliteMemory::open(tmp.path().to_str().unwrap()).unwrap();
        let prompt = "  Hello   World \n";
        assert!(db.lookup_response(prompt).unwrap().is_none());
        db.store_response(prompt, "Answer 1").unwrap();
        let got = db.lookup_response("Hello World").unwrap().unwrap();
        assert_eq!(got, "Answer 1");
    }

    #[test]
    fn cache_expires_after_one_minute() {
        let tmp = NamedTempFile::new().unwrap();
        let db = SqliteMemory::open(tmp.path().to_str().unwrap()).unwrap();
        let prompt = "cache me";
        db.store_response(prompt, "fresh").unwrap();

        let key = SqliteMemory::hash_prompt(&SqliteMemory::normalize_prompt(prompt));
        let conn = db.conn.lock().expect("sqlite cache mutex poisoned");
        conn.execute(
            "UPDATE responses SET created_at = ?1 WHERE prompt_hash = ?2",
            params![now_unix_ms() - RESPONSE_CACHE_TTL_MS - 1, key],
        )
        .unwrap();
        drop(conn);

        assert!(db.lookup_response(prompt).unwrap().is_none());
    }

    #[test]
    fn cache_keeps_entries_within_ttl() {
        let tmp = NamedTempFile::new().unwrap();
        let db = SqliteMemory::open(tmp.path().to_str().unwrap()).unwrap();
        let prompt = "still fresh";
        db.store_response(prompt, "fresh").unwrap();

        assert_eq!(db.lookup_response(prompt).unwrap().as_deref(), Some("fresh"));
    }

    #[test]
    fn memory_store_and_query() {
        let tmp = NamedTempFile::new().unwrap();
        let db = SqliteMemory::open(tmp.path().to_str().unwrap()).unwrap();
        let a = vec![1.0f32, 0.0, 0.0];
        let b = vec![0.0f32, 1.0, 0.0];
        db.add_memory("hello", &a).unwrap();
        db.add_memory("goodbye", &b).unwrap();
        let res = db.query_similar(&a, 2).unwrap();
        assert_eq!(res.len(), 2);
        assert!(res[0].1 > res[1].1);
    }
}
