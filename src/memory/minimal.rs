use anyhow::{Context, Result};
use blake3::Hasher;
use rusqlite::{params, Connection, OptionalExtension};
use std::sync::Mutex;
use std::time::{SystemTime, UNIX_EPOCH};

const RESPONSE_CACHE_TTL_MS: i64 = 60_000;

/// Compact sqlite-backed response cache.
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
            "CREATE TABLE IF NOT EXISTS responses (
               prompt_hash TEXT PRIMARY KEY,
               response TEXT NOT NULL,
               created_at INTEGER NOT NULL
             );",
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
            "INSERT OR REPLACE INTO responses (prompt_hash, response, created_at) VALUES (?1, ?2, ?3)",
            params![key, response, now],
        )?;
        Ok(())
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
}
