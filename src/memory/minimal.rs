use anyhow::{Context, Result};
use blake3::Hasher;
use rusqlite::{Connection, OptionalExtension, params};
use std::path::{Path, PathBuf};
use std::sync::{Arc, Mutex};
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
    /// Open a sqlite file and ensure tables exist. Creates the parent dir if
    /// missing so a configured path under a fresh directory just works.
    pub fn open(path: impl AsRef<Path>) -> Result<Self> {
        let path = path.as_ref();
        if let Some(parent) = path.parent()
            && !parent.as_os_str().is_empty()
        {
            std::fs::create_dir_all(parent)
                .with_context(|| format!("creating cache dir {}", parent.display()))?;
        }
        let conn = Connection::open(path)
            .with_context(|| format!("opening sqlite file {}", path.display()))?;
        conn.pragma_update(None, "journal_mode", "wal")?;

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

    pub fn purge_expired(&self) -> Result<usize> {
        let cutoff = now_unix_ms() - RESPONSE_CACHE_TTL_MS;
        let conn = self.conn.lock().expect("sqlite cache mutex poisoned");
        let removed = conn
            .execute(
                "DELETE FROM responses WHERE created_at < ?1",
                params![cutoff],
            )
            .context("purging expired response cache entries")?;
        Ok(removed)
    }

    pub fn clear(&self) -> Result<()> {
        let conn = self.conn.lock().expect("sqlite cache mutex poisoned");
        conn.execute("DELETE FROM responses", [])
            .context("clearing response cache")?;
        Ok(())
    }
}

const CACHE_FILENAME: &str = "memory-cache.sqlite3";

/// Configured cache path (`RESPONSE_CACHE_PATH`) or a writable temp-dir default.
/// The default avoids assuming a writable CWD — the container runs non-root with
/// a root-owned WORKDIR — and the short TTL makes an ephemeral file fine.
fn cache_path_or_default(env_value: Option<String>) -> PathBuf {
    match env_value {
        Some(p) if !p.trim().is_empty() => PathBuf::from(p),
        _ => std::env::temp_dir().join("mcp-master").join(CACHE_FILENAME),
    }
}

/// Open the cache at `path`, degrading to `None` on failure — a best-effort
/// cache must never abort startup.
fn open_or_disabled(path: impl AsRef<Path>) -> Option<Arc<SqliteMemory>> {
    let path = path.as_ref();
    match SqliteMemory::open(path) {
        Ok(cache) => Some(Arc::new(cache)),
        Err(e) => {
            tracing::warn!(path = %path.display(), "response cache disabled: {e:#}");
            None
        }
    }
}

/// Open the response cache at the configured-or-default path.
pub fn open_response_cache() -> Option<Arc<SqliteMemory>> {
    open_or_disabled(cache_path_or_default(
        std::env::var("RESPONSE_CACHE_PATH").ok(),
    ))
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

        assert_eq!(
            db.lookup_response(prompt).unwrap().as_deref(),
            Some("fresh")
        );
    }

    #[test]
    fn purge_expired_removes_only_stale_rows() {
        let tmp = NamedTempFile::new().unwrap();
        let db = SqliteMemory::open(tmp.path().to_str().unwrap()).unwrap();
        db.store_response("fresh", "f").unwrap();
        db.store_response("stale", "s").unwrap();

        let key = SqliteMemory::hash_prompt(&SqliteMemory::normalize_prompt("stale"));
        {
            let conn = db.conn.lock().expect("sqlite cache mutex poisoned");
            conn.execute(
                "UPDATE responses SET created_at = ?1 WHERE prompt_hash = ?2",
                params![now_unix_ms() - RESPONSE_CACHE_TTL_MS - 1, key],
            )
            .unwrap();
        }

        assert_eq!(db.purge_expired().unwrap(), 1);
        assert_eq!(db.lookup_response("fresh").unwrap().as_deref(), Some("f"));
        assert!(db.lookup_response("stale").unwrap().is_none());
    }

    #[test]
    fn clear_empties_cache() {
        let tmp = NamedTempFile::new().unwrap();
        let db = SqliteMemory::open(tmp.path().to_str().unwrap()).unwrap();
        db.store_response("a", "1").unwrap();
        db.store_response("b", "2").unwrap();

        db.clear().unwrap();

        assert!(db.lookup_response("a").unwrap().is_none());
        assert!(db.lookup_response("b").unwrap().is_none());
    }

    #[test]
    fn cache_path_uses_env_override() {
        let p = cache_path_or_default(Some("custom/dir/cache.sqlite3".to_string()));
        assert_eq!(p, PathBuf::from("custom/dir/cache.sqlite3"));
    }

    #[test]
    fn cache_path_defaults_to_temp_dir_when_unset_or_blank() {
        for v in [None, Some("   ".to_string())] {
            let p = cache_path_or_default(v);
            assert!(p.starts_with(std::env::temp_dir()));
            assert!(p.ends_with("memory-cache.sqlite3"));
        }
    }

    #[test]
    fn open_creates_missing_parent_dirs() {
        let dir = tempfile::tempdir().unwrap();
        let nested = dir.path().join("a").join("b").join("cache.sqlite3");
        let db = SqliteMemory::open(&nested).unwrap();
        db.store_response("k", "v").unwrap();
        assert_eq!(db.lookup_response("k").unwrap().as_deref(), Some("v"));
    }

    #[test]
    fn open_or_disabled_degrades_to_none_on_bad_path() {
        let file = NamedTempFile::new().unwrap();
        let bad = file.path().join("cache.sqlite3");
        assert!(open_or_disabled(&bad).is_none());
    }

    #[test]
    fn open_or_disabled_returns_some_on_good_path() {
        let dir = tempfile::tempdir().unwrap();
        assert!(open_or_disabled(dir.path().join("cache.sqlite3")).is_some());
    }
}
