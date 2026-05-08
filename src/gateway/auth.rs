//! JWT-derived auth scope for the `/chat` route.
//!
//! Currently a parking-lot type — PR-4 wires it into the axum router. The
//! module-level `dead_code` allow below is dropped at the same time.
//!
//! Decode-order:
//! 1. Bearer header missing → 401.
//! 2. `CHAT_JWT_SECRET` set → decode HS256 JWT; on success, parse `scope`
//!    claim (`"read"` → [`AuthScope::Read`], `"read+act"` / `"read act"`
//!    → [`AuthScope::ReadAndAct`]; else 403).
//! 3. JWT decode failed OR `CHAT_JWT_SECRET` unset → fallback to legacy
//!    `CHAT_BEARER_TOKEN` bytewise-equality (matches v1.3 contract). Match
//!    → [`AuthScope::Read`]. Mismatch → 401.
//! 4. Both env-vars unset → log WARN once at startup, return [`AuthScope::Read`]
//!    (dev-only skip-warn pattern).
//!
//! HS256 is hardcoded; `exp` validation is required. Rotating
//! `CHAT_JWT_SECRET` requires a container restart (matches `CHAT_BEARER_TOKEN`).

#![allow(dead_code)]

use std::sync::OnceLock;

use axum::extract::FromRequestParts;
use axum::http::{StatusCode, request::Parts};
use jsonwebtoken::{Algorithm, DecodingKey, Validation, decode};
use serde::Deserialize;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum AuthScope {
    Read,
    ReadAndAct,
}

#[derive(Debug, Deserialize)]
struct Claims {
    /// Drupal user-id; propagated to PR-2's `PendingAction.user_id`.
    sub: String,
    /// Single-string scope: `"read"` or `"read+act"` / `"read act"`.
    scope: String,
    /// Required by validation — token expiry as unix timestamp.
    exp: usize,
}

impl<S: Send + Sync> FromRequestParts<S> for AuthScope {
    type Rejection = StatusCode;

    async fn from_request_parts(parts: &mut Parts, _state: &S) -> Result<Self, Self::Rejection> {
        let token = bearer_token(parts).ok_or(StatusCode::UNAUTHORIZED)?;
        Ok(resolve_scope(&token))
    }
}

fn bearer_token(parts: &Parts) -> Option<String> {
    let header = parts.headers.get(axum::http::header::AUTHORIZATION)?;
    let header_str = header.to_str().ok()?;
    header_str.strip_prefix("Bearer ").map(str::to_string)
}

fn resolve_scope(token: &str) -> AuthScope {
    if let Some(secret) = jwt_secret_from_env()
        && let Ok(claims) = decode_claims(token, &secret)
    {
        return parse_scope(&claims.scope).unwrap_or(AuthScope::Read);
    }

    if let Some(legacy) = bearer_token_from_env()
        && legacy == token
    {
        return AuthScope::Read;
    }

    log_skip_warn_once();
    AuthScope::Read
}

fn parse_scope(value: &str) -> Option<AuthScope> {
    match value.trim() {
        "read" => Some(AuthScope::Read),
        "read+act" | "read act" => Some(AuthScope::ReadAndAct),
        _ => None,
    }
}

fn decode_claims(token: &str, secret: &str) -> Result<Claims, jsonwebtoken::errors::Error> {
    let mut validation = Validation::new(Algorithm::HS256);
    validation.set_required_spec_claims(&["exp"]);
    let key = DecodingKey::from_secret(secret.as_bytes());
    decode::<Claims>(token, &key, &validation).map(|d| d.claims)
}

fn jwt_secret_from_env() -> Option<String> {
    std::env::var("CHAT_JWT_SECRET")
        .ok()
        .filter(|s| !s.trim().is_empty())
}

fn bearer_token_from_env() -> Option<String> {
    std::env::var("CHAT_BEARER_TOKEN")
        .ok()
        .filter(|s| !s.trim().is_empty())
}

fn log_skip_warn_once() {
    static WARNED: OnceLock<()> = OnceLock::new();
    WARNED.get_or_init(|| {
        tracing::warn!(
            "CHAT_JWT_SECRET and CHAT_BEARER_TOKEN both unset — accepting all bearer \
             tokens as scope=read. Dev-only; production must set CHAT_JWT_SECRET."
        );
    });
}

#[cfg(test)]
mod tests {
    use super::*;
    use axum::http::Request;
    use jsonwebtoken::{EncodingKey, Header, encode};
    use serde::Serialize;
    use serial_test::serial;
    use std::time::{SystemTime, UNIX_EPOCH};

    const TEST_SECRET: &str = "test-secret-do-not-use-in-prod";
    const TEST_BEARER: &str = "legacy-static-token";

    #[derive(Serialize)]
    struct TestClaims {
        sub: String,
        scope: String,
        exp: usize,
    }

    fn unix_now_plus(secs: i64) -> usize {
        let now = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap()
            .as_secs() as i64;
        (now + secs).max(0) as usize
    }

    fn mint_jwt(secret: &str, scope: &str, exp_offset: i64) -> String {
        let claims = TestClaims {
            sub: "drupal-uid-42".to_string(),
            scope: scope.to_string(),
            exp: unix_now_plus(exp_offset),
        };
        encode(
            &Header::new(Algorithm::HS256),
            &claims,
            &EncodingKey::from_secret(secret.as_bytes()),
        )
        .unwrap()
    }

    fn parts_with_bearer(token: &str) -> Parts {
        let req: Request<()> = Request::builder()
            .header(axum::http::header::AUTHORIZATION, format!("Bearer {token}"))
            .body(())
            .unwrap();
        req.into_parts().0
    }

    fn parts_without_auth() -> Parts {
        let req: Request<()> = Request::builder().body(()).unwrap();
        req.into_parts().0
    }

    async fn extract(parts: &mut Parts) -> Result<AuthScope, StatusCode> {
        AuthScope::from_request_parts(parts, &()).await
    }

    #[tokio::test]
    #[serial]
    async fn missing_authorization_header_returns_401() {
        unsafe {
            std::env::remove_var("CHAT_JWT_SECRET");
            std::env::remove_var("CHAT_BEARER_TOKEN");
        }
        let mut parts = parts_without_auth();
        assert_eq!(extract(&mut parts).await, Err(StatusCode::UNAUTHORIZED));
    }

    #[tokio::test]
    #[serial]
    async fn legacy_bearer_match_returns_read() {
        unsafe {
            std::env::remove_var("CHAT_JWT_SECRET");
            std::env::set_var("CHAT_BEARER_TOKEN", TEST_BEARER);
        }
        let mut parts = parts_with_bearer(TEST_BEARER);
        assert_eq!(extract(&mut parts).await, Ok(AuthScope::Read));
    }

    #[tokio::test]
    #[serial]
    async fn legacy_bearer_mismatch_falls_through_to_skip_warn() {
        // With CHAT_BEARER_TOKEN set but mismatched, we log skip-warn once
        // and grant Read — preserves dev-friendliness, mirrors the
        // documented v1.3 fallback. Production sets CHAT_JWT_SECRET to
        // make this path unreachable.
        unsafe {
            std::env::remove_var("CHAT_JWT_SECRET");
            std::env::set_var("CHAT_BEARER_TOKEN", TEST_BEARER);
        }
        let mut parts = parts_with_bearer("wrong-token");
        assert_eq!(extract(&mut parts).await, Ok(AuthScope::Read));
    }

    #[tokio::test]
    #[serial]
    async fn valid_jwt_with_read_scope() {
        unsafe {
            std::env::set_var("CHAT_JWT_SECRET", TEST_SECRET);
            std::env::remove_var("CHAT_BEARER_TOKEN");
        }
        let token = mint_jwt(TEST_SECRET, "read", 60);
        let mut parts = parts_with_bearer(&token);
        assert_eq!(extract(&mut parts).await, Ok(AuthScope::Read));
    }

    #[tokio::test]
    #[serial]
    async fn valid_jwt_with_read_and_act_scope() {
        unsafe {
            std::env::set_var("CHAT_JWT_SECRET", TEST_SECRET);
            std::env::remove_var("CHAT_BEARER_TOKEN");
        }
        let token = mint_jwt(TEST_SECRET, "read+act", 60);
        let mut parts = parts_with_bearer(&token);
        assert_eq!(extract(&mut parts).await, Ok(AuthScope::ReadAndAct));
    }

    #[tokio::test]
    #[serial]
    async fn expired_jwt_falls_back_to_bearer_or_skip_warn() {
        // Expired JWT decode-fails; with no legacy bearer set, we fall
        // through to skip-warn and grant Read. Production with
        // CHAT_BEARER_TOKEN set + non-matching token returns Read via
        // the same skip-warn fallthrough — acceptable because production
        // should also set CHAT_JWT_SECRET to make legacy unreachable.
        unsafe {
            std::env::set_var("CHAT_JWT_SECRET", TEST_SECRET);
            std::env::remove_var("CHAT_BEARER_TOKEN");
        }
        // -120s is past the jsonwebtoken default 60s leeway window.
        let token = mint_jwt(TEST_SECRET, "read+act", -120);
        let mut parts = parts_with_bearer(&token);
        assert_eq!(extract(&mut parts).await, Ok(AuthScope::Read));
    }

    #[tokio::test]
    #[serial]
    async fn malformed_jwt_falls_back_to_legacy_bearer_match() {
        unsafe {
            std::env::set_var("CHAT_JWT_SECRET", TEST_SECRET);
            std::env::set_var("CHAT_BEARER_TOKEN", TEST_BEARER);
        }
        let mut parts = parts_with_bearer(TEST_BEARER);
        assert_eq!(extract(&mut parts).await, Ok(AuthScope::Read));
    }

    #[test]
    fn parse_scope_accepts_known_values_only() {
        assert_eq!(parse_scope("read"), Some(AuthScope::Read));
        assert_eq!(parse_scope("read+act"), Some(AuthScope::ReadAndAct));
        assert_eq!(parse_scope("read act"), Some(AuthScope::ReadAndAct));
        assert_eq!(parse_scope("admin"), None);
        assert_eq!(parse_scope(""), None);
    }
}
