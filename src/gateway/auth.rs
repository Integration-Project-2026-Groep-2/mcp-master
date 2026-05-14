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
use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
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

/// Re-decode the JWT sub claim from the request headers. Returns `None` for
/// the legacy bearer / skip-warn paths (no JWT present). Duplicate work with
/// the `AuthScope` extractor — R2.5 cleanup will thread `Claims` through as
/// a request extension. For PR-4 we keep it duplicate to minimise the
/// extractor's API surface.
pub fn current_user_id(headers: &axum::http::HeaderMap) -> Option<String> {
    let secret = jwt_secret_from_env()?;
    let header = headers.get(axum::http::header::AUTHORIZATION)?;
    let header_str = header.to_str().ok()?;
    let token = header_str.strip_prefix("Bearer ")?;
    decode_claims(token, &secret).ok().map(|c| c.sub)
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
mod tests;
