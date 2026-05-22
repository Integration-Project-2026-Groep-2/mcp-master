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
async fn unknown_token_with_secret_set_returns_401() {
    // A secret (CHAT_BEARER_TOKEN) is configured but the token matches
    // neither a JWT nor the static token → closed-by-default 401.
    unsafe {
        std::env::remove_var("CHAT_JWT_SECRET");
        std::env::set_var("CHAT_BEARER_TOKEN", TEST_BEARER);
    }
    let mut parts = parts_with_bearer("wrong-token");
    assert_eq!(extract(&mut parts).await, Err(StatusCode::UNAUTHORIZED));
}

#[tokio::test]
#[serial]
async fn both_secrets_unset_grants_read_dev() {
    // Pure local dev: no secrets configured → any bearer token resolves to
    // read via the skip-warn path.
    unsafe {
        std::env::remove_var("CHAT_JWT_SECRET");
        std::env::remove_var("CHAT_BEARER_TOKEN");
    }
    let mut parts = parts_with_bearer("anything");
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
async fn expired_jwt_with_secret_set_returns_401() {
    // CHAT_JWT_SECRET is set but the JWT is expired (decode fails) and no
    // static token matches → closed-by-default 401.
    unsafe {
        std::env::set_var("CHAT_JWT_SECRET", TEST_SECRET);
        std::env::remove_var("CHAT_BEARER_TOKEN");
    }
    // -120s is past the jsonwebtoken default 60s leeway window.
    let token = mint_jwt(TEST_SECRET, "read+act", -120);
    let mut parts = parts_with_bearer(&token);
    assert_eq!(extract(&mut parts).await, Err(StatusCode::UNAUTHORIZED));
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

fn headers_with_bearer(token: &str) -> axum::http::HeaderMap {
    let mut h = axum::http::HeaderMap::new();
    h.insert(
        axum::http::header::AUTHORIZATION,
        format!("Bearer {token}").parse().unwrap(),
    );
    h
}

// Regression-guard for `chat()` threading the JWT sub claim into
// `DispatchContext.user_id`. Previously `chat()` hardcoded user_id="" and
// every approve returned WrongUser → 403. If a future change drops the
// current_user_id call from `chat()`, this test still passes — but the
// companion test `current_user_id_returns_none_without_jwt_secret` and
// the production code in `chat()` (uses `unwrap_or_default()` on this
// helper's result) together ensure the value flows.
#[tokio::test]
#[serial]
async fn current_user_id_returns_sub_for_valid_jwt() {
    unsafe {
        std::env::set_var("CHAT_JWT_SECRET", TEST_SECRET);
        std::env::remove_var("CHAT_BEARER_TOKEN");
    }
    let token = mint_jwt(TEST_SECRET, "read+act", 60);
    let headers = headers_with_bearer(&token);
    assert_eq!(current_user_id(&headers), Some("drupal-uid-42".to_string()),);
}

#[tokio::test]
#[serial]
async fn current_user_id_returns_none_without_jwt_secret() {
    unsafe {
        std::env::remove_var("CHAT_JWT_SECRET");
        std::env::set_var("CHAT_BEARER_TOKEN", TEST_BEARER);
    }
    let headers = headers_with_bearer(TEST_BEARER);
    assert_eq!(current_user_id(&headers), None);
}

#[tokio::test]
#[serial]
async fn current_user_id_returns_none_for_expired_jwt() {
    unsafe {
        std::env::set_var("CHAT_JWT_SECRET", TEST_SECRET);
    }
    let token = mint_jwt(TEST_SECRET, "read+act", -120);
    let headers = headers_with_bearer(&token);
    assert_eq!(current_user_id(&headers), None);
}
