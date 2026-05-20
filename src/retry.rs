//! Exponential backoff with jitter — shared across reconnect/retry paths
//! (Anthropic API, AMQP consumer/publisher, MCP-client sessions).
//!
//! All paths use the same exponential schedule + jitter so retry-storms
//! during a shared incident (e.g. RabbitMQ broker restart) don't synchronise
//! across components within one process.

use std::time::Duration;

/// Cap on the exponent so attempts ≥ 7 don't shift past 32s. Attempt N
/// shifts by `N - 1` bits, so MAX_EXPONENT=5 caps the base at `1000 << 5 = 32_000`.
/// Without this, `1000_u64 << 30` would overflow well before reaching `attempt = 64`.
const MAX_EXPONENT: u32 = 5;

/// Jitter window in milliseconds, added to the deterministic exponential
/// base. 500 ms is enough to de-correlate ~10 instances retrying a shared
/// dependency; a larger window is needed only at much higher fan-out.
const JITTER_MS: u64 = 500;

/// Compute the sleep before retry attempt `attempt` (1-based).
///
/// Schedule: ~1s, 2s, 4s, 8s, 16s, 32s for attempts 1..=6, then capped at
/// 32s + jitter for any further attempt. Jitter is uniform 0..500 ms.
///
/// Callers decide their own attempt-cap or "give up" criterion — this
/// function never returns an error and never panics on extreme inputs.
pub fn backoff_with_jitter(attempt: u32) -> Duration {
    debug_assert!(
        attempt >= 1,
        "attempt must be 1-based when computing backoff"
    );
    let exponent = attempt.saturating_sub(1).min(MAX_EXPONENT);
    let base_ms: u64 = 1000_u64 << exponent;
    let jitter_ms: u64 = rand::random::<u64>() % JITTER_MS;
    Duration::from_millis(base_ms + jitter_ms)
}

#[cfg(test)]
mod tests;
