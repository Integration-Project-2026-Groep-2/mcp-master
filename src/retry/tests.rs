use super::*;

#[test]
fn backoff_with_jitter_grows_exponentially() {
    // For each attempt, run multiple samples and assert all fall in the
    // expected [base, base+JITTER) window. Probabilistic but tight —
    // a regression to wrong-base would fail every sample.
    let cases: &[(u32, u64)] = &[
        (1, 1000),
        (2, 2000),
        (3, 4000),
        (4, 8000),
        (5, 16000),
        (6, 32000),
    ];
    for &(attempt, base_ms) in cases {
        for _ in 0..50 {
            let d = backoff_with_jitter(attempt).as_millis() as u64;
            assert!(
                d >= base_ms && d < base_ms + JITTER_MS,
                "attempt={attempt} expected [{base_ms},{}), got {d}",
                base_ms + JITTER_MS,
            );
        }
    }
}

#[test]
fn backoff_with_jitter_caps_at_attempt_6() {
    // Attempts 7, 8, 9, 100 should all stay in the same window as 6 —
    // no overflow, no wraparound.
    for attempt in [7_u32, 8, 9, 100, 1000, u32::MAX] {
        for _ in 0..20 {
            let d = backoff_with_jitter(attempt).as_millis() as u64;
            assert!(
                (32_000..32_000 + JITTER_MS).contains(&d),
                "attempt={attempt} should cap at attempt-6 window, got {d}"
            );
        }
    }
}

#[test]
fn backoff_with_jitter_includes_jitter() {
    // Run 100 samples for a fixed attempt and assert we see >= 10 distinct
    // values. With JITTER_MS=500 and uniform random, P(<10 distinct in 100)
    // is astronomically small — this catches a "jitter accidentally zero"
    // regression without being flaky.
    let mut seen: std::collections::HashSet<u64> = std::collections::HashSet::new();
    for _ in 0..100 {
        seen.insert(backoff_with_jitter(3).as_millis() as u64);
    }
    assert!(
        seen.len() >= 10,
        "expected jitter to produce ≥10 distinct delays in 100 samples, got {}",
        seen.len()
    );
}
