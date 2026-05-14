use std::time::{Duration, Instant};

use dashmap::DashMap;

const DEFAULT_WINDOW_SECONDS: u64 = 300;
const ENV_VAR: &str = "INCIDENT_DEBOUNCE_SECONDS";

pub struct Debouncer {
    window: Duration,
    last_fired: DashMap<String, Instant>,
}

impl Debouncer {
    pub fn from_env() -> Self {
        let secs = std::env::var(ENV_VAR)
            .ok()
            .and_then(|raw| raw.trim().parse::<u64>().ok())
            .filter(|&n| n > 0)
            .unwrap_or_else(|| {
                tracing::info!("{ENV_VAR} unset/invalid — using default {DEFAULT_WINDOW_SECONDS}s");
                DEFAULT_WINDOW_SECONDS
            });
        Self::new(Duration::from_secs(secs))
    }

    pub fn new(window: Duration) -> Self {
        Self {
            window,
            last_fired: DashMap::new(),
        }
    }

    pub fn window(&self) -> Duration {
        self.window
    }

    /// Atomic compare-and-swap on the last-fired timestamp. Returns `Ok(())`
    /// when the event is fresh enough to allow (and updates the slot), or
    /// `Err(elapsed)` when within-window — caller publishes a skip-event with
    /// the elapsed duration as audit-evidence.
    pub fn check(&self, service: &str) -> Result<(), Duration> {
        use dashmap::mapref::entry::Entry;
        let now = Instant::now();
        match self.last_fired.entry(service.to_string()) {
            Entry::Occupied(mut o) => {
                let elapsed = now.duration_since(*o.get());
                if elapsed < self.window {
                    Err(elapsed)
                } else {
                    o.insert(now);
                    Ok(())
                }
            }
            Entry::Vacant(v) => {
                v.insert(now);
                Ok(())
            }
        }
    }
}

#[cfg(test)]
mod tests;
