use std::sync::Mutex;
use std::time::{Duration, Instant};

const DEFAULT_MAX_PER_HOUR: u32 = 200;
const ENV_VAR: &str = "INCIDENT_MAX_PER_HOUR";
const WINDOW: Duration = Duration::from_secs(3600);

pub struct Budget {
    max_per_hour: u32,
    state: Mutex<BudgetState>,
}

struct BudgetState {
    accepted_in_window: u32,
    window_started_at: Instant,
    open_until: Option<Instant>,
}

#[derive(Debug, PartialEq, Eq)]
pub enum BudgetOutcome {
    Allowed,
    /// Circuit is open until `reset_at`. Caller publishes a circuit-open
    /// event and skips the diagnosis pipeline for the remaining duration.
    CircuitOpen {
        reset_at: Instant,
    },
}

impl Budget {
    pub fn from_env() -> Self {
        let max = std::env::var(ENV_VAR)
            .ok()
            .and_then(|raw| raw.trim().parse::<u32>().ok())
            .filter(|&n| n > 0)
            .unwrap_or_else(|| {
                tracing::info!(
                    "{ENV_VAR} unset/invalid — using default {DEFAULT_MAX_PER_HOUR}/hour"
                );
                DEFAULT_MAX_PER_HOUR
            });
        Self::new(max)
    }

    pub fn new(max_per_hour: u32) -> Self {
        Self {
            max_per_hour,
            state: Mutex::new(BudgetState {
                accepted_in_window: 0,
                window_started_at: Instant::now(),
                open_until: None,
            }),
        }
    }

    pub fn max_per_hour(&self) -> u32 {
        self.max_per_hour
    }

    /// Atomic check-and-increment. Returns `Allowed` and counts the event,
    /// or `CircuitOpen { reset_at }` when the budget is exhausted. Auto-resets
    /// after the rolling 1h window expires.
    pub fn try_consume(&self) -> BudgetOutcome {
        let now = Instant::now();
        let mut state = match self.state.lock() {
            Ok(s) => s,
            Err(poison) => poison.into_inner(),
        };

        if let Some(until) = state.open_until {
            if now < until {
                return BudgetOutcome::CircuitOpen { reset_at: until };
            }
            state.open_until = None;
            state.window_started_at = now;
            state.accepted_in_window = 0;
        }

        if now.duration_since(state.window_started_at) >= WINDOW {
            state.window_started_at = now;
            state.accepted_in_window = 0;
        }

        if state.accepted_in_window >= self.max_per_hour {
            let reset_at = now + WINDOW;
            state.open_until = Some(reset_at);
            return BudgetOutcome::CircuitOpen { reset_at };
        }

        state.accepted_in_window += 1;
        BudgetOutcome::Allowed
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use serial_test::serial;

    #[test]
    fn first_event_allowed() {
        let b = Budget::new(10);
        assert_eq!(b.try_consume(), BudgetOutcome::Allowed);
    }

    #[test]
    fn allows_up_to_max_then_opens_circuit() {
        let b = Budget::new(3);
        assert_eq!(b.try_consume(), BudgetOutcome::Allowed);
        assert_eq!(b.try_consume(), BudgetOutcome::Allowed);
        assert_eq!(b.try_consume(), BudgetOutcome::Allowed);
        match b.try_consume() {
            BudgetOutcome::CircuitOpen { .. } => {}
            other => panic!("expected CircuitOpen, got {other:?}"),
        }
    }

    #[test]
    fn circuit_stays_open_for_subsequent_calls() {
        let b = Budget::new(1);
        let _ = b.try_consume();
        let first_open = b.try_consume();
        let second_open = b.try_consume();
        match (first_open, second_open) {
            (BudgetOutcome::CircuitOpen { .. }, BudgetOutcome::CircuitOpen { .. }) => {}
            other => panic!("expected both CircuitOpen, got {other:?}"),
        }
    }

    #[test]
    fn reset_at_is_in_the_future() {
        let b = Budget::new(0);
        let now = Instant::now();
        match b.try_consume() {
            BudgetOutcome::CircuitOpen { reset_at } => {
                assert!(reset_at > now);
                assert!(reset_at <= now + WINDOW + Duration::from_secs(1));
            }
            other => panic!("expected CircuitOpen, got {other:?}"),
        }
    }

    #[test]
    #[serial]
    fn from_env_uses_default_when_unset() {
        unsafe {
            std::env::remove_var(ENV_VAR);
        }
        let b = Budget::from_env();
        assert_eq!(b.max_per_hour(), DEFAULT_MAX_PER_HOUR);
    }

    #[test]
    #[serial]
    fn from_env_parses_valid_value() {
        unsafe {
            std::env::set_var(ENV_VAR, "50");
        }
        let b = Budget::from_env();
        assert_eq!(b.max_per_hour(), 50);
        unsafe {
            std::env::remove_var(ENV_VAR);
        }
    }

    #[test]
    #[serial]
    fn from_env_falls_back_on_zero_or_garbage() {
        unsafe {
            std::env::set_var(ENV_VAR, "0");
        }
        let b = Budget::from_env();
        assert_eq!(b.max_per_hour(), DEFAULT_MAX_PER_HOUR);
        unsafe {
            std::env::set_var(ENV_VAR, "not-a-number");
        }
        let b = Budget::from_env();
        assert_eq!(b.max_per_hour(), DEFAULT_MAX_PER_HOUR);
        unsafe {
            std::env::remove_var(ENV_VAR);
        }
    }
}
