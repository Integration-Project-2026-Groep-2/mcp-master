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
