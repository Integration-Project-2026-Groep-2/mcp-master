use super::*;
use serial_test::serial;
use std::thread;

#[test]
fn first_event_allowed() {
    let d = Debouncer::new(Duration::from_secs(60));
    assert!(d.check("kassa").is_ok());
}

#[test]
fn second_event_within_window_denied() {
    let d = Debouncer::new(Duration::from_secs(60));
    d.check("kassa").unwrap();
    let r = d.check("kassa");
    assert!(r.is_err());
}

#[test]
fn different_services_are_independent() {
    let d = Debouncer::new(Duration::from_secs(60));
    assert!(d.check("kassa").is_ok());
    assert!(d.check("crm").is_ok());
    assert!(d.check("kassa").is_err());
    assert!(d.check("crm").is_err());
}

#[test]
fn after_window_allowed_again() {
    let d = Debouncer::new(Duration::from_millis(40));
    d.check("kassa").unwrap();
    thread::sleep(Duration::from_millis(60));
    assert!(d.check("kassa").is_ok());
}

#[test]
fn rapid_fire_ten_one_allowed_nine_denied() {
    let d = Debouncer::new(Duration::from_secs(60));
    let mut allowed = 0;
    let mut denied = 0;
    for _ in 0..10 {
        if d.check("kassa").is_ok() {
            allowed += 1;
        } else {
            denied += 1;
        }
    }
    assert_eq!(allowed, 1);
    assert_eq!(denied, 9);
}

#[test]
#[serial]
fn from_env_uses_default_when_unset() {
    unsafe {
        std::env::remove_var(ENV_VAR);
    }
    let d = Debouncer::from_env();
    assert_eq!(d.window(), Duration::from_secs(DEFAULT_WINDOW_SECONDS));
}

#[test]
#[serial]
fn from_env_parses_valid_value() {
    unsafe {
        std::env::set_var(ENV_VAR, "60");
    }
    let d = Debouncer::from_env();
    assert_eq!(d.window(), Duration::from_secs(60));
    unsafe {
        std::env::remove_var(ENV_VAR);
    }
}

#[test]
#[serial]
fn from_env_falls_back_on_garbage() {
    unsafe {
        std::env::set_var(ENV_VAR, "not-a-number");
    }
    let d = Debouncer::from_env();
    assert_eq!(d.window(), Duration::from_secs(DEFAULT_WINDOW_SECONDS));
    unsafe {
        std::env::remove_var(ENV_VAR);
    }
}

#[test]
#[serial]
fn from_env_falls_back_on_zero() {
    unsafe {
        std::env::set_var(ENV_VAR, "0");
    }
    let d = Debouncer::from_env();
    assert_eq!(d.window(), Duration::from_secs(DEFAULT_WINDOW_SECONDS));
    unsafe {
        std::env::remove_var(ENV_VAR);
    }
}
