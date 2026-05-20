use super::*;

#[test]
fn host_for_logging_strips_userinfo() {
    let cfg = RabbitMqConfig {
        url: "amqp://lapin:supersecret@rabbitmq_management:5672//".to_string(),
        exchange: "ai.events".to_string(),
    };
    let s = cfg.host_for_logging();
    assert!(!s.contains("supersecret"), "must not leak password: {s}");
    assert!(!s.contains("lapin"), "must not leak username: {s}");
    assert!(
        s.contains("rabbitmq_management:5672"),
        "must keep host:port: {s}"
    );
    assert!(s.starts_with("amqp://"), "must keep scheme: {s}");
}

#[test]
fn host_for_logging_handles_no_port() {
    let cfg = RabbitMqConfig {
        url: "amqp://guest:guest@localhost//".to_string(),
        exchange: "x".to_string(),
    };
    let s = cfg.host_for_logging();
    assert!(!s.contains("guest"), "must not leak guest credentials: {s}");
    assert!(s.contains("localhost"));
}

#[test]
fn host_for_logging_invalid_url_returns_placeholder() {
    let cfg = RabbitMqConfig {
        url: "not a url".to_string(),
        exchange: "x".to_string(),
    };
    assert_eq!(cfg.host_for_logging(), "<invalid URL>");
}

#[test]
fn normalize_single_slash_to_percent_encoded() {
    assert_eq!(
        normalize_amqp_url("amqp://lapin:pw@rabbitmq:5672/"),
        "amqp://lapin:pw@rabbitmq:5672/%2F"
    );
}

#[test]
fn normalize_double_slash_unchanged() {
    let url = "amqp://lapin:pw@rabbitmq:5672//";
    assert_eq!(normalize_amqp_url(url), url);
}

#[test]
fn normalize_percent_encoded_unchanged() {
    let url = "amqp://lapin:pw@rabbitmq:5672/%2F";
    assert_eq!(normalize_amqp_url(url), url);
    let lowercase = "amqp://lapin:pw@rabbitmq:5672/%2f";
    assert_eq!(normalize_amqp_url(lowercase), lowercase);
}

#[test]
fn normalize_no_path_unchanged() {
    let url = "amqp://lapin:pw@rabbitmq:5672";
    assert_eq!(normalize_amqp_url(url), url);
}

#[test]
fn normalize_named_vhost_unchanged() {
    let url = "amqp://lapin:pw@rabbitmq:5672/myhost";
    assert_eq!(normalize_amqp_url(url), url);
}

#[test]
fn normalize_amqps_single_slash() {
    assert_eq!(
        normalize_amqp_url("amqps://lapin:pw@rabbitmq:5671/"),
        "amqps://lapin:pw@rabbitmq:5671/%2F"
    );
}

#[test]
fn normalize_trims_whitespace() {
    assert_eq!(
        normalize_amqp_url("  amqp://lapin:pw@rabbitmq:5672/  "),
        "amqp://lapin:pw@rabbitmq:5672/%2F"
    );
}

#[test]
fn normalize_no_scheme_unchanged() {
    let url = "not a url";
    assert_eq!(normalize_amqp_url(url), url);
}
