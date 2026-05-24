use super::*;

fn ts(s: &str) -> DateTime<Utc> {
    DateTime::parse_from_rfc3339(s).unwrap().with_timezone(&Utc)
}

#[test]
fn parse_heartbeat_extracts_lowercased_service_and_timestamp() {
    let now = ts("2026-05-24T13:00:00Z");
    let body = b"<heartbeat><serviceId>CRM</serviceId><timestamp>2026-05-24T12:59:30Z</timestamp></heartbeat>";

    let (service, last) = parse_heartbeat(body, now).unwrap();

    assert_eq!(service, "crm");
    assert_eq!(last, ts("2026-05-24T12:59:30Z"));
}

#[test]
fn parse_heartbeat_falls_back_to_now_when_timestamp_absent() {
    let now = ts("2026-05-24T13:00:00Z");
    let body = b"<heartbeat><serviceId>kassa</serviceId></heartbeat>";

    let (service, last) = parse_heartbeat(body, now).unwrap();

    assert_eq!(service, "kassa");
    assert_eq!(last, now);
}

#[test]
fn parse_heartbeat_falls_back_to_now_when_timestamp_empty() {
    let now = ts("2026-05-24T13:00:00Z");
    let body = b"<heartbeat><serviceId>crm</serviceId><timestamp></timestamp></heartbeat>";

    let (service, last) = parse_heartbeat(body, now).unwrap();

    assert_eq!(service, "crm");
    assert_eq!(last, now);
}

#[test]
fn parse_heartbeat_falls_back_to_now_when_timestamp_lacks_timezone() {
    let now = ts("2026-05-24T13:00:00Z");
    let body = b"<heartbeat><serviceId>crm</serviceId><timestamp>2026-05-24T12:59:30</timestamp></heartbeat>";

    let (service, last) = parse_heartbeat(body, now).unwrap();

    assert_eq!(service, "crm");
    assert_eq!(last, now);
}

#[test]
fn parse_heartbeat_accepts_non_utc_offset_and_converts() {
    let now = ts("2026-05-24T13:00:00Z");
    let body = b"<heartbeat><serviceId>crm</serviceId><timestamp>2026-05-24T14:59:30+02:00</timestamp></heartbeat>";

    let (service, last) = parse_heartbeat(body, now).unwrap();

    assert_eq!(service, "crm");
    assert_eq!(last, ts("2026-05-24T12:59:30Z"));
}

#[test]
fn parse_heartbeat_rejects_blank_service() {
    let now = ts("2026-05-24T13:00:00Z");
    let body = b"<heartbeat><serviceId>   </serviceId></heartbeat>";

    let err = parse_heartbeat(body, now).unwrap_err();

    assert!(err.to_string().contains("serviceId"));
}

#[test]
fn snapshot_marks_fresh_up_stale_down_sorted_by_name() {
    let now = ts("2026-05-24T13:00:00Z");
    let state = HeartbeatState::new();
    state.insert("kassa".into(), ts("2026-05-24T12:59:30Z"));
    state.insert("crm".into(), ts("2026-05-24T12:57:00Z"));

    let snap = snapshot(&state, now);

    assert_eq!(snap.len(), 2);
    assert_eq!(snap[0].name, "crm");
    assert_eq!(snap[0].status, "down");
    assert_eq!(snap[0].age_seconds, 180);
    assert_eq!(snap[1].name, "kassa");
    assert_eq!(snap[1].status, "up");
    assert_eq!(snap[1].age_seconds, 30);
}
