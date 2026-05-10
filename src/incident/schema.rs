use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};

#[derive(Debug, Deserialize, PartialEq)]
pub struct IncidentEvent {
    pub event: String,
    pub source: String,
    pub timestamp: DateTime<Utc>,
    pub payload: IncidentPayload,
}

#[derive(Debug, Deserialize, PartialEq)]
pub struct IncidentPayload {
    pub summary: String,
    pub severity: Severity,
    pub component: String,
    #[serde(default)]
    pub group: Option<String>,
    #[serde(default)]
    pub class: Option<String>,
    #[serde(default)]
    pub custom_details: serde_json::Value,
}

#[derive(Debug, Deserialize, Serialize, PartialEq, Eq, Clone, Copy)]
#[serde(rename_all = "lowercase")]
pub enum Severity {
    Info,
    Warning,
    Error,
    Critical,
}

impl Severity {
    // Wired into severity-filter in P2 (debounce + skip publish).
    #[allow(dead_code)]
    pub fn is_actionable(self) -> bool {
        matches!(self, Severity::Critical | Severity::Error)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const VALID_PAYLOAD: &str = r#"{
        "event": "heartbeat_failed",
        "source": "controlroom-watchdog",
        "timestamp": "2026-05-10T14:23:17Z",
        "payload": {
            "summary": "kassa heartbeat missed (count 0 in last 60s)",
            "severity": "critical",
            "component": "kassa",
            "group": "festival-services",
            "class": "heartbeat-loss",
            "custom_details": {
                "heartbeat_count_last_60s": 0,
                "threshold": 30
            }
        }
    }"#;

    #[test]
    fn happy_path_pd_cef_envelope() {
        let evt: IncidentEvent = serde_json::from_str(VALID_PAYLOAD).expect("parses");
        assert_eq!(evt.event, "heartbeat_failed");
        assert_eq!(evt.source, "controlroom-watchdog");
        assert_eq!(evt.payload.severity, Severity::Critical);
        assert_eq!(evt.payload.component, "kassa");
        assert_eq!(evt.payload.group.as_deref(), Some("festival-services"));
        assert_eq!(evt.payload.class.as_deref(), Some("heartbeat-loss"));
    }

    #[test]
    fn missing_optional_fields_default_to_none() {
        let json = r#"{
            "event": "heartbeat_failed",
            "source": "controlroom-watchdog",
            "timestamp": "2026-05-10T14:23:17Z",
            "payload": {
                "summary": "x down",
                "severity": "critical",
                "component": "x"
            }
        }"#;
        let evt: IncidentEvent = serde_json::from_str(json).expect("parses without optionals");
        assert!(evt.payload.group.is_none());
        assert!(evt.payload.class.is_none());
        assert!(evt.payload.custom_details.is_null());
    }

    #[test]
    fn unknown_severity_rejected() {
        let bad = VALID_PAYLOAD.replace("\"critical\"", "\"meltdown\"");
        let r: Result<IncidentEvent, _> = serde_json::from_str(&bad);
        assert!(r.is_err(), "unknown severity must fail deserialization");
    }

    #[test]
    fn missing_required_field_rejected() {
        let json = r#"{
            "event": "heartbeat_failed",
            "source": "controlroom-watchdog",
            "timestamp": "2026-05-10T14:23:17Z",
            "payload": {
                "severity": "critical",
                "component": "x"
            }
        }"#;
        let r: Result<IncidentEvent, _> = serde_json::from_str(json);
        assert!(r.is_err(), "missing summary must fail");
    }

    #[test]
    fn severity_is_actionable_filters_low_signal() {
        assert!(Severity::Critical.is_actionable());
        assert!(Severity::Error.is_actionable());
        assert!(!Severity::Warning.is_actionable());
        assert!(!Severity::Info.is_actionable());
    }

    #[test]
    fn severity_lowercase_wire_format() {
        let s = serde_json::to_string(&Severity::Critical).unwrap();
        assert_eq!(s, "\"critical\"");
    }
}
