use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};

#[derive(Debug, Serialize, Deserialize, Clone, PartialEq, Eq)]
pub struct IncidentDiagnosis {
    pub root_cause: String,
    pub critical_failure: String,
    pub impact: String,
    pub confidence: Confidence,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub suggested_action: Option<String>,
    pub evidence_summary: String,
}

#[derive(Debug, Serialize, Deserialize, Clone, Copy, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum Confidence {
    InsufficientEvidence,
    Low,
    Medium,
    High,
}

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

    #[test]
    fn confidence_snake_case_wire_format() {
        assert_eq!(
            serde_json::to_string(&Confidence::InsufficientEvidence).unwrap(),
            "\"insufficient_evidence\""
        );
        assert_eq!(
            serde_json::to_string(&Confidence::High).unwrap(),
            "\"high\""
        );
    }

    #[test]
    fn incident_diagnosis_round_trips_with_suggested_action() {
        let d = IncidentDiagnosis {
            root_cause: "deploy abc123 broke DB pool sizing".into(),
            critical_failure: "connection pool exhausted".into(),
            impact: "checkout flow blocked".into(),
            confidence: Confidence::High,
            suggested_action: Some("rollback to deadbeef".into()),
            evidence_summary: "47 timeouts after 14:18 deploy".into(),
        };
        let json = serde_json::to_string(&d).unwrap();
        let parsed: IncidentDiagnosis = serde_json::from_str(&json).unwrap();
        assert_eq!(d, parsed);
    }

    #[test]
    fn incident_diagnosis_omits_suggested_action_when_none() {
        let d = IncidentDiagnosis {
            root_cause: "x".into(),
            critical_failure: "x".into(),
            impact: "x".into(),
            confidence: Confidence::Low,
            suggested_action: None,
            evidence_summary: "x".into(),
        };
        let json = serde_json::to_string(&d).unwrap();
        assert!(!json.contains("suggested_action"));
    }

    #[test]
    fn incident_diagnosis_rejects_unknown_confidence() {
        let bad = r#"{
            "root_cause": "x",
            "critical_failure": "x",
            "impact": "x",
            "confidence": "uncertain",
            "evidence_summary": "x"
        }"#;
        let r: Result<IncidentDiagnosis, _> = serde_json::from_str(bad);
        assert!(r.is_err());
    }
}
