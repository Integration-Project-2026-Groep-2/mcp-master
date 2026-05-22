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

impl Confidence {
    pub fn as_str(&self) -> &'static str {
        match self {
            Self::InsufficientEvidence => "insufficient_evidence",
            Self::Low => "low",
            Self::Medium => "medium",
            Self::High => "high",
        }
    }
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
mod tests;
