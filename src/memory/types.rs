use serde::{Deserialize, Serialize};
use std::fmt;
use std::time::{SystemTime, UNIX_EPOCH};

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum MemorySource {
    Chat,
    ScheduledSummary,
    IncidentEvidence,
    IncidentDiagnosis,
}

impl MemorySource {
    pub fn as_str(self) -> &'static str {
        match self {
            MemorySource::Chat => "chat",
            MemorySource::ScheduledSummary => "scheduled_summary",
            MemorySource::IncidentEvidence => "incident_evidence",
            MemorySource::IncidentDiagnosis => "incident_diagnosis",
        }
    }
}

impl fmt::Display for MemorySource {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(self.as_str())
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MemoryInteraction {
    pub namespace: String,
    pub source: MemorySource,
    pub correlation_id: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub user_id: Option<String>,
    pub prompt: String,
    pub answer: String,
    pub created_at_unix_ms: i64,
}

impl MemoryInteraction {
    pub fn new(
        namespace: impl Into<String>,
        source: MemorySource,
        correlation_id: impl Into<String>,
        user_id: Option<impl Into<String>>,
        prompt: impl Into<String>,
        answer: impl Into<String>,
    ) -> Self {
        Self {
            namespace: namespace.into(),
            source,
            correlation_id: correlation_id.into(),
            user_id: user_id.map(Into::into),
            prompt: prompt.into(),
            answer: answer.into(),
            created_at_unix_ms: now_unix_ms(),
        }
    }
}

#[allow(dead_code)]
#[derive(Debug, Clone)]
pub struct MemoryHit {
    pub score: f32,
    pub namespace: String,
    pub source: MemorySource,
    pub correlation_id: String,
    pub user_id: Option<String>,
    pub text: String,
    pub chunk_index: u32,
    pub chunk_count: u32,
    pub created_at_unix_ms: i64,
}

fn now_unix_ms() -> i64 {
    let duration = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default();
    duration.as_millis() as i64
}
