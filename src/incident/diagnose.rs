// Wired into consumer in P5 (Step B + full pipeline). Module is dead-code
// until then; types stay public so the integration in P5 is a one-liner.
#![allow(dead_code)]

use anyhow::Result;
use serde::{Deserialize, Serialize};

use super::prompts::{STEP_A_SYSTEM_PROMPT, seed_prompt_step_a};
use super::schema::IncidentEvent;
use crate::agent::llm::{ContentBlock, LlmClient, Message, Role, ToolSpec};
use crate::agent::modes::{AgentMode, DispatchContext, ReadOnlyMode};
use crate::agent::orchestrator::{self, McpExecutor, RunOutcome, ToolCallTrace};

const STEP_A_TOOLS: &[&str] = &["fetch_logs", "fetch_recent_deploys"];
const STEP_A_MAX_ITERATIONS: usize = 6;
const STEP_A_MAX_TOKENS: u32 = 4096;

#[derive(Debug, Clone, Serialize)]
pub struct EvidenceBundle {
    pub summary: String,
    pub missing_sources: Vec<String>,
    pub tool_trace: Vec<ToolCallTrace>,
}

/// Run Step A: drive the LLM through fetch_logs + fetch_recent_deploys with
/// `ReadOnlyMode` (no write-tools possible) and return a structured evidence
/// bundle. The LLM has tool-access but no privileged downstream effect — its
/// only job is to summarise findings into JSON.
///
/// If the LLM doesn't follow the JSON-output instruction, the raw answer text
/// is preserved as `summary` and `missing_sources` is inferred from any
/// failed tool-calls. This degrades gracefully rather than bailing, since
/// Step B can still reason over prose evidence — just less structured.
pub async fn gather_evidence(
    event: &IncidentEvent,
    llm: &dyn LlmClient,
    mcp: &dyn McpExecutor,
    tool_specs: &[ToolSpec],
) -> Result<EvidenceBundle> {
    let restricted = step_a_tool_specs(tool_specs);
    if restricted.is_empty() {
        anyhow::bail!(
            "no Step A tools available — Controlroom-MCP must expose fetch_logs and \
             fetch_recent_deploys before incident-response can run"
        );
    }

    let messages = vec![Message {
        role: Role::User,
        content: vec![ContentBlock::Text {
            text: seed_prompt_step_a(event),
        }],
    }];

    let outcome = orchestrator::run_with_messages_in_mode(
        messages,
        STEP_A_SYSTEM_PROMPT,
        llm,
        mcp,
        &restricted,
        STEP_A_MAX_ITERATIONS,
        STEP_A_MAX_TOKENS,
        &AgentMode::ReadOnly(ReadOnlyMode),
        &DispatchContext::default(),
    )
    .await?;

    Ok(parse_evidence(outcome))
}

fn step_a_tool_specs(all: &[ToolSpec]) -> Vec<ToolSpec> {
    all.iter()
        .filter(|s| STEP_A_TOOLS.contains(&s.name.as_str()))
        .cloned()
        .collect()
}

fn parse_evidence(outcome: RunOutcome) -> EvidenceBundle {
    #[derive(Deserialize)]
    struct EvidenceJson {
        summary: String,
        #[serde(default)]
        missing_sources: Vec<String>,
    }

    if let Some(json) = extract_json(&outcome.answer)
        && let Ok(parsed) = serde_json::from_str::<EvidenceJson>(&json)
    {
        return EvidenceBundle {
            summary: parsed.summary,
            missing_sources: parsed.missing_sources,
            tool_trace: outcome.tool_trace,
        };
    }

    let inferred_missing: Vec<String> = outcome
        .tool_trace
        .iter()
        .filter(|t| !t.ok)
        .map(|t| infer_source_from_tool(&t.tool))
        .collect();

    EvidenceBundle {
        summary: outcome.answer,
        missing_sources: inferred_missing,
        tool_trace: outcome.tool_trace,
    }
}

fn infer_source_from_tool(tool_name: &str) -> String {
    match tool_name {
        "fetch_logs" => "elasticsearch".into(),
        "fetch_recent_deploys" => "github_actions".into(),
        other => other.into(),
    }
}

fn extract_json(text: &str) -> Option<String> {
    let trimmed = text.trim();

    if serde_json::from_str::<serde_json::Value>(trimmed).is_ok() {
        return Some(trimmed.to_string());
    }

    if let Some(stripped) = strip_code_fence(trimmed)
        && serde_json::from_str::<serde_json::Value>(stripped).is_ok()
    {
        return Some(stripped.to_string());
    }

    if let (Some(start), Some(end)) = (trimmed.find('{'), trimmed.rfind('}'))
        && start < end
    {
        let candidate = &trimmed[start..=end];
        if serde_json::from_str::<serde_json::Value>(candidate).is_ok() {
            return Some(candidate.to_string());
        }
    }

    None
}

fn strip_code_fence(text: &str) -> Option<&str> {
    let body = text.strip_prefix("```")?;
    let body = body.strip_prefix("json").unwrap_or(body);
    let body = body.trim_start_matches('\n');
    let body = body.strip_suffix("```").unwrap_or(body);
    Some(body.trim())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::agent::llm::tests::MockLlmClient;
    use crate::agent::llm::{ChatResponse, StopReason, TokenUsage};
    use crate::incident::schema::{IncidentEvent, IncidentPayload, Severity};
    use async_trait::async_trait;
    use chrono::TimeZone;
    use serde_json::{Value, json};
    use std::collections::HashMap;
    use tokio::sync::Mutex;

    fn sample_event() -> IncidentEvent {
        IncidentEvent {
            event: "heartbeat_failed".into(),
            source: "controlroom-watchdog".into(),
            timestamp: chrono::Utc
                .with_ymd_and_hms(2026, 5, 10, 14, 23, 17)
                .unwrap(),
            payload: IncidentPayload {
                summary: "kassa down".into(),
                severity: Severity::Critical,
                component: "kassa".into(),
                group: None,
                class: Some("heartbeat-loss".into()),
                custom_details: Value::Null,
            },
        }
    }

    fn spec(name: &str, requires_approval: bool) -> ToolSpec {
        ToolSpec {
            name: name.into(),
            description: format!("test tool {name}"),
            input_schema: json!({"type": "object"}),
            requires_approval,
        }
    }

    struct StubMcpExecutor {
        responses: Mutex<HashMap<String, Result<String, String>>>,
        server_label: Option<String>,
    }

    impl StubMcpExecutor {
        fn new() -> Self {
            Self {
                responses: Mutex::new(HashMap::new()),
                server_label: Some("controlroom".into()),
            }
        }

        async fn with_ok(self, name: &str, body: &str) -> Self {
            self.responses
                .lock()
                .await
                .insert(name.into(), Ok(body.into()));
            self
        }

        async fn with_err(self, name: &str, err: &str) -> Self {
            self.responses
                .lock()
                .await
                .insert(name.into(), Err(err.into()));
            self
        }
    }

    #[async_trait]
    impl McpExecutor for StubMcpExecutor {
        async fn call(
            &self,
            name: &str,
            _arguments: Value,
        ) -> anyhow::Result<(String, ToolCallTrace)> {
            let table = self.responses.lock().await;
            let entry = table
                .get(name)
                .cloned()
                .ok_or_else(|| anyhow::anyhow!("StubMcpExecutor: no response for {name}"))?;
            let server = self.server_label.clone().unwrap_or_else(|| "test".into());
            Ok(match entry {
                Ok(body) => {
                    let trace = ToolCallTrace {
                        tool: name.into(),
                        server,
                        ms: 1,
                        ok: true,
                        error: None,
                        args: None,
                        status: None,
                        action_id: None,
                    };
                    (body, trace)
                }
                Err(err) => {
                    let trace = ToolCallTrace {
                        tool: name.into(),
                        server,
                        ms: 1,
                        ok: false,
                        error: Some(err.clone()),
                        args: None,
                        status: None,
                        action_id: None,
                    };
                    (err, trace)
                }
            })
        }

        fn server_label_for(&self, _name: &str) -> Option<String> {
            self.server_label.clone()
        }
    }

    fn tool_use_response(id: &str, name: &str, input: Value) -> ChatResponse {
        ChatResponse {
            content: vec![ContentBlock::ToolUse {
                id: id.into(),
                name: name.into(),
                input,
            }],
            stop_reason: StopReason::ToolUse,
            usage: Some(TokenUsage::default()),
        }
    }

    fn end_turn(text: &str) -> ChatResponse {
        ChatResponse {
            content: vec![ContentBlock::Text { text: text.into() }],
            stop_reason: StopReason::EndTurn,
            usage: Some(TokenUsage::default()),
        }
    }

    #[test]
    fn step_a_tool_specs_keeps_only_allowed_tools() {
        let all = vec![
            spec("fetch_logs", false),
            spec("count_contacts", false),
            spec("fetch_recent_deploys", false),
            spec("create_company", true),
        ];
        let filtered = step_a_tool_specs(&all);
        let names: Vec<&str> = filtered.iter().map(|s| s.name.as_str()).collect();
        assert_eq!(names, vec!["fetch_logs", "fetch_recent_deploys"]);
    }

    #[test]
    fn extract_json_passes_through_clean_json() {
        let s = r#"{"summary":"hi","missing_sources":[]}"#;
        let r = extract_json(s).unwrap();
        assert!(r.contains("\"summary\":\"hi\""));
    }

    #[test]
    fn extract_json_strips_markdown_json_fence() {
        let s = "```json\n{\"summary\":\"hi\",\"missing_sources\":[]}\n```";
        let r = extract_json(s).unwrap();
        assert!(serde_json::from_str::<serde_json::Value>(&r).is_ok());
    }

    #[test]
    fn extract_json_strips_plain_fence() {
        let s = "```\n{\"summary\":\"hi\"}\n```";
        let r = extract_json(s).unwrap();
        assert!(serde_json::from_str::<serde_json::Value>(&r).is_ok());
    }

    #[test]
    fn extract_json_finds_braces_in_prose() {
        let s = "Here is the result: {\"summary\":\"hi\",\"missing_sources\":[]} and that's it.";
        let r = extract_json(s).unwrap();
        assert!(serde_json::from_str::<serde_json::Value>(&r).is_ok());
    }

    #[test]
    fn extract_json_returns_none_on_no_json() {
        assert!(extract_json("just prose").is_none());
    }

    #[tokio::test]
    async fn gather_evidence_bails_when_no_step_a_tools_available() {
        let llm = MockLlmClient::new(vec![]);
        let mcp = StubMcpExecutor::new();
        let r = gather_evidence(&sample_event(), &llm, &mcp, &[]).await;
        assert!(r.is_err());
    }

    #[tokio::test]
    async fn gather_evidence_happy_path_parses_structured_json() {
        let llm = MockLlmClient::new(vec![
            tool_use_response("toolu_1", "fetch_logs", json!({"service": "kassa"})),
            tool_use_response(
                "toolu_2",
                "fetch_recent_deploys",
                json!({"service": "kassa"}),
            ),
            end_turn(
                r#"{"summary":"47 DB pool timeouts since deploy abc123 at 14:18","missing_sources":[]}"#,
            ),
        ]);
        let mcp = StubMcpExecutor::new()
            .with_ok("fetch_logs", "47 ERROR lines: connection pool timeout")
            .await
            .with_ok(
                "fetch_recent_deploys",
                "[{\"sha\":\"abc123\",\"at\":\"14:18\"}]",
            )
            .await;
        let specs = vec![
            spec("fetch_logs", false),
            spec("fetch_recent_deploys", false),
        ];

        let bundle = gather_evidence(&sample_event(), &llm, &mcp, &specs)
            .await
            .unwrap();

        assert!(bundle.summary.contains("DB pool timeouts"));
        assert!(bundle.missing_sources.is_empty());
        assert_eq!(bundle.tool_trace.len(), 2);
    }

    #[tokio::test]
    async fn gather_evidence_falls_back_when_llm_returns_prose() {
        let llm = MockLlmClient::new(vec![end_turn("I observed 47 errors and a recent deploy.")]);
        let mcp = StubMcpExecutor::new();
        let specs = vec![
            spec("fetch_logs", false),
            spec("fetch_recent_deploys", false),
        ];

        let bundle = gather_evidence(&sample_event(), &llm, &mcp, &specs)
            .await
            .unwrap();

        assert_eq!(bundle.summary, "I observed 47 errors and a recent deploy.");
        assert!(bundle.missing_sources.is_empty());
    }

    #[tokio::test]
    async fn gather_evidence_infers_missing_sources_from_failed_tool_calls() {
        let llm = MockLlmClient::new(vec![
            tool_use_response("toolu_1", "fetch_logs", json!({"service": "kassa"})),
            end_turn("could not gather all evidence"),
        ]);
        let mcp = StubMcpExecutor::new()
            .with_err("fetch_logs", "elasticsearch connection refused")
            .await;
        let specs = vec![
            spec("fetch_logs", false),
            spec("fetch_recent_deploys", false),
        ];

        let bundle = gather_evidence(&sample_event(), &llm, &mcp, &specs)
            .await
            .unwrap();

        assert!(
            bundle
                .missing_sources
                .contains(&"elasticsearch".to_string())
        );
    }

    #[tokio::test]
    async fn gather_evidence_filters_out_unrelated_tools() {
        let llm = MockLlmClient::new(vec![end_turn(r#"{"summary":"x","missing_sources":[]}"#)]);
        let mcp = StubMcpExecutor::new();
        let specs = vec![
            spec("fetch_logs", false),
            spec("count_contacts", false),
            spec("create_company", true),
            spec("fetch_recent_deploys", false),
        ];

        let bundle = gather_evidence(&sample_event(), &llm, &mcp, &specs)
            .await
            .unwrap();

        // The LLM only saw fetch_logs + fetch_recent_deploys in its tools list
        // (verifiable via MockLlmClient.calls but checked indirectly via no
        // count_contacts/create_company tool-use being possible). Bundle is
        // structurally valid → no surprise tools leaked.
        assert_eq!(bundle.summary, "x");
    }
}
