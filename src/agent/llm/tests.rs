use super::*;
use tokio::sync::Mutex;

/// Captured per `chat()` invocation.
#[derive(Debug, Clone)]
pub struct MockCall {
    /// Reserved for tests that want to assert on the system prompt.
    #[allow(dead_code)]
    pub system: String,
    pub messages: Vec<Message>,
}

/// Test double: pops responses from a queue in order, records calls.
pub struct MockLlmClient {
    responses: Mutex<Vec<ChatResponse>>,
    streams: Mutex<Vec<Vec<anyhow::Result<StreamEvent>>>>,
    calls: Mutex<Vec<MockCall>>,
}

impl MockLlmClient {
    pub fn new(responses: Vec<ChatResponse>) -> Self {
        Self {
            responses: Mutex::new(responses),
            streams: Mutex::new(Vec::new()),
            calls: Mutex::new(Vec::new()),
        }
    }

    /// Queue a custom event sequence for the next `stream_chat` call.
    /// When no stream is queued, `stream_chat` falls back to the trait
    /// default — `chat` + a single `Done` event.
    #[allow(dead_code)] // exercised from orchestrator tests via stream pathway
    pub async fn queue_stream(&self, events: Vec<StreamEvent>) {
        let wrapped: Vec<anyhow::Result<StreamEvent>> = events.into_iter().map(Ok).collect();
        self.streams.lock().await.push(wrapped);
    }

    /// Like `queue_stream` but accepts raw `Result` items so tests can
    /// inject mid-stream transport errors as `Err`.
    #[allow(dead_code)]
    pub async fn queue_stream_results(&self, items: Vec<anyhow::Result<StreamEvent>>) {
        self.streams.lock().await.push(items);
    }

    pub async fn calls(&self) -> Vec<MockCall> {
        self.calls.lock().await.clone()
    }
}

#[async_trait::async_trait]
impl LlmClient for MockLlmClient {
    async fn chat(
        &self,
        system: &str,
        messages: &[Message],
        _tools: &[ToolSpec],
        _max_tokens: u32,
    ) -> anyhow::Result<ChatResponse> {
        self.calls.lock().await.push(MockCall {
            system: system.to_string(),
            messages: messages.to_vec(),
        });
        let mut q = self.responses.lock().await;
        if q.is_empty() {
            anyhow::bail!("MockLlmClient: response queue exhausted");
        }
        Ok(q.remove(0))
    }

    async fn stream_chat(
        &self,
        system: &str,
        messages: &[Message],
        tools: &[ToolSpec],
        max_tokens: u32,
    ) -> anyhow::Result<BoxStream<'static, anyhow::Result<StreamEvent>>> {
        let queued = {
            let mut q = self.streams.lock().await;
            if q.is_empty() {
                None
            } else {
                Some(q.remove(0))
            }
        };
        if let Some(items) = queued {
            self.calls.lock().await.push(MockCall {
                system: system.to_string(),
                messages: messages.to_vec(),
            });
            return Ok(Box::pin(stream::iter(items)));
        }
        let resp = self.chat(system, messages, tools, max_tokens).await?;
        let event = StreamEvent::Done {
            stop_reason: resp.stop_reason,
            usage: resp.usage,
            full_content: resp.content,
        };
        Ok(Box::pin(stream::once(async move { Ok(event) })))
    }
}

// Orchestrator tests live next to the orchestrator (in src/orchestrator.rs)
// because they need the test-only `TestExecutor` from that module. This
// module's job is the trait + types + the MockLlmClient test double.

#[test]
fn token_usage_add_sums_input_and_output() {
    let mut a = TokenUsage {
        input: 100,
        output: 50,
        cache_creation_input: None,
        cache_read_input: None,
    };
    let b = TokenUsage {
        input: 200,
        output: 30,
        cache_creation_input: None,
        cache_read_input: None,
    };
    a.add(&b);
    assert_eq!(a.input, 300);
    assert_eq!(a.output, 80);
    assert_eq!(a.cache_creation_input, None);
    assert_eq!(a.cache_read_input, None);
}

#[tokio::test]
async fn stream_chat_default_impl_emits_single_done_event() {
    use futures_util::StreamExt;
    let response = ChatResponse {
        content: vec![ContentBlock::Text { text: "hi".into() }],
        stop_reason: StopReason::EndTurn,
        usage: Some(TokenUsage {
            input: 7,
            output: 2,
            cache_creation_input: None,
            cache_read_input: None,
        }),
    };
    let mock = MockLlmClient::new(vec![response.clone()]);
    let mut s = mock.stream_chat("sys", &[], &[], 1024).await.unwrap();
    let first = s.next().await.expect("event").unwrap();
    match first {
        StreamEvent::Done {
            stop_reason,
            usage,
            full_content,
        } => {
            assert_eq!(stop_reason, StopReason::EndTurn);
            assert_eq!(usage, response.usage);
            assert_eq!(full_content, response.content);
        }
        other => panic!("expected Done, got {other:?}"),
    }
    assert!(s.next().await.is_none(), "stream should terminate");
}

#[tokio::test]
async fn stream_chat_queued_stream_emits_events_in_order() {
    use futures_util::StreamExt;
    let events = vec![
        StreamEvent::TextDelta("Hel".into()),
        StreamEvent::TextDelta("lo".into()),
        StreamEvent::Done {
            stop_reason: StopReason::EndTurn,
            usage: None,
            full_content: vec![ContentBlock::Text {
                text: "Hello".into(),
            }],
        },
    ];
    let mock = MockLlmClient::new(vec![]);
    mock.queue_stream(events.clone()).await;

    let mut s = mock.stream_chat("sys", &[], &[], 1024).await.unwrap();
    let mut got = Vec::new();
    while let Some(item) = s.next().await {
        got.push(item.unwrap());
    }
    assert_eq!(got, events);
}

#[test]
fn token_usage_add_sums_optional_cache_fields() {
    let mut a = TokenUsage {
        input: 0,
        output: 0,
        cache_creation_input: Some(10),
        cache_read_input: None,
    };
    let b = TokenUsage {
        input: 0,
        output: 0,
        cache_creation_input: Some(5),
        cache_read_input: Some(7),
    };
    a.add(&b);
    assert_eq!(a.cache_creation_input, Some(15));
    assert_eq!(a.cache_read_input, Some(7));
}
