use super::*;
use serde_json::json;
use wiremock::matchers::{header, method, path};
use wiremock::{Mock, MockServer, ResponseTemplate};

#[test]
fn to_wire_tools_drops_requires_approval() {
    // requires_approval is mcp-master-internal metadata; the wire-payload
    // sent to Anthropic must contain only name/description/input_schema.
    let specs = vec![ToolSpec {
        name: "delete_company".into(),
        description: "Soft-delete an Account.".into(),
        input_schema: json!({"type": "object"}),
        requires_approval: true,
    }];
    let wire = to_wire_tools(&specs);
    let value = serde_json::to_value(&wire).unwrap();
    let obj = value.as_array().unwrap()[0].as_object().unwrap();
    assert!(obj.contains_key("name"));
    assert!(obj.contains_key("description"));
    assert!(obj.contains_key("input_schema"));
    assert!(
        !obj.contains_key("requires_approval"),
        "requires_approval must NOT leak to Anthropic wire-payload",
    );
}

#[test]
fn translation_preserves_text_and_tool_use_shapes() {
    let messages = vec![
        Message {
            role: Role::User,
            content: vec![ContentBlock::Text {
                text: "Wat zijn de heartbeats?".to_string(),
            }],
        },
        Message {
            role: Role::Assistant,
            content: vec![ContentBlock::ToolUse {
                id: "toolu_1".to_string(),
                name: "heartbeat_status".to_string(),
                input: json!({"limit": 5}),
            }],
        },
        Message {
            role: Role::User,
            content: vec![ContentBlock::ToolResult {
                tool_use_id: "toolu_1".to_string(),
                content: "[]".to_string(),
                is_error: false,
            }],
        },
    ];

    let wire = to_wire_messages(&messages);
    let json_value = serde_json::to_value(&wire).unwrap();

    let expected = json!([
        { "role": "user",      "content": [{ "type": "text", "text": "Wat zijn de heartbeats?" }] },
        { "role": "assistant", "content": [{ "type": "tool_use", "id": "toolu_1", "name": "heartbeat_status", "input": {"limit": 5} }] },
        { "role": "user",      "content": [{ "type": "tool_result", "tool_use_id": "toolu_1", "content": "[]", "is_error": false }] }
    ]);

    assert_eq!(json_value, expected);
}

#[test]
fn from_wire_response_maps_stop_reasons() {
    let happy = AnthropicResponse {
        content: vec![AnthropicContent::Text { text: "ok".into() }],
        stop_reason: "end_turn".into(),
        usage: None,
    };
    let mapped = from_wire_response(happy);
    assert_eq!(mapped.stop_reason, StopReason::EndTurn);

    let tu = AnthropicResponse {
        content: vec![],
        stop_reason: "tool_use".into(),
        usage: None,
    };
    assert_eq!(from_wire_response(tu).stop_reason, StopReason::ToolUse);

    let mt = AnthropicResponse {
        content: vec![],
        stop_reason: "max_tokens".into(),
        usage: None,
    };
    assert_eq!(from_wire_response(mt).stop_reason, StopReason::MaxTokens);

    let unknown = AnthropicResponse {
        content: vec![],
        stop_reason: "pause_turn".into(),
        usage: None,
    };
    assert_eq!(
        from_wire_response(unknown).stop_reason,
        StopReason::Other("pause_turn".into())
    );
}

#[tokio::test]
async fn chat_against_wiremock_end_turn_path() {
    let server = MockServer::start().await;
    Mock::given(method("POST"))
        .and(path("/v1/messages"))
        .and(header("x-api-key", "test-key"))
        .and(header("anthropic-version", "2023-06-01"))
        .respond_with(ResponseTemplate::new(200).set_body_json(json!({
            "content": [{ "type": "text", "text": "Klaar." }],
            "stop_reason": "end_turn"
        })))
        .mount(&server)
        .await;

    let client = AnthropicClient::new("test-key".into()).with_base_url(server.uri());
    let resp = client.chat("system", &[], &[], 4096).await.unwrap();

    assert_eq!(resp.stop_reason, StopReason::EndTurn);
    assert_eq!(
        resp.content,
        vec![ContentBlock::Text {
            text: "Klaar.".into()
        }]
    );
}

#[tokio::test]
async fn chat_against_wiremock_tool_use_path() {
    let server = MockServer::start().await;
    Mock::given(method("POST"))
        .and(path("/v1/messages"))
        .respond_with(ResponseTemplate::new(200).set_body_json(json!({
            "content": [{
                "type": "tool_use",
                "id": "toolu_42",
                "name": "heartbeat_status",
                "input": {"limit": 5}
            }],
            "stop_reason": "tool_use"
        })))
        .mount(&server)
        .await;

    let client = AnthropicClient::new("k".into()).with_base_url(server.uri());
    let resp = client.chat("sys", &[], &[], 4096).await.unwrap();

    assert_eq!(resp.stop_reason, StopReason::ToolUse);
    match &resp.content[0] {
        ContentBlock::ToolUse { id, name, input } => {
            assert_eq!(id, "toolu_42");
            assert_eq!(name, "heartbeat_status");
            assert_eq!(input, &json!({"limit": 5}));
        }
        other => panic!("expected ToolUse, got {other:?}"),
    }
}

#[tokio::test]
async fn chat_retries_on_429_then_succeeds() {
    let server = MockServer::start().await;
    // First request: 429
    Mock::given(method("POST"))
        .and(path("/v1/messages"))
        .respond_with(ResponseTemplate::new(429).set_body_string("rate limited"))
        .up_to_n_times(1)
        .mount(&server)
        .await;
    // Subsequent requests: 200
    Mock::given(method("POST"))
        .and(path("/v1/messages"))
        .respond_with(ResponseTemplate::new(200).set_body_json(json!({
            "content": [{ "type": "text", "text": "ok after retry" }],
            "stop_reason": "end_turn"
        })))
        .mount(&server)
        .await;

    let client = AnthropicClient::new("k".into()).with_base_url(server.uri());
    let resp = client.chat("sys", &[], &[], 4096).await.unwrap();
    assert_eq!(resp.stop_reason, StopReason::EndTurn);
    match &resp.content[0] {
        ContentBlock::Text { text } => assert_eq!(text, "ok after retry"),
        other => panic!("expected Text, got {other:?}"),
    }
}

#[tokio::test]
async fn chat_does_not_retry_on_401() {
    let server = MockServer::start().await;
    Mock::given(method("POST"))
        .and(path("/v1/messages"))
        .respond_with(ResponseTemplate::new(401).set_body_string("invalid api key"))
        .expect(1) // Hard assert: must receive exactly 1 request, no retries
        .mount(&server)
        .await;

    let client = AnthropicClient::new("bad".into()).with_base_url(server.uri());
    let err = client.chat("sys", &[], &[], 4096).await.unwrap_err();
    let msg = format!("{err}");
    assert!(msg.contains("401"), "error should mention status: {msg}");
}

#[tokio::test]
async fn chat_decodes_usage_field() {
    let server = MockServer::start().await;
    Mock::given(method("POST"))
        .and(path("/v1/messages"))
        .respond_with(ResponseTemplate::new(200).set_body_json(json!({
            "content": [{ "type": "text", "text": "ok" }],
            "stop_reason": "end_turn",
            "usage": {
                "input_tokens": 100,
                "output_tokens": 50,
                "cache_creation_input_tokens": 12,
                "cache_read_input_tokens": 8
            }
        })))
        .mount(&server)
        .await;

    let client = AnthropicClient::new("k".into()).with_base_url(server.uri());
    let resp = client.chat("sys", &[], &[], 4096).await.unwrap();

    let usage = resp.usage.expect("usage should be Some");
    assert_eq!(usage.input, 100);
    assert_eq!(usage.output, 50);
    assert_eq!(usage.cache_creation_input, Some(12));
    assert_eq!(usage.cache_read_input, Some(8));
}

#[tokio::test]
async fn chat_handles_missing_usage_field() {
    let server = MockServer::start().await;
    Mock::given(method("POST"))
        .and(path("/v1/messages"))
        .respond_with(ResponseTemplate::new(200).set_body_json(json!({
            "content": [{ "type": "text", "text": "ok" }],
            "stop_reason": "end_turn"
        })))
        .mount(&server)
        .await;

    let client = AnthropicClient::new("k".into()).with_base_url(server.uri());
    let resp = client.chat("sys", &[], &[], 4096).await.unwrap();

    assert_eq!(resp.usage, None);
}

#[tokio::test]
async fn stream_chat_text_delta_path() {
    use futures_util::StreamExt;
    let server = MockServer::start().await;
    let sse_body = concat!(
        "event: message_start\n",
        "data: {\"type\":\"message_start\",\"message\":{\"id\":\"msg_1\",\"usage\":{\"input_tokens\":5,\"output_tokens\":0}}}\n\n",
        "event: content_block_start\n",
        "data: {\"type\":\"content_block_start\",\"index\":0,\"content_block\":{\"type\":\"text\",\"text\":\"\"}}\n\n",
        "event: ping\n",
        "data: {\"type\":\"ping\"}\n\n",
        "event: content_block_delta\n",
        "data: {\"type\":\"content_block_delta\",\"index\":0,\"delta\":{\"type\":\"text_delta\",\"text\":\"Hello\"}}\n\n",
        "event: content_block_delta\n",
        "data: {\"type\":\"content_block_delta\",\"index\":0,\"delta\":{\"type\":\"text_delta\",\"text\":\" world\"}}\n\n",
        "event: content_block_stop\n",
        "data: {\"type\":\"content_block_stop\",\"index\":0}\n\n",
        "event: message_delta\n",
        "data: {\"type\":\"message_delta\",\"delta\":{\"stop_reason\":\"end_turn\"},\"usage\":{\"output_tokens\":3}}\n\n",
        "event: message_stop\n",
        "data: {\"type\":\"message_stop\"}\n\n",
    );
    Mock::given(method("POST"))
        .and(path("/v1/messages"))
        .respond_with(
            ResponseTemplate::new(200)
                .insert_header("content-type", "text/event-stream")
                .set_body_string(sse_body),
        )
        .mount(&server)
        .await;

    let client = AnthropicClient::new("k".into()).with_base_url(server.uri());
    let mut stream = client.stream_chat("sys", &[], &[], 4096).await.unwrap();
    let mut events = Vec::new();
    while let Some(e) = stream.next().await {
        events.push(e.unwrap());
    }
    assert_eq!(events.len(), 3, "got: {events:#?}");
    match &events[0] {
        StreamEvent::TextDelta(t) => assert_eq!(t, "Hello"),
        other => panic!("expected TextDelta, got {other:?}"),
    }
    match &events[1] {
        StreamEvent::TextDelta(t) => assert_eq!(t, " world"),
        other => panic!("expected TextDelta, got {other:?}"),
    }
    match &events[2] {
        StreamEvent::Done {
            stop_reason,
            usage,
            full_content,
        } => {
            assert_eq!(*stop_reason, StopReason::EndTurn);
            let u = usage.as_ref().expect("usage on Done");
            assert_eq!(u.input, 5);
            assert_eq!(u.output, 3);
            assert_eq!(
                full_content,
                &vec![ContentBlock::Text {
                    text: "Hello world".into()
                }]
            );
        }
        other => panic!("expected Done, got {other:?}"),
    }
}

#[tokio::test]
async fn stream_chat_reassembles_input_json_delta_fragments() {
    use futures_util::StreamExt;
    let server = MockServer::start().await;
    let sse_body = concat!(
        "event: message_start\n",
        "data: {\"type\":\"message_start\",\"message\":{}}\n\n",
        "event: content_block_start\n",
        "data: {\"type\":\"content_block_start\",\"index\":0,\"content_block\":{\"type\":\"tool_use\",\"id\":\"toolu_42\",\"name\":\"heartbeat_status\",\"input\":{}}}\n\n",
        "event: content_block_delta\n",
        "data: {\"type\":\"content_block_delta\",\"index\":0,\"delta\":{\"type\":\"input_json_delta\",\"partial_json\":\"{\\\"lim\"}}\n\n",
        "event: content_block_delta\n",
        "data: {\"type\":\"content_block_delta\",\"index\":0,\"delta\":{\"type\":\"input_json_delta\",\"partial_json\":\"it\\\":5}\"}}\n\n",
        "event: content_block_stop\n",
        "data: {\"type\":\"content_block_stop\",\"index\":0}\n\n",
        "event: message_delta\n",
        "data: {\"type\":\"message_delta\",\"delta\":{\"stop_reason\":\"tool_use\"}}\n\n",
        "event: message_stop\n",
        "data: {\"type\":\"message_stop\"}\n\n",
    );
    Mock::given(method("POST"))
        .and(path("/v1/messages"))
        .respond_with(
            ResponseTemplate::new(200)
                .insert_header("content-type", "text/event-stream")
                .set_body_string(sse_body),
        )
        .mount(&server)
        .await;

    let client = AnthropicClient::new("k".into()).with_base_url(server.uri());
    let mut stream = client.stream_chat("sys", &[], &[], 4096).await.unwrap();
    let mut events = Vec::new();
    while let Some(e) = stream.next().await {
        events.push(e.unwrap());
    }

    match &events[0] {
        StreamEvent::ToolUseStart { id, name } => {
            assert_eq!(id, "toolu_42");
            assert_eq!(name, "heartbeat_status");
        }
        other => panic!("expected ToolUseStart, got {other:?}"),
    }
    // ToolUseDelta fragments are forwarded individually...
    assert!(matches!(events[1], StreamEvent::ToolUseDelta { .. }));
    assert!(matches!(events[2], StreamEvent::ToolUseDelta { .. }));
    assert!(matches!(events[3], StreamEvent::ToolUseStop { .. }));
    // ...but Done.full_content has the parsed result, not the raw concat.
    match &events[4] {
        StreamEvent::Done {
            stop_reason,
            full_content,
            ..
        } => {
            assert_eq!(*stop_reason, StopReason::ToolUse);
            assert_eq!(full_content.len(), 1);
            match &full_content[0] {
                ContentBlock::ToolUse { id, name, input } => {
                    assert_eq!(id, "toolu_42");
                    assert_eq!(name, "heartbeat_status");
                    assert_eq!(input, &json!({"limit": 5}));
                }
                other => panic!("expected ToolUse, got {other:?}"),
            }
        }
        other => panic!("expected Done, got {other:?}"),
    }
}

#[tokio::test]
async fn stream_chat_preserves_thinking_signature_byte_for_byte() {
    use futures_util::StreamExt;
    let server = MockServer::start().await;
    let sse_body = concat!(
        "event: message_start\n",
        "data: {\"type\":\"message_start\",\"message\":{}}\n\n",
        "event: content_block_start\n",
        "data: {\"type\":\"content_block_start\",\"index\":0,\"content_block\":{\"type\":\"thinking\",\"thinking\":\"\"}}\n\n",
        "event: content_block_delta\n",
        "data: {\"type\":\"content_block_delta\",\"index\":0,\"delta\":{\"type\":\"thinking_delta\",\"thinking\":\"Let me\"}}\n\n",
        "event: content_block_delta\n",
        "data: {\"type\":\"content_block_delta\",\"index\":0,\"delta\":{\"type\":\"thinking_delta\",\"thinking\":\" consider.\"}}\n\n",
        "event: content_block_delta\n",
        "data: {\"type\":\"content_block_delta\",\"index\":0,\"delta\":{\"type\":\"signature_delta\",\"signature\":\"sig_part_one_\"}}\n\n",
        "event: content_block_delta\n",
        "data: {\"type\":\"content_block_delta\",\"index\":0,\"delta\":{\"type\":\"signature_delta\",\"signature\":\"sig_part_two\"}}\n\n",
        "event: content_block_stop\n",
        "data: {\"type\":\"content_block_stop\",\"index\":0}\n\n",
        "event: content_block_start\n",
        "data: {\"type\":\"content_block_start\",\"index\":1,\"content_block\":{\"type\":\"text\",\"text\":\"\"}}\n\n",
        "event: content_block_delta\n",
        "data: {\"type\":\"content_block_delta\",\"index\":1,\"delta\":{\"type\":\"text_delta\",\"text\":\"OK.\"}}\n\n",
        "event: content_block_stop\n",
        "data: {\"type\":\"content_block_stop\",\"index\":1}\n\n",
        "event: message_delta\n",
        "data: {\"type\":\"message_delta\",\"delta\":{\"stop_reason\":\"end_turn\"}}\n\n",
        "event: message_stop\n",
        "data: {\"type\":\"message_stop\"}\n\n",
    );
    Mock::given(method("POST"))
        .and(path("/v1/messages"))
        .respond_with(
            ResponseTemplate::new(200)
                .insert_header("content-type", "text/event-stream")
                .set_body_string(sse_body),
        )
        .mount(&server)
        .await;

    let client = AnthropicClient::new("k".into()).with_base_url(server.uri());
    let mut stream = client.stream_chat("sys", &[], &[], 4096).await.unwrap();
    let mut events = Vec::new();
    while let Some(e) = stream.next().await {
        events.push(e.unwrap());
    }
    let done = events.last().expect("at least one event");
    match done {
        StreamEvent::Done { full_content, .. } => {
            assert_eq!(full_content.len(), 2);
            match &full_content[0] {
                ContentBlock::Thinking {
                    thinking,
                    signature,
                } => {
                    assert_eq!(thinking, "Let me consider.");
                    // Byte-for-byte signature reconstruction is the critical
                    // invariant: Anthropic rejects the next call otherwise.
                    assert_eq!(signature, "sig_part_one_sig_part_two");
                }
                other => panic!("expected Thinking, got {other:?}"),
            }
            match &full_content[1] {
                ContentBlock::Text { text } => assert_eq!(text, "OK."),
                other => panic!("expected Text, got {other:?}"),
            }
        }
        other => panic!("expected Done, got {other:?}"),
    }
}

#[tokio::test]
async fn stream_chat_partial_text_on_premature_close_yields_done_not_err() {
    use futures_util::StreamExt;
    let server = MockServer::start().await;
    // Body deliberately omits `message_stop` — simulates Anthropic
    // closing the TCP connection mid-stream after some content
    // (the partial-network-failure mid-stream case).
    let sse_body = concat!(
        "event: message_start\n",
        "data: {\"type\":\"message_start\",\"message\":{\"usage\":{\"input_tokens\":7}}}\n\n",
        "event: content_block_start\n",
        "data: {\"type\":\"content_block_start\",\"index\":0,\"content_block\":{\"type\":\"text\",\"text\":\"\"}}\n\n",
        "event: content_block_delta\n",
        "data: {\"type\":\"content_block_delta\",\"index\":0,\"delta\":{\"type\":\"text_delta\",\"text\":\"partial answer\"}}\n\n",
    );
    Mock::given(method("POST"))
        .and(path("/v1/messages"))
        .respond_with(
            ResponseTemplate::new(200)
                .insert_header("content-type", "text/event-stream")
                .set_body_string(sse_body),
        )
        .mount(&server)
        .await;

    let client = AnthropicClient::new("k".into()).with_base_url(server.uri());
    let mut stream = client.stream_chat("sys", &[], &[], 4096).await.unwrap();
    let mut events = Vec::new();
    while let Some(e) = stream.next().await {
        events.push(e.unwrap());
    }
    // Expect: TextDelta + terminal Done (NOT Err)
    assert!(matches!(events[0], StreamEvent::TextDelta(_)));
    match events.last().expect("at least one event") {
        StreamEvent::Done {
            stop_reason,
            full_content,
            ..
        } => {
            assert_eq!(*stop_reason, StopReason::Other("premature_close".into()));
            assert_eq!(
                full_content,
                &vec![ContentBlock::Text {
                    text: "partial answer".into()
                }]
            );
        }
        other => panic!("expected terminal Done, got {other:?}"),
    }
}

#[tokio::test]
async fn stream_chat_retries_on_429_then_succeeds() {
    use futures_util::StreamExt;
    let server = MockServer::start().await;
    Mock::given(method("POST"))
        .and(path("/v1/messages"))
        .respond_with(ResponseTemplate::new(429).set_body_string("rate limited"))
        .up_to_n_times(1)
        .mount(&server)
        .await;
    let sse_body = concat!(
        "event: message_start\n",
        "data: {\"type\":\"message_start\",\"message\":{}}\n\n",
        "event: message_delta\n",
        "data: {\"type\":\"message_delta\",\"delta\":{\"stop_reason\":\"end_turn\"}}\n\n",
        "event: message_stop\n",
        "data: {\"type\":\"message_stop\"}\n\n",
    );
    Mock::given(method("POST"))
        .and(path("/v1/messages"))
        .respond_with(
            ResponseTemplate::new(200)
                .insert_header("content-type", "text/event-stream")
                .set_body_string(sse_body),
        )
        .mount(&server)
        .await;

    let client = AnthropicClient::new("k".into()).with_base_url(server.uri());
    let mut stream = client.stream_chat("sys", &[], &[], 4096).await.unwrap();
    let mut events = Vec::new();
    while let Some(e) = stream.next().await {
        events.push(e.unwrap());
    }
    assert!(matches!(events.last(), Some(StreamEvent::Done { .. })));
}

#[tokio::test]
async fn chat_propagates_non_2xx_with_body() {
    let server = MockServer::start().await;
    Mock::given(method("POST"))
        .and(path("/v1/messages"))
        .respond_with(ResponseTemplate::new(401).set_body_string("invalid api key"))
        .mount(&server)
        .await;

    let client = AnthropicClient::new("bad".into()).with_base_url(server.uri());
    let err = client.chat("sys", &[], &[], 4096).await.unwrap_err();

    let msg = format!("{err}");
    assert!(msg.contains("401"), "error should mention status: {msg}");
    assert!(
        msg.contains("invalid api key"),
        "error should include body: {msg}"
    );
}
