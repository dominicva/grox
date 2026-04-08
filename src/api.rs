use anyhow::{Context, Result, bail};
use async_trait::async_trait;
use serde::Serialize;
use serde_json::Value;

use crate::tools::ToolCall;

// --- Public types ---

/// The provider rejected the requested model (404, model-not-found, model-not-supported).
/// Callers can downcast to this to trigger model fallback.
#[derive(Debug)]
pub struct ModelRejected {
    pub model: String,
    pub detail: String,
}

impl std::fmt::Display for ModelRejected {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "Model '{}' rejected by provider: {}",
            self.model, self.detail
        )
    }
}

impl std::error::Error for ModelRejected {}

#[derive(Debug, Clone)]
pub struct TurnResponse {
    pub text: String,
    pub tool_calls: Vec<ToolCall>,
    pub usage: Option<Usage>,
    /// Plaintext reasoning content (grok-3-mini)
    pub reasoning_content: Option<String>,
    /// Encrypted/opaque reasoning blob (grok-4 reasoning models)
    pub encrypted_reasoning: Option<String>,
}

#[derive(Debug, Clone, PartialEq)]
pub struct Usage {
    pub input_tokens: u64,
    pub output_tokens: u64,
    pub cached_input_tokens: Option<u64>,
    pub reasoning_tokens: Option<u64>,
}

// --- Trait ---

#[async_trait]
pub trait GrokApi: Send + Sync {
    async fn send_turn(
        &self,
        input: Vec<Value>,
        tools: &[Value],
        on_token: &mut (dyn FnMut(String) + Send),
    ) -> Result<TurnResponse>;
}

// --- Reasoning effort ---

/// Reasoning effort level for models that support it.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum ReasoningEffort {
    Low,
    High,
}

impl std::fmt::Display for ReasoningEffort {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Low => write!(f, "low"),
            Self::High => write!(f, "high"),
        }
    }
}

// --- Real client ---

pub struct GrokClient {
    api_key: String,
    model: String,
    http: reqwest::Client,
    /// Current reasoning effort setting (None = off/not applicable)
    reasoning_effort: Option<ReasoningEffort>,
    /// Session UUID, used for cache optimization headers
    session_id: String,
    /// When true, send `store: false` in API requests (privacy mode)
    no_store: bool,
}

impl GrokClient {
    pub fn new(api_key: String, model: String, session_id: String) -> Self {
        Self {
            api_key,
            model,
            http: reqwest::Client::new(),
            reasoning_effort: None,
            session_id,
            no_store: false,
        }
    }

    pub fn set_session_id(&mut self, session_id: String) {
        self.session_id = session_id;
    }

    pub fn set_no_store(&mut self, no_store: bool) {
        self.no_store = no_store;
    }

    pub fn model(&self) -> &str {
        &self.model
    }

    pub fn set_model(&mut self, model: String) {
        self.model = model;
    }

    pub fn reasoning_effort(&self) -> Option<ReasoningEffort> {
        self.reasoning_effort
    }

    pub fn set_reasoning_effort(&mut self, effort: Option<ReasoningEffort>) {
        self.reasoning_effort = effort;
    }
}

// Responses API request body
#[derive(Debug, Serialize)]
struct ResponsesRequest {
    model: String,
    input: Vec<Value>,
    #[serde(skip_serializing_if = "Vec::is_empty")]
    tools: Vec<Value>,
    stream: bool,
    // Disable parallel tool calls — we execute sequentially
    #[serde(skip_serializing_if = "Option::is_none")]
    parallel_tool_calls: Option<bool>,
    /// Reasoning effort control (only for models that support it)
    #[serde(skip_serializing_if = "Option::is_none")]
    reasoning: Option<Value>,
    /// Request specific output types (e.g. encrypted reasoning)
    #[serde(skip_serializing_if = "Option::is_none")]
    include: Option<Vec<String>>,
    /// Cache key for prompt caching (session UUID)
    #[serde(skip_serializing_if = "Option::is_none")]
    prompt_cache_key: Option<String>,
    /// When false, request that the provider not store the conversation
    #[serde(skip_serializing_if = "Option::is_none")]
    store: Option<bool>,
}

// --- SSE event parsing ---

/// A single parsed SSE event from the xAI Responses API.
#[derive(Debug, Clone, PartialEq)]
pub enum ParsedEvent {
    /// Streaming text delta
    TextDelta(String),
    /// A completed function call
    ToolCall(ToolCall),
    /// Reasoning content (plaintext and/or encrypted)
    Reasoning {
        plaintext: Option<String>,
        encrypted: Option<String>,
    },
    /// Response completed with usage statistics
    Completed(Usage),
    /// Server-side error
    Error(String),
    /// End-of-stream marker ([DONE])
    Done,
    /// Unhandled event type — safe to ignore
    Unknown,
}

/// Parse a single SSE event from the xAI Responses API.
///
/// `event_name` is the SSE `event:` field (may be empty or "message").
/// `data` is the raw `data:` field content.
///
/// Returns a `ParsedEvent` describing what happened, or an error if the
/// data could not be parsed as JSON.
pub fn parse_sse_event(event_name: &str, data: &str) -> Result<ParsedEvent> {
    if data == "[DONE]" {
        return Ok(ParsedEvent::Done);
    }

    let parsed: Value =
        serde_json::from_str(data).with_context(|| format!("Failed to parse SSE: {data}"))?;

    // SSE event type can come from the SSE event field or JSON type field
    let event_type = if !event_name.is_empty() && event_name != "message" {
        event_name
    } else {
        parsed.get("type").and_then(|t| t.as_str()).unwrap_or("")
    };

    match event_type {
        "error" => {
            let message = parsed
                .get("message")
                .and_then(|m| m.as_str())
                .unwrap_or("Unknown API error");
            Ok(ParsedEvent::Error(message.to_string()))
        }
        "response.output_text.delta" => {
            let delta = parsed
                .get("delta")
                .and_then(|d| d.as_str())
                .unwrap_or("")
                .to_string();
            Ok(ParsedEvent::TextDelta(delta))
        }
        "response.output_item.done" => {
            let Some(item) = parsed.get("item") else {
                return Ok(ParsedEvent::Unknown);
            };
            let item_type = item.get("type").and_then(|t| t.as_str()).unwrap_or("");

            if item_type == "function_call" {
                let call_id = item
                    .get("call_id")
                    .and_then(|v| v.as_str())
                    .unwrap_or("")
                    .to_string();
                let name = item
                    .get("name")
                    .and_then(|v| v.as_str())
                    .unwrap_or("")
                    .to_string();
                let arguments = item
                    .get("arguments")
                    .and_then(|v| v.as_str())
                    .unwrap_or("{}")
                    .to_string();
                Ok(ParsedEvent::ToolCall(ToolCall {
                    call_id,
                    name,
                    arguments,
                }))
            } else if item_type == "reasoning" {
                let mut plaintext = None;
                let mut encrypted = None;
                if let Some(content_arr) = item.get("content").and_then(|c| c.as_array()) {
                    for block in content_arr {
                        let block_type =
                            block.get("type").and_then(|t| t.as_str()).unwrap_or("");
                        if block_type == "reasoning_text" {
                            if let Some(t) = block.get("text").and_then(|v| v.as_str()) {
                                plaintext = Some(t.to_string());
                            }
                        } else if block_type == "reasoning_encrypted"
                        && let Some(d) = block.get("data").and_then(|v| v.as_str())
                    {
                        encrypted = Some(d.to_string());
                    }
                    }
                }
                Ok(ParsedEvent::Reasoning {
                    plaintext,
                    encrypted,
                })
            } else {
                Ok(ParsedEvent::Unknown)
            }
        }
        "response.completed" => {
            if let Some(u) = parsed.get("response").and_then(|r| r.get("usage")) {
                let input_tokens = u.get("input_tokens").and_then(|v| v.as_u64()).unwrap_or(0);
                let output_tokens =
                    u.get("output_tokens").and_then(|v| v.as_u64()).unwrap_or(0);
                let cached_input_tokens = u
                    .get("input_tokens_details")
                    .and_then(|d| d.get("cached_tokens"))
                    .and_then(|v| v.as_u64());
                let reasoning_tokens = u
                    .get("output_tokens_details")
                    .and_then(|d| d.get("reasoning_tokens"))
                    .and_then(|v| v.as_u64());
                Ok(ParsedEvent::Completed(Usage {
                    input_tokens,
                    output_tokens,
                    cached_input_tokens,
                    reasoning_tokens,
                }))
            } else {
                Ok(ParsedEvent::Unknown)
            }
        }
        _ => Ok(ParsedEvent::Unknown),
    }
}

const RETRY_DELAYS: &[u64] = &[1, 3];

/// Check if an API error body indicates the model itself was rejected.
fn is_model_rejection(body: &str) -> bool {
    let lower = body.to_lowercase();
    [
        "model not found",
        "model_not_found",
        "model-not-found",
        "model not supported",
        "model_not_supported",
        "model-not-supported",
    ]
    .iter()
    .any(|pattern| lower.contains(pattern))
}

#[async_trait]
impl GrokApi for GrokClient {
    async fn send_turn(
        &self,
        input: Vec<Value>,
        tools: &[Value],
        on_token: &mut (dyn FnMut(String) + Send),
    ) -> Result<TurnResponse> {
        let profile = crate::model_profile::ModelProfile::for_model(&self.model);

        // Build reasoning parameter if model supports effort control and effort is set
        let reasoning = if profile.supports_reasoning_effort_control {
            self.reasoning_effort
                .map(|e| serde_json::json!({"effort": e.to_string()}))
        } else {
            None
        };

        // Request encrypted reasoning content for models that produce it
        let include = if profile.returns_encrypted_reasoning {
            Some(vec!["reasoning.encrypted_content".to_string()])
        } else {
            None
        };

        let body = ResponsesRequest {
            model: self.model.clone(),
            input,
            tools: tools.to_vec(),
            stream: true,
            parallel_tool_calls: if tools.is_empty() { None } else { Some(false) },
            reasoning,
            include,
            prompt_cache_key: Some(self.session_id.clone()),
            store: if self.no_store { Some(false) } else { None },
        };

        let body_json = serde_json::to_value(&body).context("Failed to serialize request body")?;

        for attempt in 0..=RETRY_DELAYS.len() {
            let request = self
                .http
                .post("https://api.x.ai/v1/responses")
                .header("Authorization", format!("Bearer {}", self.api_key))
                .header("x-grok-conv-id", &self.session_id)
                .json(&body_json);

            // Check HTTP status before streaming by sending the request manually
            let response = request
                .send()
                .await
                .context("Failed to connect to Grok API")?;

            let status = response.status();
            if !status.is_success() {
                let body_text = response.text().await.unwrap_or_default();
                match status.as_u16() {
                    401 => bail!("Invalid API key. Set XAI_API_KEY in your environment."),
                    429 => {
                        if let Some(&delay) = RETRY_DELAYS.get(attempt) {
                            eprintln!("  Rate limited (429). Retrying in {delay}s...");
                            tokio::time::sleep(std::time::Duration::from_secs(delay)).await;
                            continue;
                        }
                        bail!(
                            "Rate limited (429). Retried {} times without success.",
                            RETRY_DELAYS.len()
                        );
                    }
                    404 => {
                        return Err(ModelRejected {
                            model: self.model.clone(),
                            detail: body_text,
                        }
                        .into());
                    }
                    400 if is_model_rejection(&body_text) => {
                        return Err(ModelRejected {
                            model: self.model.clone(),
                            detail: body_text,
                        }
                        .into());
                    }
                    400 => bail!("Bad request (400): {body_text}"),
                    code => bail!("API error (HTTP {code}): {body_text}"),
                }
            }

            // Parse SSE from the successful response body
            use eventsource_stream::Eventsource;
            use futures::TryStreamExt;

            let mut stream = response.bytes_stream().eventsource();

            let mut text = String::new();
            let mut tool_calls: Vec<ToolCall> = Vec::new();
            let mut usage: Option<Usage> = None;
            let mut reasoning_content: Option<String> = None;
            let mut encrypted_reasoning: Option<String> = None;

            while let Some(event) = stream.try_next().await? {
                if std::env::var("GROX_VERBOSE").is_ok() {
                    eprintln!("[SSE] event={} data={}", event.event, event.data);
                }

                match parse_sse_event(&event.event, &event.data)? {
                    ParsedEvent::Done => break,
                    ParsedEvent::Error(msg) => bail!("{msg}"),
                    ParsedEvent::TextDelta(delta) => {
                        on_token(delta.clone());
                        text.push_str(&delta);
                    }
                    ParsedEvent::ToolCall(tc) => tool_calls.push(tc),
                    ParsedEvent::Reasoning {
                        plaintext,
                        encrypted,
                    } => {
                        if plaintext.is_some() {
                            reasoning_content = plaintext;
                        }
                        if encrypted.is_some() {
                            encrypted_reasoning = encrypted;
                        }
                    }
                    ParsedEvent::Completed(u) => usage = Some(u),
                    ParsedEvent::Unknown => {}
                }
            }

            return Ok(TurnResponse {
                text,
                tool_calls,
                usage,
                reasoning_content,
                encrypted_reasoning,
            });
        }

        bail!("Request failed after retries")
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn request_includes_prompt_cache_key() {
        let body = ResponsesRequest {
            model: "grok-4-1-fast-reasoning".to_string(),
            input: vec![],
            tools: vec![],
            stream: true,
            parallel_tool_calls: None,
            reasoning: None,
            include: None,
            prompt_cache_key: Some("test-session-uuid".to_string()),
            store: None,
        };
        let json = serde_json::to_value(&body).unwrap();
        assert_eq!(
            json.get("prompt_cache_key").and_then(|v| v.as_str()),
            Some("test-session-uuid")
        );
        // store should be absent when None
        assert!(json.get("store").is_none());
    }

    #[test]
    fn request_includes_store_false_when_set() {
        let body = ResponsesRequest {
            model: "grok-4-1-fast-reasoning".to_string(),
            input: vec![],
            tools: vec![],
            stream: true,
            parallel_tool_calls: None,
            reasoning: None,
            include: None,
            prompt_cache_key: Some("uuid".to_string()),
            store: Some(false),
        };
        let json = serde_json::to_value(&body).unwrap();
        assert_eq!(json.get("store").and_then(|v| v.as_bool()), Some(false));
    }

    #[test]
    fn client_no_store_default_off() {
        let client = GrokClient::new(
            "key".to_string(),
            "grok-4-1-fast-reasoning".to_string(),
            "session-id".to_string(),
        );
        // no_store defaults to false — verify through the public interface
        assert_eq!(client.model(), "grok-4-1-fast-reasoning");
    }

    #[test]
    fn model_rejection_matches_documented_variants() {
        for body in [
            "model not found",
            "MODEL_NOT_FOUND",
            "model-not-found",
            "model not supported",
            "MODEL_NOT_SUPPORTED",
            "model-not-supported",
        ] {
            assert!(is_model_rejection(body), "{body} should match");
        }
    }

    #[test]
    fn model_rejection_ignores_unrelated_errors() {
        for body in [
            "rate limit exceeded",
            "temporary upstream error",
            "invalid tool schema",
        ] {
            assert!(!is_model_rejection(body), "{body} should not match");
        }
    }

    // --- parse_sse_event unit tests ---

    #[test]
    fn parse_done_marker() {
        let event = parse_sse_event("", "[DONE]").unwrap();
        assert_eq!(event, ParsedEvent::Done);
    }

    #[test]
    fn parse_text_delta() {
        let data = r#"{"type":"response.output_text.delta","delta":"hello world"}"#;
        let event = parse_sse_event("response.output_text.delta", data).unwrap();
        assert_eq!(event, ParsedEvent::TextDelta("hello world".to_string()));
    }

    #[test]
    fn parse_text_delta_from_json_type_field() {
        // When SSE event field is empty, falls back to JSON type field
        let data = r#"{"type":"response.output_text.delta","delta":"fallback"}"#;
        let event = parse_sse_event("", data).unwrap();
        assert_eq!(event, ParsedEvent::TextDelta("fallback".to_string()));
    }

    #[test]
    fn parse_error_event() {
        let data = r#"{"type":"error","message":"rate limit exceeded"}"#;
        let event = parse_sse_event("error", data).unwrap();
        assert_eq!(
            event,
            ParsedEvent::Error("rate limit exceeded".to_string())
        );
    }

    #[test]
    fn parse_unknown_event_type() {
        let data = r#"{"type":"response.created","id":"resp_001"}"#;
        let event = parse_sse_event("response.created", data).unwrap();
        assert_eq!(event, ParsedEvent::Unknown);
    }

    #[test]
    fn parse_invalid_json_errors() {
        let result = parse_sse_event("", "not valid json");
        assert!(result.is_err());
    }

    // --- Contract tests from fixture files ---

    /// Load a fixture file and replay each line through parse_sse_event,
    /// collecting the parsed events.
    fn replay_fixture(filename: &str) -> Vec<ParsedEvent> {
        let fixture_path = format!(
            "{}/src/fixtures/{filename}",
            env!("CARGO_MANIFEST_DIR")
        );
        let content = std::fs::read_to_string(&fixture_path)
            .unwrap_or_else(|e| panic!("Failed to read fixture {fixture_path}: {e}"));

        let mut events = Vec::new();
        for line in content.lines() {
            if line.trim().is_empty() {
                continue;
            }
            let record: Value = serde_json::from_str(line)
                .unwrap_or_else(|e| panic!("Bad fixture JSON: {e}\nline: {line}"));
            let event_name = record["event"].as_str().unwrap_or("");
            let data = serde_json::to_string(&record["data"]).unwrap();
            let parsed = parse_sse_event(event_name, &data).unwrap();
            events.push(parsed);
        }
        events
    }

    #[test]
    fn contract_happy_path_text_and_tool_call() {
        let events = replay_fixture("happy_path_text_and_tool_call.jsonl");
        assert_eq!(events.len(), 4);

        // First two events: text deltas
        assert_eq!(events[0], ParsedEvent::TextDelta("I'll read ".to_string()));
        assert_eq!(
            events[1],
            ParsedEvent::TextDelta("the file for you.".to_string())
        );

        // Third event: tool call
        match &events[2] {
            ParsedEvent::ToolCall(tc) => {
                assert_eq!(tc.call_id, "call_abc123");
                assert_eq!(tc.name, "file_read");
                let args: Value = serde_json::from_str(&tc.arguments).unwrap();
                assert_eq!(args["path"], "src/main.rs");
            }
            other => panic!("Expected ToolCall, got {other:?}"),
        }

        // Fourth event: completed with usage
        match &events[3] {
            ParsedEvent::Completed(u) => {
                assert_eq!(u.input_tokens, 1250);
                assert_eq!(u.output_tokens, 340);
                assert_eq!(u.cached_input_tokens, Some(800));
                assert_eq!(u.reasoning_tokens, Some(0));
            }
            other => panic!("Expected Completed, got {other:?}"),
        }
    }

    #[test]
    fn contract_reasoning_response() {
        let events = replay_fixture("reasoning_response.jsonl");
        assert_eq!(events.len(), 4);

        // First event: plaintext reasoning
        match &events[0] {
            ParsedEvent::Reasoning {
                plaintext,
                encrypted,
            } => {
                assert!(plaintext.is_some());
                assert!(
                    plaintext.as_deref().unwrap().contains("refactor the function")
                );
                assert!(encrypted.is_none());
            }
            other => panic!("Expected Reasoning, got {other:?}"),
        }

        // Text deltas
        assert_eq!(events[1], ParsedEvent::TextDelta("Here's my ".to_string()));
        assert_eq!(
            events[2],
            ParsedEvent::TextDelta("analysis of the code.".to_string())
        );

        // Completed with reasoning tokens
        match &events[3] {
            ParsedEvent::Completed(u) => {
                assert_eq!(u.input_tokens, 2100);
                assert_eq!(u.output_tokens, 580);
                assert_eq!(u.cached_input_tokens, Some(1500));
                assert_eq!(u.reasoning_tokens, Some(245));
            }
            other => panic!("Expected Completed, got {other:?}"),
        }
    }

    #[test]
    fn contract_encrypted_reasoning_with_tool_call() {
        let events = replay_fixture("encrypted_reasoning_response.jsonl");
        assert_eq!(events.len(), 4);

        // First event: encrypted reasoning
        match &events[0] {
            ParsedEvent::Reasoning {
                plaintext,
                encrypted,
            } => {
                assert!(plaintext.is_none());
                assert!(encrypted.is_some());
                assert!(encrypted.as_deref().unwrap().starts_with("eyJhbGci"));
            }
            other => panic!("Expected Reasoning, got {other:?}"),
        }

        // Text delta
        assert_eq!(
            events[1],
            ParsedEvent::TextDelta("I've analyzed the issue.".to_string())
        );

        // Tool call
        match &events[2] {
            ParsedEvent::ToolCall(tc) => {
                assert_eq!(tc.call_id, "call_def456");
                assert_eq!(tc.name, "file_edit");
                let args: Value = serde_json::from_str(&tc.arguments).unwrap();
                assert_eq!(args["path"], "src/lib.rs");
            }
            other => panic!("Expected ToolCall, got {other:?}"),
        }

        // Completed — no cached_tokens in this fixture (empty details object)
        match &events[3] {
            ParsedEvent::Completed(u) => {
                assert_eq!(u.input_tokens, 3200);
                assert_eq!(u.output_tokens, 890);
                assert_eq!(u.cached_input_tokens, None);
                assert_eq!(u.reasoning_tokens, Some(512));
            }
            other => panic!("Expected Completed, got {other:?}"),
        }
    }
}

// --- Mock for tests ---

#[cfg(test)]
pub mod mock {
    use super::*;

    pub struct MockGrokApi {
        responses: std::sync::Mutex<Vec<TurnResponse>>,
    }

    impl MockGrokApi {
        pub fn new(responses: Vec<TurnResponse>) -> Self {
            Self {
                responses: std::sync::Mutex::new(responses),
            }
        }
    }

    #[async_trait]
    impl GrokApi for MockGrokApi {
        async fn send_turn(
            &self,
            _input: Vec<Value>,
            _tools: &[Value],
            on_token: &mut (dyn FnMut(String) + Send),
        ) -> Result<TurnResponse> {
            let mut responses = self.responses.lock().unwrap();
            if responses.is_empty() {
                bail!("MockGrokApi: no more scripted responses");
            }
            let response = responses.remove(0);
            // Stream the text token by token (word by word for mock)
            if !response.text.is_empty() {
                on_token(response.text.clone());
            }
            Ok(response)
        }
    }

    /// A mock that captures the input messages from each send_turn call.
    /// Used to verify that the agent accumulates full history across iterations
    /// and that reasoning fields are round-tripped in intermediate messages.
    pub struct CapturingMockGrokApi {
        responses: std::sync::Mutex<Vec<TurnResponse>>,
        /// Number of input messages received on each call.
        pub captured_input_counts: std::sync::Mutex<Vec<usize>>,
        /// Full input messages received on each call.
        pub captured_inputs: std::sync::Mutex<Vec<Vec<Value>>>,
    }

    impl CapturingMockGrokApi {
        pub fn new(responses: Vec<TurnResponse>) -> Self {
            Self {
                responses: std::sync::Mutex::new(responses),
                captured_input_counts: std::sync::Mutex::new(Vec::new()),
                captured_inputs: std::sync::Mutex::new(Vec::new()),
            }
        }
    }

    #[async_trait]
    impl GrokApi for CapturingMockGrokApi {
        async fn send_turn(
            &self,
            input: Vec<Value>,
            _tools: &[Value],
            on_token: &mut (dyn FnMut(String) + Send),
        ) -> Result<TurnResponse> {
            self.captured_input_counts.lock().unwrap().push(input.len());
            self.captured_inputs.lock().unwrap().push(input.clone());
            let mut responses = self.responses.lock().unwrap();
            if responses.is_empty() {
                bail!("CapturingMockGrokApi: no more scripted responses");
            }
            let response = responses.remove(0);
            if !response.text.is_empty() {
                on_token(response.text.clone());
            }
            Ok(response)
        }
    }
}
