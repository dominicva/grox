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

#[derive(Debug, Clone)]
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
}

impl GrokClient {
    pub fn new(api_key: String, model: String) -> Self {
        Self {
            api_key,
            model,
            http: reqwest::Client::new(),
            reasoning_effort: None,
        }
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
}

// SSE events are parsed from raw serde_json::Value rather than typed structs,
// since the Responses API sends many event types and we only care about a few.

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
        };

        let body_json = serde_json::to_value(&body).context("Failed to serialize request body")?;

        for attempt in 0..=RETRY_DELAYS.len() {
            let request = self
                .http
                .post("https://api.x.ai/v1/responses")
                .header("Authorization", format!("Bearer {}", self.api_key))
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
                if event.data == "[DONE]" {
                    break;
                }

                if std::env::var("GROX_VERBOSE").is_ok() {
                    eprintln!("[SSE] event={} data={}", event.event, event.data);
                }

                let parsed: Value = serde_json::from_str(&event.data)
                    .with_context(|| format!("Failed to parse SSE: {}", event.data))?;

                // SSE event type can come from the SSE event field or JSON type field
                let event_type = if !event.event.is_empty() && event.event != "message" {
                    event.event.as_str()
                } else {
                    parsed.get("type").and_then(|t| t.as_str()).unwrap_or("")
                };

                match event_type {
                    // Server-side error
                    "error" => {
                        let message = parsed
                            .get("message")
                            .and_then(|m| m.as_str())
                            .unwrap_or("Unknown API error");
                        bail!("{message}");
                    }
                    // Streaming text delta
                    "response.output_text.delta" => {
                        if let Some(delta) = parsed.get("delta").and_then(|d| d.as_str()) {
                            let delta = delta.to_string();
                            on_token(delta.clone());
                            text.push_str(&delta);
                        }
                    }
                    // A complete output item (message, function_call, or reasoning)
                    "response.output_item.done" => {
                        if let Some(item) = parsed.get("item") {
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

                                tool_calls.push(ToolCall {
                                    call_id,
                                    name,
                                    arguments,
                                });
                            } else if item_type == "reasoning" {
                                // Reasoning output: extract plaintext or encrypted content
                                if let Some(content_arr) =
                                    item.get("content").and_then(|c| c.as_array())
                                {
                                    for block in content_arr {
                                        let block_type = block
                                            .get("type")
                                            .and_then(|t| t.as_str())
                                            .unwrap_or("");
                                        if block_type == "reasoning_text" {
                                            if let Some(t) =
                                                block.get("text").and_then(|v| v.as_str())
                                            {
                                                reasoning_content = Some(t.to_string());
                                            }
                                        } else if block_type == "reasoning_encrypted"
                                            && let Some(data) =
                                                block.get("data").and_then(|v| v.as_str())
                                        {
                                            encrypted_reasoning = Some(data.to_string());
                                        }
                                    }
                                }
                            }
                        }
                    }
                    // Response completed — extract usage
                    "response.completed" => {
                        if let Some(u) = parsed.get("response").and_then(|r| r.get("usage")) {
                            let input_tokens =
                                u.get("input_tokens").and_then(|v| v.as_u64()).unwrap_or(0);
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
                            usage = Some(Usage {
                                input_tokens,
                                output_tokens,
                                cached_input_tokens,
                                reasoning_tokens,
                            });
                        }
                    }
                    _ => {}
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
    use super::is_model_rejection;

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
