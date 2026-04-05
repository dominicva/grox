use anyhow::{Context, Result, bail};
use async_trait::async_trait;
use futures::StreamExt;
use reqwest_eventsource::{Event, EventSource};
use serde::Serialize;
use serde_json::Value;

use crate::tools::ToolCall;

// --- Public types ---

#[derive(Debug, Clone)]
pub struct TurnResponse {
    pub text: String,
    pub tool_calls: Vec<ToolCall>,
    pub usage: Option<Usage>,
    pub response_id: Option<String>,
}

#[derive(Debug, Clone)]
pub struct Usage {
    pub input_tokens: u64,
    pub output_tokens: u64,
}

// --- Trait ---

#[async_trait]
pub trait GrokApi: Send + Sync {
    async fn send_turn(
        &self,
        input: Vec<Value>,
        tools: &[Value],
        previous_response_id: Option<&str>,
        on_token: &mut (dyn FnMut(String) + Send),
    ) -> Result<TurnResponse>;
}

// --- Real client ---

pub struct GrokClient {
    api_key: String,
    model: String,
    http: reqwest::Client,
}

impl GrokClient {
    pub fn new(api_key: String, model: String) -> Self {
        Self {
            api_key,
            model,
            http: reqwest::Client::new(),
        }
    }

    pub fn model(&self) -> &str {
        &self.model
    }

    pub fn set_model(&mut self, model: String) {
        self.model = model;
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
    #[serde(skip_serializing_if = "Option::is_none")]
    previous_response_id: Option<String>,
    // Disable parallel tool calls — we execute sequentially
    #[serde(skip_serializing_if = "Option::is_none")]
    parallel_tool_calls: Option<bool>,
}

// SSE events are parsed from raw serde_json::Value rather than typed structs,
// since the Responses API sends many event types and we only care about a few.

#[async_trait]
impl GrokApi for GrokClient {
    async fn send_turn(
        &self,
        input: Vec<Value>,
        tools: &[Value],
        previous_response_id: Option<&str>,
        on_token: &mut (dyn FnMut(String) + Send),
    ) -> Result<TurnResponse> {
        let body = ResponsesRequest {
            model: self.model.clone(),
            input,
            tools: tools.to_vec(),
            stream: true,
            previous_response_id: previous_response_id.map(String::from),
            parallel_tool_calls: if tools.is_empty() { None } else { Some(false) },
        };

        let request = self
            .http
            .post("https://api.x.ai/v1/responses")
            .header("Authorization", format!("Bearer {}", self.api_key))
            .json(&body);

        let mut es = EventSource::new(request)
            .context("Failed to connect to Grok API")?;

        let mut text = String::new();
        let mut tool_calls: Vec<ToolCall> = Vec::new();
        let mut usage: Option<Usage> = None;
        let mut response_id: Option<String> = None;

        while let Some(event) = es.next().await {
            match event {
                Ok(Event::Open) => {}
                Ok(Event::Message(msg)) => {
                    if msg.data == "[DONE]" {
                        break;
                    }

                    if std::env::var("GROX_VERBOSE").is_ok() {
                        eprintln!("[SSE] event={} data={}", msg.event, msg.data);
                    }

                    let parsed: Value = serde_json::from_str(&msg.data)
                        .with_context(|| format!("Failed to parse SSE: {}", msg.data))?;

                    // SSE event type can come from the SSE event field or JSON type field
                    let event_type = if !msg.event.is_empty() && msg.event != "message" {
                        msg.event.as_str()
                    } else {
                        parsed.get("type")
                            .and_then(|t| t.as_str())
                            .unwrap_or("")
                    };

                    match event_type {
                        // Server-side error
                        "error" => {
                            let message = parsed.get("message")
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
                        // A complete output item (message or function_call)
                        "response.output_item.done" => {
                            if let Some(item) = parsed.get("item") {
                                let item_type = item.get("type")
                                    .and_then(|t| t.as_str())
                                    .unwrap_or("");

                                if item_type == "function_call" {
                                    let call_id = item.get("call_id")
                                        .and_then(|v| v.as_str())
                                        .unwrap_or("")
                                        .to_string();
                                    let name = item.get("name")
                                        .and_then(|v| v.as_str())
                                        .unwrap_or("")
                                        .to_string();
                                    let arguments = item.get("arguments")
                                        .and_then(|v| v.as_str())
                                        .unwrap_or("{}")
                                        .to_string();

                                    tool_calls.push(ToolCall {
                                        call_id,
                                        name,
                                        arguments,
                                    });
                                }
                            }
                        }
                        // Response completed — extract usage and response_id
                        "response.completed" => {
                            if let Some(resp) = parsed.get("response") {
                                response_id = resp.get("id")
                                    .and_then(|v| v.as_str())
                                    .map(String::from);

                                if let Some(u) = resp.get("usage") {
                                    let input_tokens = u.get("input_tokens")
                                        .and_then(|v| v.as_u64())
                                        .unwrap_or(0);
                                    let output_tokens = u.get("output_tokens")
                                        .and_then(|v| v.as_u64())
                                        .unwrap_or(0);
                                    usage = Some(Usage { input_tokens, output_tokens });
                                }
                            }
                        }
                        _ => {}
                    }
                }
                Err(reqwest_eventsource::Error::StreamEnded) => break,
                Err(e) => bail!("Stream error: {e}"),
            }
        }

        Ok(TurnResponse {
            text,
            tool_calls,
            usage,
            response_id,
        })
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
            _previous_response_id: Option<&str>,
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
}
