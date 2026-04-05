use anyhow::{Context, Result, bail};
use futures::StreamExt;
use reqwest_eventsource::{Event, EventSource};
use serde::{Deserialize, Serialize};

#[derive(Debug, Serialize, Clone)]
pub struct Message {
    pub role: String,
    pub content: String,
}

#[derive(Debug, Serialize)]
struct ChatRequest {
    model: String,
    messages: Vec<Message>,
    stream: bool,
}

// SSE streaming response types
#[derive(Debug, Deserialize)]
struct StreamChunk {
    choices: Vec<StreamChoice>,
}

#[derive(Debug, Deserialize)]
struct StreamChoice {
    delta: Delta,
    #[allow(dead_code)]
    finish_reason: Option<String>,
}

#[derive(Debug, Deserialize)]
struct Delta {
    content: Option<String>,
}

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

    /// Stream a chat completion from Grok, calling `on_token` for each token as it arrives.
    /// Returns the full assembled response.
    pub async fn stream_chat(
        &self,
        messages: &[Message],
        mut on_token: impl FnMut(&str),
    ) -> Result<String> {
        let body = ChatRequest {
            model: self.model.clone(),
            messages: messages.to_vec(),
            stream: true,
        };

        let request = self
            .http
            .post("https://api.x.ai/v1/chat/completions")
            .header("Authorization", format!("Bearer {}", self.api_key))
            .json(&body);

        let mut es = EventSource::new(request)
            .context("Failed to connect to Grok API")?;

        let mut full_response = String::new();

        while let Some(event) = es.next().await {
            match event {
                Ok(Event::Open) => {}
                Ok(Event::Message(msg)) => {
                    if msg.data == "[DONE]" {
                        break;
                    }
                    let chunk: StreamChunk = serde_json::from_str(&msg.data)
                        .with_context(|| format!("Failed to parse chunk: {}", msg.data))?;

                    for choice in &chunk.choices {
                        if let Some(content) = &choice.delta.content {
                            on_token(content);
                            full_response.push_str(content);
                        }
                    }
                }
                Err(reqwest_eventsource::Error::StreamEnded) => break,
                Err(e) => bail!("Stream error: {e}"),
            }
        }

        Ok(full_response)
    }
}
