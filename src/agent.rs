use std::path::{Path, PathBuf};

use anyhow::Result;
use serde_json::{Value, json};

use crate::api::GrokApi;
use crate::session::TranscriptEntry;
use crate::tools::Tool;

const MAX_TURNS: usize = 25;

pub struct Agent<'a> {
    api: &'a dyn GrokApi,
    tool_defs: Vec<Value>,
    project_root: PathBuf,
}

impl<'a> Agent<'a> {
    pub fn new(api: &'a dyn GrokApi, project_root: &Path) -> Self {
        Self {
            api,
            tool_defs: Tool::definitions(),
            project_root: project_root.to_path_buf(),
        }
    }

    /// Run the agent loop for a single user turn.
    ///
    /// Takes the full message array (system prompt + conversation history + current user message)
    /// built by ContextAssembler. Accumulates messages locally across inner tool-call iterations.
    ///
    /// Returns the final text, usage, and all transcript entries generated during this turn
    /// (assistant messages, tool calls, tool results) for persistence.
    ///
    /// `on_authorize` is called before executing each tool. It receives the tool name
    /// and arguments, and returns true if the tool should be executed.
    pub async fn run(
        &self,
        input: Vec<Value>,
        on_token: &mut (dyn FnMut(String) + Send),
        on_tool_call: &mut (dyn FnMut(&str, &str) + Send),
        on_tool_result: &mut (dyn FnMut(&str, &str) + Send),
        on_authorize: &mut (dyn FnMut(&str, &str) -> bool + Send),
    ) -> Result<AgentResult> {
        let mut messages = input;
        let mut entries: Vec<TranscriptEntry> = Vec::new();
        let mut final_text = String::new();

        for _turn in 0..MAX_TURNS {
            let response = self
                .api
                .send_turn(
                    messages.clone(),
                    &self.tool_defs,
                    on_token,
                )
                .await?;

            final_text = response.text.clone();

            if response.tool_calls.is_empty() {
                // Final assistant message
                if !final_text.is_empty() {
                    entries.push(TranscriptEntry::assistant_message(&final_text));
                }
                return Ok(AgentResult {
                    text: final_text,
                    usage: response.usage,
                    entries,
                });
            }

            // Intermediate assistant message (may be empty if model went straight to tools)
            if !response.text.is_empty() {
                let entry = TranscriptEntry::assistant_message(&response.text);
                messages.push(json!({"role": "assistant", "content": response.text}));
                entries.push(entry);
            }

            // Execute tool calls and accumulate into messages
            for tc in &response.tool_calls {
                on_tool_call(&tc.name, &tc.arguments);

                // Record tool call in transcript and messages
                let tc_entry = TranscriptEntry::tool_call(&tc.call_id, &tc.name, &tc.arguments);
                messages.push(json!({
                    "type": "function_call",
                    "name": tc.name,
                    "arguments": tc.arguments,
                    "call_id": tc.call_id,
                }));
                entries.push(tc_entry);

                // Check permission before executing
                let output = if !on_authorize(&tc.name, &tc.arguments) {
                    "Permission denied by user".to_string()
                } else {
                    match Tool::from_name(&tc.name) {
                        Some(tool) => match tool.execute(&tc.arguments, &self.project_root).await {
                            Ok(result) => result,
                            Err(e) => format!("Error: {e}"),
                        },
                        None => format!("Unknown tool: {}", tc.name),
                    }
                };

                on_tool_result(&tc.name, &output);

                // Record tool result in transcript and messages
                let tr_entry = TranscriptEntry::tool_result(&tc.call_id, &tc.name, &output);
                messages.push(json!({
                    "type": "function_call_output",
                    "call_id": tc.call_id,
                    "output": output,
                }));
                entries.push(tr_entry);
            }
        }

        // Hit max turns
        Ok(AgentResult {
            text: final_text,
            usage: None,
            entries,
        })
    }
}

#[derive(Debug)]
pub struct AgentResult {
    pub text: String,
    pub usage: Option<crate::api::Usage>,
    /// Transcript entries generated during this turn, for persistence.
    pub entries: Vec<TranscriptEntry>,
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::api::TurnResponse;
    use crate::api::mock::MockGrokApi;
    use crate::tools::ToolCall;

    fn noop_token(_: String) {}
    fn noop_tool_call(_: &str, _: &str) {}
    fn noop_tool_result(_: &str, _: &str) {}
    fn allow_all(_: &str, _: &str) -> bool { true }

    #[tokio::test]
    async fn single_tool_call_round_trip() {
        let tmp = tempfile::NamedTempFile::new().unwrap();
        std::io::Write::write_all(&mut &tmp, b"fn main() {}").unwrap();
        let path = tmp.path().to_str().unwrap();

        let mock = MockGrokApi::new(vec![
            TurnResponse {
                text: String::new(),
                tool_calls: vec![ToolCall {
                    call_id: "call_1".into(),
                    name: "file_read".into(),
                    arguments: format!(r#"{{"path": "{}"}}"#, path),
                }],
                usage: None,
                response_id: Some("resp_1".into()),
            },
            TurnResponse {
                text: "The file contains a main function.".into(),
                tool_calls: vec![],
                usage: None,
                response_id: Some("resp_2".into()),
            },
        ]);

        let agent = Agent::new(&mock, std::path::Path::new("/tmp"));
        let input = vec![json!({"role": "user", "content": "read the file"})];

        let result = agent
            .run(input, &mut noop_token, &mut noop_tool_call, &mut noop_tool_result, &mut allow_all)
            .await
            .unwrap();

        assert_eq!(result.text, "The file contains a main function.");
    }

    #[tokio::test]
    async fn tool_error_returned_to_model() {
        let mock = MockGrokApi::new(vec![
            TurnResponse {
                text: String::new(),
                tool_calls: vec![ToolCall {
                    call_id: "call_1".into(),
                    name: "file_read".into(),
                    arguments: r#"{"path": "/nonexistent/file.rs"}"#.into(),
                }],
                usage: None,
                response_id: Some("resp_1".into()),
            },
            TurnResponse {
                text: "That file doesn't exist.".into(),
                tool_calls: vec![],
                usage: None,
                response_id: Some("resp_2".into()),
            },
        ]);

        let agent = Agent::new(&mock, std::path::Path::new("/tmp"));
        let input = vec![json!({"role": "user", "content": "read /nonexistent/file.rs"})];

        let result = agent
            .run(input, &mut noop_token, &mut noop_tool_call, &mut noop_tool_result, &mut allow_all)
            .await
            .unwrap();

        assert_eq!(result.text, "That file doesn't exist.");
    }

    #[tokio::test]
    async fn no_tool_calls_exits_immediately() {
        let mock = MockGrokApi::new(vec![TurnResponse {
            text: "Hello! How can I help?".into(),
            tool_calls: vec![],
            usage: None,
            response_id: Some("resp_1".into()),
        }]);

        let agent = Agent::new(&mock, std::path::Path::new("/tmp"));
        let input = vec![json!({"role": "user", "content": "hello"})];

        let result = agent
            .run(input, &mut noop_token, &mut noop_tool_call, &mut noop_tool_result, &mut allow_all)
            .await
            .unwrap();

        assert_eq!(result.text, "Hello! How can I help?");
    }

    #[tokio::test]
    async fn max_turns_stops_loop() {
        let responses: Vec<TurnResponse> = (0..MAX_TURNS)
            .map(|i| TurnResponse {
                text: String::new(),
                tool_calls: vec![ToolCall {
                    call_id: format!("call_{i}"),
                    name: "file_read".into(),
                    arguments: r#"{"path": "/dev/null"}"#.into(),
                }],
                usage: None,
                response_id: Some(format!("resp_{i}")),
            })
            .collect();

        let mock = MockGrokApi::new(responses);
        let agent = Agent::new(&mock, std::path::Path::new("/tmp"));
        let input = vec![json!({"role": "user", "content": "loop forever"})];

        let result = agent
            .run(input, &mut noop_token, &mut noop_tool_call, &mut noop_tool_result, &mut allow_all)
            .await
            .unwrap();

        assert!(result.text.is_empty());
    }

    #[tokio::test]
    async fn unknown_tool_returns_error_to_model() {
        let mock = MockGrokApi::new(vec![
            TurnResponse {
                text: String::new(),
                tool_calls: vec![ToolCall {
                    call_id: "call_1".into(),
                    name: "nonexistent_tool".into(),
                    arguments: "{}".into(),
                }],
                usage: None,
                response_id: Some("resp_1".into()),
            },
            TurnResponse {
                text: "I don't have that tool.".into(),
                tool_calls: vec![],
                usage: None,
                response_id: Some("resp_2".into()),
            },
        ]);

        let agent = Agent::new(&mock, std::path::Path::new("/tmp"));
        let input = vec![json!({"role": "user", "content": "use a fake tool"})];

        let result = agent
            .run(input, &mut noop_token, &mut noop_tool_call, &mut noop_tool_result, &mut allow_all)
            .await
            .unwrap();

        assert_eq!(result.text, "I don't have that tool.");
    }

    #[tokio::test]
    async fn permission_denied_returns_message_to_model() {
        let mock = MockGrokApi::new(vec![
            TurnResponse {
                text: String::new(),
                tool_calls: vec![ToolCall {
                    call_id: "call_1".into(),
                    name: "file_write".into(),
                    arguments: r#"{"path": "/tmp/test.txt", "content": "hello"}"#.into(),
                }],
                usage: None,
                response_id: Some("resp_1".into()),
            },
            TurnResponse {
                text: "Write was denied.".into(),
                tool_calls: vec![],
                usage: None,
                response_id: Some("resp_2".into()),
            },
        ]);

        let agent = Agent::new(&mock, std::path::Path::new("/tmp"));
        let input = vec![json!({"role": "user", "content": "write a file"})];

        let mut deny_all = |_: &str, _: &str| -> bool { false };

        let result = agent
            .run(input, &mut noop_token, &mut noop_tool_call, &mut noop_tool_result, &mut deny_all)
            .await
            .unwrap();

        assert_eq!(result.text, "Write was denied.");
    }

    // --- New tests for local history accumulation ---

    #[tokio::test]
    async fn returns_transcript_entries_for_simple_response() {
        let mock = MockGrokApi::new(vec![TurnResponse {
            text: "Hello!".into(),
            tool_calls: vec![],
            usage: None,
            response_id: None,
        }]);

        let agent = Agent::new(&mock, std::path::Path::new("/tmp"));
        let input = vec![json!({"role": "user", "content": "hi"})];

        let result = agent
            .run(input, &mut noop_token, &mut noop_tool_call, &mut noop_tool_result, &mut allow_all)
            .await
            .unwrap();

        assert_eq!(result.entries.len(), 1);
        match &result.entries[0] {
            TranscriptEntry::AssistantMessage { content, .. } => {
                assert_eq!(content, "Hello!");
            }
            other => panic!("Expected AssistantMessage, got {other:?}"),
        }
    }

    #[tokio::test]
    async fn returns_transcript_entries_for_tool_use() {
        let tmp = tempfile::NamedTempFile::new().unwrap();
        std::io::Write::write_all(&mut &tmp, b"test content").unwrap();
        let path = tmp.path().to_str().unwrap();

        let mock = MockGrokApi::new(vec![
            TurnResponse {
                text: String::new(),
                tool_calls: vec![ToolCall {
                    call_id: "call_1".into(),
                    name: "file_read".into(),
                    arguments: format!(r#"{{"path": "{}"}}"#, path),
                }],
                usage: None,
                response_id: None,
            },
            TurnResponse {
                text: "The file says test content.".into(),
                tool_calls: vec![],
                usage: None,
                response_id: None,
            },
        ]);

        let agent = Agent::new(&mock, std::path::Path::new("/tmp"));
        let input = vec![json!({"role": "user", "content": "read it"})];

        let result = agent
            .run(input, &mut noop_token, &mut noop_tool_call, &mut noop_tool_result, &mut allow_all)
            .await
            .unwrap();

        // Should have: ToolCall, ToolResult, AssistantMessage
        assert_eq!(result.entries.len(), 3);
        assert!(matches!(&result.entries[0], TranscriptEntry::ToolCall { name, .. } if name == "file_read"));
        assert!(matches!(&result.entries[1], TranscriptEntry::ToolResult { call_id, .. } if call_id == "call_1"));
        assert!(matches!(&result.entries[2], TranscriptEntry::AssistantMessage { content, .. } if content == "The file says test content."));
    }

    #[tokio::test]
    async fn empty_final_text_produces_no_assistant_entry() {
        let mock = MockGrokApi::new(vec![TurnResponse {
            text: String::new(),
            tool_calls: vec![],
            usage: None,
            response_id: None,
        }]);

        let agent = Agent::new(&mock, std::path::Path::new("/tmp"));
        let input = vec![json!({"role": "user", "content": "hi"})];

        let result = agent
            .run(input, &mut noop_token, &mut noop_tool_call, &mut noop_tool_result, &mut allow_all)
            .await
            .unwrap();

        assert!(result.entries.is_empty());
    }
}
