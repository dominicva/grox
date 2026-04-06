use std::path::{Path, PathBuf};

use anyhow::Result;
use serde_json::{Value, json};

use crate::api::GrokApi;
use crate::session::TranscriptEntry;
use crate::tools::Tool;

const MAX_TURNS: usize = 25;

/// Tools that mutate the filesystem or environment.
/// After these execute, repo context should be refreshed.
const MUTATING_TOOLS: &[&str] = &["file_write", "file_edit", "shell_exec"];

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
    ///
    /// `on_context_refresh` is called after mutating tools execute. It should return
    /// a fresh system prompt (with updated repo context) to replace the existing one.
    pub async fn run(
        &self,
        input: Vec<Value>,
        on_token: &mut (dyn FnMut(String) + Send),
        on_tool_call: &mut (dyn FnMut(&str, &str) + Send),
        on_tool_result: &mut (dyn FnMut(&str, &str) + Send),
        on_authorize: &mut (dyn FnMut(&str, &str) -> bool + Send),
        on_context_refresh: &mut (dyn FnMut() -> Value + Send),
    ) -> Result<AgentResult> {
        let mut messages = input;
        let mut entries: Vec<TranscriptEntry> = Vec::new();
        let mut final_text = String::new();
        let mut cumulative_usage: Option<crate::api::Usage> = None;

        for _turn in 0..MAX_TURNS {
            let response = self
                .api
                .send_turn(
                    messages.clone(),
                    &self.tool_defs,
                    on_token,
                )
                .await?;

            // Accumulate usage across all inner iterations
            if let Some(u) = &response.usage {
                cumulative_usage = Some(match cumulative_usage {
                    Some(prev) => crate::api::Usage {
                        input_tokens: prev.input_tokens + u.input_tokens,
                        output_tokens: prev.output_tokens + u.output_tokens,
                    },
                    None => u.clone(),
                });
            }

            final_text = response.text.clone();

            if response.tool_calls.is_empty() {
                // Final assistant message
                if !final_text.is_empty() {
                    entries.push(TranscriptEntry::assistant_message(&final_text));
                }
                return Ok(AgentResult {
                    text: final_text,
                    usage: cumulative_usage,
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
            let mut had_mutation = false;

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
                let authorized = on_authorize(&tc.name, &tc.arguments);
                let output = if !authorized {
                    "Permission denied by user".to_string()
                } else {
                    if MUTATING_TOOLS.contains(&tc.name.as_str()) {
                        had_mutation = true;
                    }
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

            // Refresh system prompt after mutating tools so the next iteration
            // sees up-to-date repo context (git status, directory tree, etc.)
            if had_mutation {
                messages[0] = on_context_refresh();
            }
        }

        // Hit max turns
        Ok(AgentResult {
            text: final_text,
            usage: cumulative_usage,
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
    use crate::api::mock::{CapturingMockGrokApi, MockGrokApi};
    use crate::tools::ToolCall;

    fn noop_token(_: String) {}
    fn noop_tool_call(_: &str, _: &str) {}
    fn noop_tool_result(_: &str, _: &str) {}
    fn allow_all(_: &str, _: &str) -> bool { true }
    fn no_refresh() -> Value { json!({"role": "system", "content": "test"}) }

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
            },
            TurnResponse {
                text: "The file contains a main function.".into(),
                tool_calls: vec![],
                usage: None,
            },
        ]);

        let agent = Agent::new(&mock, std::path::Path::new("/tmp"));
        let input = vec![json!({"role": "user", "content": "read the file"})];

        let result = agent
            .run(input, &mut noop_token, &mut noop_tool_call, &mut noop_tool_result, &mut allow_all, &mut no_refresh)
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
            },
            TurnResponse {
                text: "That file doesn't exist.".into(),
                tool_calls: vec![],
                usage: None,
            },
        ]);

        let agent = Agent::new(&mock, std::path::Path::new("/tmp"));
        let input = vec![json!({"role": "user", "content": "read /nonexistent/file.rs"})];

        let result = agent
            .run(input, &mut noop_token, &mut noop_tool_call, &mut noop_tool_result, &mut allow_all, &mut no_refresh)
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
        }]);

        let agent = Agent::new(&mock, std::path::Path::new("/tmp"));
        let input = vec![json!({"role": "user", "content": "hello"})];

        let result = agent
            .run(input, &mut noop_token, &mut noop_tool_call, &mut noop_tool_result, &mut allow_all, &mut no_refresh)
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
            })
            .collect();

        let mock = MockGrokApi::new(responses);
        let agent = Agent::new(&mock, std::path::Path::new("/tmp"));
        let input = vec![json!({"role": "user", "content": "loop forever"})];

        let result = agent
            .run(input, &mut noop_token, &mut noop_tool_call, &mut noop_tool_result, &mut allow_all, &mut no_refresh)
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
            },
            TurnResponse {
                text: "I don't have that tool.".into(),
                tool_calls: vec![],
                usage: None,
            },
        ]);

        let agent = Agent::new(&mock, std::path::Path::new("/tmp"));
        let input = vec![json!({"role": "user", "content": "use a fake tool"})];

        let result = agent
            .run(input, &mut noop_token, &mut noop_tool_call, &mut noop_tool_result, &mut allow_all, &mut no_refresh)
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
            },
            TurnResponse {
                text: "Write was denied.".into(),
                tool_calls: vec![],
                usage: None,
            },
        ]);

        let agent = Agent::new(&mock, std::path::Path::new("/tmp"));
        let input = vec![json!({"role": "user", "content": "write a file"})];

        let mut deny_all = |_: &str, _: &str| -> bool { false };

        let result = agent
            .run(input, &mut noop_token, &mut noop_tool_call, &mut noop_tool_result, &mut deny_all, &mut no_refresh)
            .await
            .unwrap();

        assert_eq!(result.text, "Write was denied.");
    }

    // --- Transcript entry tests ---

    #[tokio::test]
    async fn returns_transcript_entries_for_simple_response() {
        let mock = MockGrokApi::new(vec![TurnResponse {
            text: "Hello!".into(),
            tool_calls: vec![],
            usage: None,
        }]);

        let agent = Agent::new(&mock, std::path::Path::new("/tmp"));
        let input = vec![json!({"role": "user", "content": "hi"})];

        let result = agent
            .run(input, &mut noop_token, &mut noop_tool_call, &mut noop_tool_result, &mut allow_all, &mut no_refresh)
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
            },
            TurnResponse {
                text: "The file says test content.".into(),
                tool_calls: vec![],
                usage: None,
            },
        ]);

        let agent = Agent::new(&mock, std::path::Path::new("/tmp"));
        let input = vec![json!({"role": "user", "content": "read it"})];

        let result = agent
            .run(input, &mut noop_token, &mut noop_tool_call, &mut noop_tool_result, &mut allow_all, &mut no_refresh)
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
        }]);

        let agent = Agent::new(&mock, std::path::Path::new("/tmp"));
        let input = vec![json!({"role": "user", "content": "hi"})];

        let result = agent
            .run(input, &mut noop_token, &mut noop_tool_call, &mut noop_tool_result, &mut allow_all, &mut no_refresh)
            .await
            .unwrap();

        assert!(result.entries.is_empty());
    }

    // --- Usage accumulation tests ---

    #[tokio::test]
    async fn usage_accumulated_across_inner_iterations() {
        let mock = MockGrokApi::new(vec![
            TurnResponse {
                text: String::new(),
                tool_calls: vec![ToolCall {
                    call_id: "call_1".into(),
                    name: "file_read".into(),
                    arguments: r#"{"path": "/dev/null"}"#.into(),
                }],
                usage: Some(crate::api::Usage { input_tokens: 100, output_tokens: 50 }),
            },
            TurnResponse {
                text: "Done.".into(),
                tool_calls: vec![],
                usage: Some(crate::api::Usage { input_tokens: 200, output_tokens: 75 }),
            },
        ]);

        let agent = Agent::new(&mock, std::path::Path::new("/tmp"));
        let input = vec![json!({"role": "user", "content": "read"})];

        let result = agent
            .run(input, &mut noop_token, &mut noop_tool_call, &mut noop_tool_result, &mut allow_all, &mut no_refresh)
            .await
            .unwrap();

        let usage = result.usage.unwrap();
        assert_eq!(usage.input_tokens, 300);  // 100 + 200
        assert_eq!(usage.output_tokens, 125); // 50 + 75
    }

    // --- Context refresh tests ---

    #[tokio::test]
    async fn context_refresh_called_after_mutating_tool() {
        let tmp = tempfile::tempdir().unwrap();
        let file_path = tmp.path().join("test.txt");

        let mock = MockGrokApi::new(vec![
            TurnResponse {
                text: String::new(),
                tool_calls: vec![ToolCall {
                    call_id: "call_1".into(),
                    name: "file_write".into(),
                    arguments: format!(r#"{{"path": "{}", "content": "hello"}}"#, file_path.display()),
                }],
                usage: None,
            },
            TurnResponse {
                text: "Written.".into(),
                tool_calls: vec![],
                usage: None,
            },
        ]);

        let agent = Agent::new(&mock, tmp.path());
        let input = vec![json!({"role": "system", "content": "old"}), json!({"role": "user", "content": "write"})];

        let mut refresh_count = 0;
        let result = agent
            .run(
                input,
                &mut noop_token,
                &mut noop_tool_call,
                &mut noop_tool_result,
                &mut allow_all,
                &mut || {
                    refresh_count += 1;
                    json!({"role": "system", "content": "refreshed"})
                },
            )
            .await
            .unwrap();

        assert_eq!(refresh_count, 1);
        assert_eq!(result.text, "Written.");
    }

    #[tokio::test]
    async fn context_refresh_not_called_for_read_only_tools() {
        let tmp = tempfile::NamedTempFile::new().unwrap();
        std::io::Write::write_all(&mut &tmp, b"data").unwrap();
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
            },
            TurnResponse {
                text: "Read it.".into(),
                tool_calls: vec![],
                usage: None,
            },
        ]);

        let agent = Agent::new(&mock, std::path::Path::new("/tmp"));
        let input = vec![json!({"role": "system", "content": "sys"}), json!({"role": "user", "content": "read"})];

        let mut refresh_count = 0;
        let result = agent
            .run(
                input,
                &mut noop_token,
                &mut noop_tool_call,
                &mut noop_tool_result,
                &mut allow_all,
                &mut || {
                    refresh_count += 1;
                    json!({"role": "system", "content": "refreshed"})
                },
            )
            .await
            .unwrap();

        assert_eq!(refresh_count, 0);
        assert_eq!(result.text, "Read it.");
    }

    #[tokio::test]
    async fn context_refresh_not_called_when_permission_denied() {
        let mock = MockGrokApi::new(vec![
            TurnResponse {
                text: String::new(),
                tool_calls: vec![ToolCall {
                    call_id: "call_1".into(),
                    name: "file_write".into(),
                    arguments: r#"{"path": "/tmp/x.txt", "content": "y"}"#.into(),
                }],
                usage: None,
            },
            TurnResponse {
                text: "Denied.".into(),
                tool_calls: vec![],
                usage: None,
            },
        ]);

        let agent = Agent::new(&mock, std::path::Path::new("/tmp"));
        let input = vec![json!({"role": "system", "content": "sys"}), json!({"role": "user", "content": "write"})];

        let mut deny_all = |_: &str, _: &str| -> bool { false };
        let mut refresh_count = 0;

        agent
            .run(
                input,
                &mut noop_token,
                &mut noop_tool_call,
                &mut noop_tool_result,
                &mut deny_all,
                &mut || {
                    refresh_count += 1;
                    json!({"role": "system", "content": "refreshed"})
                },
            )
            .await
            .unwrap();

        assert_eq!(refresh_count, 0, "should not refresh when tool was denied");
    }

    // --- History accumulation verification ---

    #[tokio::test]
    async fn full_history_sent_across_inner_iterations() {
        // Verifies that the agent sends growing message arrays on each inner
        // iteration, proving local history accumulation works end-to-end.
        let tmp = tempfile::NamedTempFile::new().unwrap();
        std::io::Write::write_all(&mut &tmp, b"content").unwrap();
        let path = tmp.path().to_str().unwrap();

        let mock = CapturingMockGrokApi::new(vec![
            // Iteration 1: model requests a file_read
            TurnResponse {
                text: String::new(),
                tool_calls: vec![ToolCall {
                    call_id: "call_1".into(),
                    name: "file_read".into(),
                    arguments: format!(r#"{{"path": "{}"}}"#, path),
                }],
                usage: None,
            },
            // Iteration 2: model requests another file_read
            TurnResponse {
                text: String::new(),
                tool_calls: vec![ToolCall {
                    call_id: "call_2".into(),
                    name: "file_read".into(),
                    arguments: format!(r#"{{"path": "{}"}}"#, path),
                }],
                usage: None,
            },
            // Iteration 3: model responds with text (no more tools)
            TurnResponse {
                text: "All done.".into(),
                tool_calls: vec![],
                usage: None,
            },
        ]);

        let agent = Agent::new(&mock, std::path::Path::new("/tmp"));
        // Initial input: system prompt + user message = 2 messages
        let input = vec![
            json!({"role": "system", "content": "sys"}),
            json!({"role": "user", "content": "read twice"}),
        ];

        let result = agent
            .run(input, &mut noop_token, &mut noop_tool_call, &mut noop_tool_result, &mut allow_all, &mut no_refresh)
            .await
            .unwrap();

        assert_eq!(result.text, "All done.");

        let counts = mock.captured_input_counts.lock().unwrap();
        assert_eq!(counts.len(), 3, "should have made 3 API calls");

        // Call 1: system + user = 2
        assert_eq!(counts[0], 2);
        // Call 2: system + user + tool_call_1 + tool_result_1 = 4
        assert_eq!(counts[1], 4);
        // Call 3: system + user + tool_call_1 + tool_result_1 + tool_call_2 + tool_result_2 = 6
        assert_eq!(counts[2], 6);
    }
}
