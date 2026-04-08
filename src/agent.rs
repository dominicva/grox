use std::path::{Path, PathBuf};

use anyhow::Result;
use serde_json::{Value, json};

use crate::api::GrokApi;
use crate::checkpoint::{self, FileSnapshot};
use crate::session::TranscriptEntry;
use crate::tools::Tool;

const MAX_TURNS: usize = 25;

/// Tools that mutate the filesystem or environment.
/// After these execute, repo context should be refreshed.
const MUTATING_TOOLS: &[&str] = &["file_write", "file_edit", "shell_exec"];

/// Tools that modify files and should be checkpointed.
/// shell_exec is NOT checkpointed (only tracked for warning).
const CHECKPOINTED_TOOLS: &[&str] = &["file_write", "file_edit"];

pub struct Agent<'a> {
    api: &'a dyn GrokApi,
    tool_defs: Vec<Value>,
    project_root: PathBuf,
    checkpoint_enabled: bool,
}

impl<'a> Agent<'a> {
    pub fn new(api: &'a dyn GrokApi, project_root: &Path) -> Self {
        let checkpoint_enabled = checkpoint::is_git_repo(project_root);
        Self {
            api,
            tool_defs: Tool::definitions(),
            project_root: project_root.to_path_buf(),
            checkpoint_enabled,
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
    #[allow(clippy::too_many_arguments, clippy::type_complexity)]
    pub async fn run(
        &self,
        input: Vec<Value>,
        on_token: &mut (dyn FnMut(String) + Send),
        on_tool_call: &mut (dyn FnMut(&str, &str) + Send),
        on_tool_result: &mut (dyn FnMut(&str, &str) + Send),
        on_authorize: &mut (dyn FnMut(&str, &str) -> bool + Send),
        on_context_refresh: &mut (dyn FnMut() -> Value + Send),
        on_entry: &mut (dyn FnMut(&TranscriptEntry) -> Result<()> + Send),
        on_reasoning: &mut (dyn FnMut(Option<&str>, Option<&str>, Option<u64>) + Send),
    ) -> Result<AgentResult> {
        let mut messages = input;
        let mut final_text = String::new();
        let mut cumulative_usage: Option<crate::api::Usage> = None;
        // Track shell_exec at the user-turn level (across all inner iterations)
        // so checkpoints carry the warning if shell_exec preceded them.
        let mut turn_had_shell_exec = false;

        for _turn in 0..MAX_TURNS {
            let response = self
                .api
                .send_turn(messages.clone(), &self.tool_defs, on_token)
                .await?;

            // Accumulate usage across all inner iterations
            if let Some(u) = &response.usage {
                cumulative_usage = Some(match cumulative_usage {
                    Some(prev) => crate::api::Usage {
                        input_tokens: prev.input_tokens + u.input_tokens,
                        output_tokens: prev.output_tokens + u.output_tokens,
                        cached_input_tokens: match (prev.cached_input_tokens, u.cached_input_tokens)
                        {
                            (Some(a), Some(b)) => Some(a + b),
                            (Some(a), None) | (None, Some(a)) => Some(a),
                            (None, None) => None,
                        },
                        reasoning_tokens: match (prev.reasoning_tokens, u.reasoning_tokens) {
                            (Some(a), Some(b)) => Some(a + b),
                            (Some(a), None) | (None, Some(a)) => Some(a),
                            (None, None) => None,
                        },
                    },
                    None => u.clone(),
                });
            }

            final_text = response.text.clone();

            // Notify caller of reasoning from every response (intermediate and final)
            if response.reasoning_content.is_some() || response.encrypted_reasoning.is_some() {
                let reasoning_tokens = response.usage.as_ref().and_then(|u| u.reasoning_tokens);
                on_reasoning(
                    response.reasoning_content.as_deref(),
                    response.encrypted_reasoning.as_deref(),
                    reasoning_tokens,
                );
            }

            if response.tool_calls.is_empty() {
                // Final assistant message — persist with reasoning payloads
                if !final_text.is_empty() {
                    let entry = TranscriptEntry::assistant_message_with_reasoning(
                        &final_text,
                        response.reasoning_content.clone(),
                        response.encrypted_reasoning.clone(),
                    );
                    on_entry(&entry)?;
                }
                return Ok(AgentResult {
                    text: final_text,
                    usage: cumulative_usage,
                });
            }

            // Intermediate assistant message (may be empty if model went straight to tools).
            // Persist reasoning payloads AND include them in the live messages buffer
            // so the next send_turn round-trips reasoning correctly.
            let has_content = !response.text.is_empty()
                || response.reasoning_content.is_some()
                || response.encrypted_reasoning.is_some();
            if has_content {
                let entry = TranscriptEntry::assistant_message_with_reasoning(
                    &response.text,
                    response.reasoning_content.clone(),
                    response.encrypted_reasoning.clone(),
                );
                let mut msg = json!({"role": "assistant", "content": response.text});
                if let Some(ref rc) = response.reasoning_content {
                    msg["reasoning_content"] = serde_json::Value::String(rc.clone());
                }
                if let Some(ref er) = response.encrypted_reasoning {
                    msg["encrypted_reasoning"] = serde_json::Value::String(er.clone());
                }
                messages.push(msg);
                on_entry(&entry)?;
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
                on_entry(&tc_entry)?;

                // Check permission before executing
                let authorized = on_authorize(&tc.name, &tc.arguments);
                let is_checkpointed = self.checkpoint_enabled
                    && authorized
                    && CHECKPOINTED_TOOLS.contains(&tc.name.as_str());

                // Snapshot file AFTER authorization but BEFORE execution.
                // Uses validate_checkpoint_path which works for files whose
                // parent directories don't exist yet (file_write creates them).
                // Returns the canonical path for consistent dedup.
                let pre_snapshot = if is_checkpointed {
                    checkpoint::extract_tool_path(&tc.arguments).and_then(|path| {
                        let abs_path =
                            checkpoint::resolve_checkpoint_path(&path, &self.project_root);
                        let canonical =
                            checkpoint::validate_checkpoint_path(&abs_path, &self.project_root)?;
                        let pre = checkpoint::snapshot_pre(&canonical, &self.project_root).ok()?;
                        Some((canonical, pre))
                    })
                } else {
                    None
                };

                let output = if !authorized {
                    "Permission denied by user".to_string()
                } else {
                    if MUTATING_TOOLS.contains(&tc.name.as_str()) {
                        had_mutation = true;
                    }
                    if tc.name == "shell_exec" {
                        // Track at turn level (persisted in turn_had_shell_exec)
                        turn_had_shell_exec = true;
                    }
                    match Tool::from_name(&tc.name) {
                        Some(tool) => tool.execute(&tc.arguments, &self.project_root).await.output,
                        None => format!("Unknown tool: {}", tc.name),
                    }
                };

                // Persist checkpoint IMMEDIATELY after execution, BEFORE
                // the tool result. This closes the gap where a file is already
                // changed on disk but no durable checkpoint exists yet.
                // Each file-modifying tool gets its own checkpoint entry.
                // Rewind processes all checkpoints in reverse order, so
                // multiple edits to the same file are handled correctly
                // without explicit dedup.
                //
                // Emit a checkpoint whenever the file was actually modified on
                // disk (post_hash differs from pre_hash), even if the tool
                // reported failure. This covers partial writes where the file
                // was mutated before an error occurred. Permission denials and
                // validation errors that never touch disk produce matching
                // hashes and are naturally excluded.
                if let Some((canonical, pre)) = pre_snapshot
                    && let Ok(post) = checkpoint::snapshot_post(&canonical, &self.project_root)
                    && post != pre
                {
                    let cp = TranscriptEntry::checkpoint(
                        vec![FileSnapshot {
                            path: canonical.display().to_string(),
                            pre_hash: pre,
                            post_hash: post,
                        }],
                        turn_had_shell_exec,
                    );
                    on_entry(&cp)?;
                }

                on_tool_result(&tc.name, &output);

                // Record tool result in transcript and messages
                let tr_entry = TranscriptEntry::tool_result(&tc.call_id, &tc.name, &output);
                messages.push(json!({
                    "type": "function_call_output",
                    "call_id": tc.call_id,
                    "output": output,
                }));
                on_entry(&tr_entry)?;
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
        })
    }
}

#[derive(Debug)]
pub struct AgentResult {
    pub text: String,
    pub usage: Option<crate::api::Usage>,
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
    fn allow_all(_: &str, _: &str) -> bool {
        true
    }
    fn no_refresh() -> Value {
        json!({"role": "system", "content": "test"})
    }
    fn noop_reasoning(_: Option<&str>, _: Option<&str>, _: Option<u64>) {}
    fn noop_entry(_: &TranscriptEntry) -> Result<()> {
        Ok(())
    }

    type EntryStore = std::sync::Arc<std::sync::Mutex<Vec<TranscriptEntry>>>;

    /// Collect entries emitted via on_entry callback.
    #[allow(clippy::type_complexity)]
    fn collecting_entries() -> (
        impl FnMut(&TranscriptEntry) -> Result<()> + Send,
        EntryStore,
    ) {
        let entries = std::sync::Arc::new(std::sync::Mutex::new(Vec::new()));
        let entries_clone = entries.clone();
        let callback = move |entry: &TranscriptEntry| -> Result<()> {
            entries_clone.lock().unwrap().push(entry.clone());
            Ok(())
        };
        (callback, entries)
    }

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
                reasoning_content: None,
                encrypted_reasoning: None,
            },
            TurnResponse {
                text: "The file contains a main function.".into(),
                tool_calls: vec![],
                usage: None,
                reasoning_content: None,
                encrypted_reasoning: None,
            },
        ]);

        let agent = Agent::new(&mock, std::path::Path::new("/tmp"));
        let input = vec![json!({"role": "user", "content": "read the file"})];

        let result = agent
            .run(
                input,
                &mut noop_token,
                &mut noop_tool_call,
                &mut noop_tool_result,
                &mut allow_all,
                &mut no_refresh,
                &mut noop_entry,
                &mut noop_reasoning,
            )
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
                reasoning_content: None,
                encrypted_reasoning: None,
            },
            TurnResponse {
                text: "That file doesn't exist.".into(),
                tool_calls: vec![],
                usage: None,
                reasoning_content: None,
                encrypted_reasoning: None,
            },
        ]);

        let agent = Agent::new(&mock, std::path::Path::new("/tmp"));
        let input = vec![json!({"role": "user", "content": "read /nonexistent/file.rs"})];

        let result = agent
            .run(
                input,
                &mut noop_token,
                &mut noop_tool_call,
                &mut noop_tool_result,
                &mut allow_all,
                &mut no_refresh,
                &mut noop_entry,
                &mut noop_reasoning,
            )
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
            reasoning_content: None,
            encrypted_reasoning: None,
        }]);

        let agent = Agent::new(&mock, std::path::Path::new("/tmp"));
        let input = vec![json!({"role": "user", "content": "hello"})];

        let result = agent
            .run(
                input,
                &mut noop_token,
                &mut noop_tool_call,
                &mut noop_tool_result,
                &mut allow_all,
                &mut no_refresh,
                &mut noop_entry,
                &mut noop_reasoning,
            )
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
                reasoning_content: None,
                encrypted_reasoning: None,
            })
            .collect();

        let mock = MockGrokApi::new(responses);
        let agent = Agent::new(&mock, std::path::Path::new("/tmp"));
        let input = vec![json!({"role": "user", "content": "loop forever"})];

        let result = agent
            .run(
                input,
                &mut noop_token,
                &mut noop_tool_call,
                &mut noop_tool_result,
                &mut allow_all,
                &mut no_refresh,
                &mut noop_entry,
                &mut noop_reasoning,
            )
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
                reasoning_content: None,
                encrypted_reasoning: None,
            },
            TurnResponse {
                text: "I don't have that tool.".into(),
                tool_calls: vec![],
                usage: None,
                reasoning_content: None,
                encrypted_reasoning: None,
            },
        ]);

        let agent = Agent::new(&mock, std::path::Path::new("/tmp"));
        let input = vec![json!({"role": "user", "content": "use a fake tool"})];

        let result = agent
            .run(
                input,
                &mut noop_token,
                &mut noop_tool_call,
                &mut noop_tool_result,
                &mut allow_all,
                &mut no_refresh,
                &mut noop_entry,
                &mut noop_reasoning,
            )
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
                reasoning_content: None,
                encrypted_reasoning: None,
            },
            TurnResponse {
                text: "Write was denied.".into(),
                tool_calls: vec![],
                usage: None,
                reasoning_content: None,
                encrypted_reasoning: None,
            },
        ]);

        let agent = Agent::new(&mock, std::path::Path::new("/tmp"));
        let input = vec![json!({"role": "user", "content": "write a file"})];

        let mut deny_all = |_: &str, _: &str| -> bool { false };

        let result = agent
            .run(
                input,
                &mut noop_token,
                &mut noop_tool_call,
                &mut noop_tool_result,
                &mut deny_all,
                &mut no_refresh,
                &mut noop_entry,
                &mut noop_reasoning,
            )
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
            reasoning_content: None,
            encrypted_reasoning: None,
        }]);

        let agent = Agent::new(&mock, std::path::Path::new("/tmp"));
        let input = vec![json!({"role": "user", "content": "hi"})];
        let (mut on_entry, collected) = collecting_entries();

        agent
            .run(
                input,
                &mut noop_token,
                &mut noop_tool_call,
                &mut noop_tool_result,
                &mut allow_all,
                &mut no_refresh,
                &mut on_entry,
                &mut noop_reasoning,
            )
            .await
            .unwrap();
        drop(on_entry);

        let entries = collected.lock().unwrap();
        assert_eq!(entries.len(), 1);
        match &entries[0] {
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
                reasoning_content: None,
                encrypted_reasoning: None,
            },
            TurnResponse {
                text: "The file says test content.".into(),
                tool_calls: vec![],
                usage: None,
                reasoning_content: None,
                encrypted_reasoning: None,
            },
        ]);

        let agent = Agent::new(&mock, std::path::Path::new("/tmp"));
        let input = vec![json!({"role": "user", "content": "read it"})];
        let (mut on_entry, collected) = collecting_entries();

        agent
            .run(
                input,
                &mut noop_token,
                &mut noop_tool_call,
                &mut noop_tool_result,
                &mut allow_all,
                &mut no_refresh,
                &mut on_entry,
                &mut noop_reasoning,
            )
            .await
            .unwrap();
        drop(on_entry);

        // Should have: ToolCall, ToolResult, AssistantMessage
        let entries = collected.lock().unwrap();
        assert_eq!(entries.len(), 3);
        assert!(
            matches!(&entries[0], TranscriptEntry::ToolCall { name, .. } if name == "file_read")
        );
        assert!(
            matches!(&entries[1], TranscriptEntry::ToolResult { call_id, .. } if call_id == "call_1")
        );
        assert!(
            matches!(&entries[2], TranscriptEntry::AssistantMessage { content, .. } if content == "The file says test content.")
        );
    }

    #[tokio::test]
    async fn empty_final_text_produces_no_assistant_entry() {
        let mock = MockGrokApi::new(vec![TurnResponse {
            text: String::new(),
            tool_calls: vec![],
            usage: None,
            reasoning_content: None,
            encrypted_reasoning: None,
        }]);

        let agent = Agent::new(&mock, std::path::Path::new("/tmp"));
        let input = vec![json!({"role": "user", "content": "hi"})];
        let (mut on_entry, collected) = collecting_entries();

        agent
            .run(
                input,
                &mut noop_token,
                &mut noop_tool_call,
                &mut noop_tool_result,
                &mut allow_all,
                &mut no_refresh,
                &mut on_entry,
                &mut noop_reasoning,
            )
            .await
            .unwrap();
        drop(on_entry);

        assert!(collected.lock().unwrap().is_empty());
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
                usage: Some(crate::api::Usage {
                    input_tokens: 100,
                    output_tokens: 50,
                    cached_input_tokens: None,
                    reasoning_tokens: None,
                }),
                reasoning_content: None,
                encrypted_reasoning: None,
            },
            TurnResponse {
                text: "Done.".into(),
                tool_calls: vec![],
                usage: Some(crate::api::Usage {
                    input_tokens: 200,
                    output_tokens: 75,
                    cached_input_tokens: None,
                    reasoning_tokens: None,
                }),
                reasoning_content: None,
                encrypted_reasoning: None,
            },
        ]);

        let agent = Agent::new(&mock, std::path::Path::new("/tmp"));
        let input = vec![json!({"role": "user", "content": "read"})];

        let result = agent
            .run(
                input,
                &mut noop_token,
                &mut noop_tool_call,
                &mut noop_tool_result,
                &mut allow_all,
                &mut no_refresh,
                &mut noop_entry,
                &mut noop_reasoning,
            )
            .await
            .unwrap();

        let usage = result.usage.unwrap();
        assert_eq!(usage.input_tokens, 300); // 100 + 200
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
                    arguments: format!(
                        r#"{{"path": "{}", "content": "hello"}}"#,
                        file_path.display()
                    ),
                }],
                usage: None,
                reasoning_content: None,
                encrypted_reasoning: None,
            },
            TurnResponse {
                text: "Written.".into(),
                tool_calls: vec![],
                usage: None,
                reasoning_content: None,
                encrypted_reasoning: None,
            },
        ]);

        let agent = Agent::new(&mock, tmp.path());
        let input = vec![
            json!({"role": "system", "content": "old"}),
            json!({"role": "user", "content": "write"}),
        ];

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
                &mut noop_entry,
                &mut noop_reasoning,
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
                reasoning_content: None,
                encrypted_reasoning: None,
            },
            TurnResponse {
                text: "Read it.".into(),
                tool_calls: vec![],
                usage: None,
                reasoning_content: None,
                encrypted_reasoning: None,
            },
        ]);

        let agent = Agent::new(&mock, std::path::Path::new("/tmp"));
        let input = vec![
            json!({"role": "system", "content": "sys"}),
            json!({"role": "user", "content": "read"}),
        ];

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
                &mut noop_entry,
                &mut noop_reasoning,
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
                reasoning_content: None,
                encrypted_reasoning: None,
            },
            TurnResponse {
                text: "Denied.".into(),
                tool_calls: vec![],
                usage: None,
                reasoning_content: None,
                encrypted_reasoning: None,
            },
        ]);

        let agent = Agent::new(&mock, std::path::Path::new("/tmp"));
        let input = vec![
            json!({"role": "system", "content": "sys"}),
            json!({"role": "user", "content": "write"}),
        ];

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
                &mut noop_entry,
                &mut noop_reasoning,
            )
            .await
            .unwrap();

        assert_eq!(refresh_count, 0, "should not refresh when tool was denied");
    }

    // --- Checkpoint tests ---

    fn setup_git_repo() -> tempfile::TempDir {
        let dir = tempfile::tempdir().unwrap();
        std::process::Command::new("git")
            .args(["init"])
            .current_dir(dir.path())
            .output()
            .unwrap();
        std::process::Command::new("git")
            .args(["config", "user.email", "test@test.com"])
            .current_dir(dir.path())
            .output()
            .unwrap();
        std::process::Command::new("git")
            .args(["config", "user.name", "Test"])
            .current_dir(dir.path())
            .output()
            .unwrap();
        dir
    }

    #[tokio::test]
    async fn checkpoint_emitted_for_file_write() {
        let repo = setup_git_repo();
        let file_path = repo.path().join("test.txt");

        let mock = MockGrokApi::new(vec![
            TurnResponse {
                text: String::new(),
                tool_calls: vec![ToolCall {
                    call_id: "call_1".into(),
                    name: "file_write".into(),
                    arguments: format!(
                        r#"{{"path": "{}", "content": "hello"}}"#,
                        file_path.display()
                    ),
                }],
                usage: None,
                reasoning_content: None,
                encrypted_reasoning: None,
            },
            TurnResponse {
                text: "Done.".into(),
                tool_calls: vec![],
                usage: None,
                reasoning_content: None,
                encrypted_reasoning: None,
            },
        ]);

        let agent = Agent::new(&mock, repo.path());
        let input = vec![
            json!({"role": "system", "content": "sys"}),
            json!({"role": "user", "content": "write"}),
        ];
        let (mut on_entry, collected) = collecting_entries();

        agent
            .run(
                input,
                &mut noop_token,
                &mut noop_tool_call,
                &mut noop_tool_result,
                &mut allow_all,
                &mut no_refresh,
                &mut on_entry,
                &mut noop_reasoning,
            )
            .await
            .unwrap();
        drop(on_entry);

        // Should have: ToolCall, ToolResult, Checkpoint, AssistantMessage
        let entries = collected.lock().unwrap();
        let checkpoint_entries: Vec<_> = entries
            .iter()
            .filter(|e| matches!(e, TranscriptEntry::Checkpoint { .. }))
            .collect();
        assert_eq!(checkpoint_entries.len(), 1);

        if let TranscriptEntry::Checkpoint {
            snapshots,
            has_shell_exec,
            ..
        } = &checkpoint_entries[0]
        {
            assert_eq!(snapshots.len(), 1);
            assert_eq!(snapshots[0].pre_hash, checkpoint::CREATED_SENTINEL);
            assert!(!snapshots[0].post_hash.is_empty());
            assert!(!has_shell_exec);
        } else {
            panic!("Expected Checkpoint");
        }
    }

    #[tokio::test]
    async fn checkpoint_emitted_for_file_edit() {
        let repo = setup_git_repo();
        let file_path = repo.path().join("test.txt");
        std::fs::write(&file_path, "old content here").unwrap();

        let mock = MockGrokApi::new(vec![
            TurnResponse {
                text: String::new(),
                tool_calls: vec![ToolCall {
                    call_id: "call_1".into(),
                    name: "file_edit".into(),
                    arguments: format!(
                        r#"{{"path": "{}", "old_string": "old content", "new_string": "new content"}}"#,
                        file_path.display()
                    ),
                }],
                usage: None,
                reasoning_content: None,
                encrypted_reasoning: None,
            },
            TurnResponse {
                text: "Edited.".into(),
                tool_calls: vec![],
                usage: None,
                reasoning_content: None,
                encrypted_reasoning: None,
            },
        ]);

        let agent = Agent::new(&mock, repo.path());
        let input = vec![
            json!({"role": "system", "content": "sys"}),
            json!({"role": "user", "content": "edit"}),
        ];
        let (mut on_entry, collected) = collecting_entries();

        agent
            .run(
                input,
                &mut noop_token,
                &mut noop_tool_call,
                &mut noop_tool_result,
                &mut allow_all,
                &mut no_refresh,
                &mut on_entry,
                &mut noop_reasoning,
            )
            .await
            .unwrap();
        drop(on_entry);

        let entries = collected.lock().unwrap();
        let checkpoint_entries: Vec<_> = entries
            .iter()
            .filter(|e| matches!(e, TranscriptEntry::Checkpoint { .. }))
            .collect();
        assert_eq!(checkpoint_entries.len(), 1);

        if let TranscriptEntry::Checkpoint { snapshots, .. } = &checkpoint_entries[0] {
            assert_eq!(snapshots.len(), 1);
            assert_ne!(snapshots[0].pre_hash, checkpoint::CREATED_SENTINEL);
            assert_ne!(snapshots[0].pre_hash, snapshots[0].post_hash);
        } else {
            panic!("Expected Checkpoint");
        }
    }

    #[tokio::test]
    async fn no_checkpoint_for_read_only_tools() {
        let repo = setup_git_repo();
        let file_path = repo.path().join("test.txt");
        std::fs::write(&file_path, "content").unwrap();

        let mock = MockGrokApi::new(vec![
            TurnResponse {
                text: String::new(),
                tool_calls: vec![ToolCall {
                    call_id: "call_1".into(),
                    name: "file_read".into(),
                    arguments: format!(r#"{{"path": "{}"}}"#, file_path.display()),
                }],
                usage: None,
                reasoning_content: None,
                encrypted_reasoning: None,
            },
            TurnResponse {
                text: "Read.".into(),
                tool_calls: vec![],
                usage: None,
                reasoning_content: None,
                encrypted_reasoning: None,
            },
        ]);

        let agent = Agent::new(&mock, repo.path());
        let input = vec![json!({"role": "user", "content": "read"})];
        let (mut on_entry, collected) = collecting_entries();

        agent
            .run(
                input,
                &mut noop_token,
                &mut noop_tool_call,
                &mut noop_tool_result,
                &mut allow_all,
                &mut no_refresh,
                &mut on_entry,
                &mut noop_reasoning,
            )
            .await
            .unwrap();
        drop(on_entry);

        let entries = collected.lock().unwrap();
        let checkpoint_entries: Vec<_> = entries
            .iter()
            .filter(|e| matches!(e, TranscriptEntry::Checkpoint { .. }))
            .collect();
        assert_eq!(checkpoint_entries.len(), 0);
    }

    #[tokio::test]
    async fn no_checkpoint_when_tool_execution_fails() {
        let repo = setup_git_repo();
        let file_path = repo.path().join("test.txt");
        // file_edit on a non-existent file will fail (old_string not found)

        let mock = MockGrokApi::new(vec![
            TurnResponse {
                text: String::new(),
                tool_calls: vec![ToolCall {
                    call_id: "call_1".into(),
                    name: "file_edit".into(),
                    arguments: format!(
                        r#"{{"path": "{}", "old_string": "nonexistent", "new_string": "x"}}"#,
                        file_path.display()
                    ),
                }],
                usage: None,
                reasoning_content: None,
                encrypted_reasoning: None,
            },
            TurnResponse {
                text: "Edit failed.".into(),
                tool_calls: vec![],
                usage: None,
                reasoning_content: None,
                encrypted_reasoning: None,
            },
        ]);

        let agent = Agent::new(&mock, repo.path());
        let input = vec![json!({"role": "user", "content": "edit"})];
        let (mut on_entry, collected) = collecting_entries();

        agent
            .run(
                input,
                &mut noop_token,
                &mut noop_tool_call,
                &mut noop_tool_result,
                &mut allow_all,
                &mut no_refresh,
                &mut on_entry,
                &mut noop_reasoning,
            )
            .await
            .unwrap();
        drop(on_entry);

        // Tool failed — no checkpoint should be emitted
        let entries = collected.lock().unwrap();
        let checkpoint_entries: Vec<_> = entries
            .iter()
            .filter(|e| matches!(e, TranscriptEntry::Checkpoint { .. }))
            .collect();
        assert_eq!(checkpoint_entries.len(), 0);
    }

    #[tokio::test]
    async fn no_checkpoint_when_permission_denied() {
        let repo = setup_git_repo();
        let file_path = repo.path().join("test.txt");

        let mock = MockGrokApi::new(vec![
            TurnResponse {
                text: String::new(),
                tool_calls: vec![ToolCall {
                    call_id: "call_1".into(),
                    name: "file_write".into(),
                    arguments: format!(r#"{{"path": "{}", "content": "x"}}"#, file_path.display()),
                }],
                usage: None,
                reasoning_content: None,
                encrypted_reasoning: None,
            },
            TurnResponse {
                text: "Denied.".into(),
                tool_calls: vec![],
                usage: None,
                reasoning_content: None,
                encrypted_reasoning: None,
            },
        ]);

        let agent = Agent::new(&mock, repo.path());
        let input = vec![json!({"role": "user", "content": "write"})];
        let mut deny_all = |_: &str, _: &str| -> bool { false };
        let (mut on_entry, collected) = collecting_entries();

        agent
            .run(
                input,
                &mut noop_token,
                &mut noop_tool_call,
                &mut noop_tool_result,
                &mut deny_all,
                &mut no_refresh,
                &mut on_entry,
                &mut noop_reasoning,
            )
            .await
            .unwrap();
        drop(on_entry);

        let entries = collected.lock().unwrap();
        let checkpoint_entries: Vec<_> = entries
            .iter()
            .filter(|e| matches!(e, TranscriptEntry::Checkpoint { .. }))
            .collect();
        assert_eq!(checkpoint_entries.len(), 0);
    }

    #[tokio::test]
    async fn no_checkpoint_for_out_of_project_path() {
        let repo = setup_git_repo();
        // Create a file outside the project root
        let outside_dir = tempfile::tempdir().unwrap();
        let outside_file = outside_dir.path().join("secret.txt");
        std::fs::write(&outside_file, "secret data").unwrap();

        let mock = MockGrokApi::new(vec![
            TurnResponse {
                text: String::new(),
                tool_calls: vec![ToolCall {
                    call_id: "call_1".into(),
                    name: "file_write".into(),
                    arguments: format!(
                        r#"{{"path": "{}", "content": "overwrite"}}"#,
                        outside_file.display()
                    ),
                }],
                usage: None,
                reasoning_content: None,
                encrypted_reasoning: None,
            },
            TurnResponse {
                text: "Done.".into(),
                tool_calls: vec![],
                usage: None,
                reasoning_content: None,
                encrypted_reasoning: None,
            },
        ]);

        let agent = Agent::new(&mock, repo.path());
        let input = vec![json!({"role": "user", "content": "write outside"})];
        let (mut on_entry, collected) = collecting_entries();

        agent
            .run(
                input,
                &mut noop_token,
                &mut noop_tool_call,
                &mut noop_tool_result,
                &mut allow_all,
                &mut no_refresh,
                &mut on_entry,
                &mut noop_reasoning,
            )
            .await
            .unwrap();
        drop(on_entry);

        // Should have no checkpoint — path is outside project root
        let entries = collected.lock().unwrap();
        let checkpoint_entries: Vec<_> = entries
            .iter()
            .filter(|e| matches!(e, TranscriptEntry::Checkpoint { .. }))
            .collect();
        assert_eq!(checkpoint_entries.len(), 0);
    }

    #[tokio::test]
    async fn checkpoint_deduplicates_same_file_edits() {
        let repo = setup_git_repo();
        let file_path = repo.path().join("test.txt");
        std::fs::write(&file_path, "original content").unwrap();

        let mock = MockGrokApi::new(vec![
            TurnResponse {
                text: String::new(),
                tool_calls: vec![
                    ToolCall {
                        call_id: "call_1".into(),
                        name: "file_edit".into(),
                        arguments: format!(
                            r#"{{"path": "{}", "old_string": "original", "new_string": "first edit"}}"#,
                            file_path.display()
                        ),
                    },
                    ToolCall {
                        call_id: "call_2".into(),
                        name: "file_edit".into(),
                        arguments: format!(
                            r#"{{"path": "{}", "old_string": "first edit", "new_string": "second edit"}}"#,
                            file_path.display()
                        ),
                    },
                ],
                usage: None,
                reasoning_content: None,
                encrypted_reasoning: None,
            },
            TurnResponse {
                text: "Edited twice.".into(),
                tool_calls: vec![],
                usage: None,
                reasoning_content: None,
                encrypted_reasoning: None,
            },
        ]);

        let agent = Agent::new(&mock, repo.path());
        let input = vec![
            json!({"role": "system", "content": "sys"}),
            json!({"role": "user", "content": "edit"}),
        ];
        let (mut on_entry, collected) = collecting_entries();

        agent
            .run(
                input,
                &mut noop_token,
                &mut noop_tool_call,
                &mut noop_tool_result,
                &mut allow_all,
                &mut no_refresh,
                &mut on_entry,
                &mut noop_reasoning,
            )
            .await
            .unwrap();
        drop(on_entry);

        let entries = collected.lock().unwrap();
        let checkpoint_entries: Vec<_> = entries
            .iter()
            .filter(|e| matches!(e, TranscriptEntry::Checkpoint { .. }))
            .collect();
        // Each file-modifying tool gets its own checkpoint (persisted immediately
        // after the file change for durability). No batched dedup needed — rewind
        // processes checkpoints in reverse order.
        assert_eq!(checkpoint_entries.len(), 2);

        // First checkpoint: original → first edit
        if let TranscriptEntry::Checkpoint { snapshots, .. } = &checkpoint_entries[0] {
            assert_eq!(snapshots.len(), 1);
            let restored =
                checkpoint::git_cat_file_blob(&snapshots[0].pre_hash, repo.path()).unwrap();
            assert_eq!(String::from_utf8(restored).unwrap(), "original content");
        }

        // Second checkpoint: first edit → second edit
        if let TranscriptEntry::Checkpoint { snapshots, .. } = &checkpoint_entries[1] {
            assert_eq!(snapshots.len(), 1);
            let restored =
                checkpoint::git_cat_file_blob(&snapshots[0].pre_hash, repo.path()).unwrap();
            assert_eq!(String::from_utf8(restored).unwrap(), "first edit content");
        }

        // File currently has "second edit content"
        let current = std::fs::read_to_string(&file_path).unwrap();
        assert_eq!(current, "second edit content");
    }

    #[tokio::test]
    async fn shell_exec_detected_by_rewind_when_after_checkpoint() {
        // file_write then shell_exec in same batch. Checkpoint is emitted
        // immediately after file_write — before shell_exec runs — so the
        // checkpoint's has_shell_exec is false. Rewind derives the warning
        // from ToolCall entries in the removed range.
        let repo = setup_git_repo();
        let file_path = repo.path().join("test.txt");

        let mock = MockGrokApi::new(vec![
            TurnResponse {
                text: String::new(),
                tool_calls: vec![
                    ToolCall {
                        call_id: "call_1".into(),
                        name: "file_write".into(),
                        arguments: format!(
                            r#"{{"path": "{}", "content": "hello"}}"#,
                            file_path.display()
                        ),
                    },
                    ToolCall {
                        call_id: "call_2".into(),
                        name: "shell_exec".into(),
                        arguments: r#"{"command": "echo hi"}"#.into(),
                    },
                ],
                usage: None,
                reasoning_content: None,
                encrypted_reasoning: None,
            },
            TurnResponse {
                text: "Done.".into(),
                tool_calls: vec![],
                usage: None,
                reasoning_content: None,
                encrypted_reasoning: None,
            },
        ]);

        let agent = Agent::new(&mock, repo.path());
        let input = vec![
            json!({"role": "system", "content": "sys"}),
            json!({"role": "user", "content": "do it"}),
        ];
        let (mut on_entry, collected) = collecting_entries();

        agent
            .run(
                input,
                &mut noop_token,
                &mut noop_tool_call,
                &mut noop_tool_result,
                &mut allow_all,
                &mut no_refresh,
                &mut on_entry,
                &mut noop_reasoning,
            )
            .await
            .unwrap();
        drop(on_entry);

        let entries = collected.lock().unwrap();
        let checkpoint_entries: Vec<_> = entries
            .iter()
            .filter(|e| matches!(e, TranscriptEntry::Checkpoint { .. }))
            .collect();
        assert_eq!(checkpoint_entries.len(), 1);

        // Checkpoint was persisted before shell_exec executed
        if let TranscriptEntry::Checkpoint { has_shell_exec, .. } = &checkpoint_entries[0] {
            assert!(!has_shell_exec, "checkpoint emitted before shell_exec runs");
        }

        // Rewind derives the warning from ToolCall entries
        let has_shell_tool = entries
            .iter()
            .any(|e| matches!(e, TranscriptEntry::ToolCall { name, .. } if name == "shell_exec"));
        assert!(has_shell_tool);
    }

    #[tokio::test]
    async fn checkpoint_new_file_in_new_directory() {
        let repo = setup_git_repo();
        // File in a directory that doesn't exist yet
        let file_path = repo.path().join("new_dir").join("nested").join("file.txt");

        let mock = MockGrokApi::new(vec![
            TurnResponse {
                text: String::new(),
                tool_calls: vec![ToolCall {
                    call_id: "call_1".into(),
                    name: "file_write".into(),
                    arguments: format!(
                        r#"{{"path": "{}", "content": "hello"}}"#,
                        file_path.display()
                    ),
                }],
                usage: None,
                reasoning_content: None,
                encrypted_reasoning: None,
            },
            TurnResponse {
                text: "Created.".into(),
                tool_calls: vec![],
                usage: None,
                reasoning_content: None,
                encrypted_reasoning: None,
            },
        ]);

        let agent = Agent::new(&mock, repo.path());
        let input = vec![
            json!({"role": "system", "content": "sys"}),
            json!({"role": "user", "content": "create"}),
        ];
        let (mut on_entry, collected) = collecting_entries();

        agent
            .run(
                input,
                &mut noop_token,
                &mut noop_tool_call,
                &mut noop_tool_result,
                &mut allow_all,
                &mut no_refresh,
                &mut on_entry,
                &mut noop_reasoning,
            )
            .await
            .unwrap();
        drop(on_entry);

        let entries = collected.lock().unwrap();
        let checkpoint_entries: Vec<_> = entries
            .iter()
            .filter(|e| matches!(e, TranscriptEntry::Checkpoint { .. }))
            .collect();
        assert_eq!(
            checkpoint_entries.len(),
            1,
            "new file in new dir should be checkpointed"
        );

        if let TranscriptEntry::Checkpoint { snapshots, .. } = &checkpoint_entries[0] {
            assert_eq!(snapshots.len(), 1);
            assert_eq!(snapshots[0].pre_hash, checkpoint::CREATED_SENTINEL);
        }
    }

    #[tokio::test]
    async fn shell_exec_warning_across_iterations() {
        // Iteration 1: shell_exec only (no checkpoint emitted)
        // Iteration 2: file_write (checkpoint emitted)
        // The checkpoint should carry has_shell_exec = true because
        // shell_exec happened before the checkpoint was emitted
        let repo = setup_git_repo();
        let file_path = repo.path().join("test.txt");

        let mock = MockGrokApi::new(vec![
            // Iteration 1: shell_exec only
            TurnResponse {
                text: String::new(),
                tool_calls: vec![ToolCall {
                    call_id: "call_1".into(),
                    name: "shell_exec".into(),
                    arguments: r#"{"command": "echo hi"}"#.into(),
                }],
                usage: None,
                reasoning_content: None,
                encrypted_reasoning: None,
            },
            // Iteration 2: file_write
            TurnResponse {
                text: String::new(),
                tool_calls: vec![ToolCall {
                    call_id: "call_2".into(),
                    name: "file_write".into(),
                    arguments: format!(
                        r#"{{"path": "{}", "content": "data"}}"#,
                        file_path.display()
                    ),
                }],
                usage: None,
                reasoning_content: None,
                encrypted_reasoning: None,
            },
            TurnResponse {
                text: "Done.".into(),
                tool_calls: vec![],
                usage: None,
                reasoning_content: None,
                encrypted_reasoning: None,
            },
        ]);

        let agent = Agent::new(&mock, repo.path());
        let input = vec![
            json!({"role": "system", "content": "sys"}),
            json!({"role": "user", "content": "go"}),
        ];
        let (mut on_entry, collected) = collecting_entries();

        agent
            .run(
                input,
                &mut noop_token,
                &mut noop_tool_call,
                &mut noop_tool_result,
                &mut allow_all,
                &mut no_refresh,
                &mut on_entry,
                &mut noop_reasoning,
            )
            .await
            .unwrap();
        drop(on_entry);

        let entries = collected.lock().unwrap();
        let checkpoint_entries: Vec<_> = entries
            .iter()
            .filter(|e| matches!(e, TranscriptEntry::Checkpoint { .. }))
            .collect();
        assert_eq!(checkpoint_entries.len(), 1);

        if let TranscriptEntry::Checkpoint { has_shell_exec, .. } = &checkpoint_entries[0] {
            assert!(
                has_shell_exec,
                "checkpoint should flag shell_exec from earlier iteration"
            );
        }
    }

    #[tokio::test]
    async fn shell_exec_after_checkpoint_detected_by_rewind() {
        // Iteration 1: file_write (checkpoint emitted with has_shell_exec=false)
        // Iteration 2: shell_exec only (no new checkpoint)
        // With incremental streaming, the checkpoint can't be retroactively patched.
        // Rewind derives shell_exec by scanning ToolCall entries instead.
        let repo = setup_git_repo();
        let file_path = repo.path().join("test.txt");

        let mock = MockGrokApi::new(vec![
            // Iteration 1: file_write
            TurnResponse {
                text: String::new(),
                tool_calls: vec![ToolCall {
                    call_id: "call_1".into(),
                    name: "file_write".into(),
                    arguments: format!(
                        r#"{{"path": "{}", "content": "data"}}"#,
                        file_path.display()
                    ),
                }],
                usage: None,
                reasoning_content: None,
                encrypted_reasoning: None,
            },
            // Iteration 2: shell_exec only
            TurnResponse {
                text: String::new(),
                tool_calls: vec![ToolCall {
                    call_id: "call_2".into(),
                    name: "shell_exec".into(),
                    arguments: r#"{"command": "echo side-effect"}"#.into(),
                }],
                usage: None,
                reasoning_content: None,
                encrypted_reasoning: None,
            },
            TurnResponse {
                text: "Done.".into(),
                tool_calls: vec![],
                usage: None,
                reasoning_content: None,
                encrypted_reasoning: None,
            },
        ]);

        let agent = Agent::new(&mock, repo.path());
        let input = vec![
            json!({"role": "system", "content": "sys"}),
            json!({"role": "user", "content": "go"}),
        ];
        let (mut on_entry, collected) = collecting_entries();

        agent
            .run(
                input,
                &mut noop_token,
                &mut noop_tool_call,
                &mut noop_tool_result,
                &mut allow_all,
                &mut no_refresh,
                &mut on_entry,
                &mut noop_reasoning,
            )
            .await
            .unwrap();
        drop(on_entry);

        let entries = collected.lock().unwrap();

        // Checkpoint was emitted before shell_exec, so has_shell_exec is false
        let checkpoint_entries: Vec<_> = entries
            .iter()
            .filter(|e| matches!(e, TranscriptEntry::Checkpoint { .. }))
            .collect();
        assert_eq!(checkpoint_entries.len(), 1);
        if let TranscriptEntry::Checkpoint { has_shell_exec, .. } = &checkpoint_entries[0] {
            assert!(
                !has_shell_exec,
                "checkpoint was emitted before shell_exec, so flag is false"
            );
        }

        // But rewind should still detect shell_exec by scanning ToolCall entries
        let has_shell_exec_tool = entries
            .iter()
            .any(|e| matches!(e, TranscriptEntry::ToolCall { name, .. } if name == "shell_exec"));
        assert!(
            has_shell_exec_tool,
            "shell_exec ToolCall should be in the entries"
        );
    }

    #[tokio::test]
    async fn checkpoint_equivalent_paths_each_get_own_checkpoint() {
        // Edit the same file via two different path representations.
        // Each gets its own checkpoint (persisted immediately for durability).
        // Rewind processes them in reverse order to restore correctly.
        let repo = setup_git_repo();
        let file_path = repo.path().join("test.txt");
        std::fs::write(&file_path, "original content here").unwrap();

        // Use relative path first, then ./relative path
        let rel_path1 = "test.txt";
        let rel_path2 = "./test.txt";

        let mock = MockGrokApi::new(vec![
            TurnResponse {
                text: String::new(),
                tool_calls: vec![
                    ToolCall {
                        call_id: "call_1".into(),
                        name: "file_edit".into(),
                        arguments: format!(
                            r#"{{"path": "{}", "old_string": "original", "new_string": "first"}}"#,
                            rel_path1
                        ),
                    },
                    ToolCall {
                        call_id: "call_2".into(),
                        name: "file_edit".into(),
                        arguments: format!(
                            r#"{{"path": "{}", "old_string": "first", "new_string": "second"}}"#,
                            rel_path2
                        ),
                    },
                ],
                usage: None,
                reasoning_content: None,
                encrypted_reasoning: None,
            },
            TurnResponse {
                text: "Edited.".into(),
                tool_calls: vec![],
                usage: None,
                reasoning_content: None,
                encrypted_reasoning: None,
            },
        ]);

        let agent = Agent::new(&mock, repo.path());
        let input = vec![
            json!({"role": "system", "content": "sys"}),
            json!({"role": "user", "content": "edit"}),
        ];
        let (mut on_entry, collected) = collecting_entries();

        agent
            .run(
                input,
                &mut noop_token,
                &mut noop_tool_call,
                &mut noop_tool_result,
                &mut allow_all,
                &mut no_refresh,
                &mut on_entry,
                &mut noop_reasoning,
            )
            .await
            .unwrap();
        drop(on_entry);

        let entries = collected.lock().unwrap();
        let checkpoint_entries: Vec<_> = entries
            .iter()
            .filter(|e| matches!(e, TranscriptEntry::Checkpoint { .. }))
            .collect();
        // Two checkpoints: one per edit (immediate persistence, no batched dedup)
        assert_eq!(checkpoint_entries.len(), 2);

        // First checkpoint: original → first edit
        if let TranscriptEntry::Checkpoint { snapshots, .. } = &checkpoint_entries[0] {
            assert_eq!(snapshots.len(), 1);
            let restored =
                checkpoint::git_cat_file_blob(&snapshots[0].pre_hash, repo.path()).unwrap();
            assert_eq!(
                String::from_utf8(restored).unwrap(),
                "original content here"
            );
        }
    }

    // --- Durability regression test ---

    #[tokio::test]
    async fn checkpoint_persisted_despite_later_api_failure() {
        // Regression test: iteration 1 does file_write (checkpoint emitted via
        // on_entry), iteration 2's send_turn fails. The on_entry callback should
        // have received the checkpoint despite agent returning Err.
        let repo = setup_git_repo();
        let file_path = repo.path().join("test.txt");
        std::fs::write(&file_path, "original content").unwrap();

        // Only one response: file_edit tool call. Second send_turn will fail
        // because MockGrokApi has no more scripted responses.
        let mock = MockGrokApi::new(vec![
            TurnResponse {
                text: String::new(),
                tool_calls: vec![ToolCall {
                    call_id: "call_1".into(),
                    name: "file_edit".into(),
                    arguments: format!(
                        r#"{{"path": "{}", "old_string": "original", "new_string": "modified"}}"#,
                        file_path.display()
                    ),
                }],
                usage: None,
                reasoning_content: None,
                encrypted_reasoning: None,
            },
            // No second response — send_turn will error
        ]);

        let agent = Agent::new(&mock, repo.path());
        let input = vec![
            json!({"role": "system", "content": "sys"}),
            json!({"role": "user", "content": "edit"}),
        ];
        let (mut on_entry, collected) = collecting_entries();

        // Agent should return Err because second send_turn fails
        let result = agent
            .run(
                input,
                &mut noop_token,
                &mut noop_tool_call,
                &mut noop_tool_result,
                &mut allow_all,
                &mut no_refresh,
                &mut on_entry,
                &mut noop_reasoning,
            )
            .await;
        drop(on_entry);

        assert!(
            result.is_err(),
            "agent should fail when send_turn has no more responses"
        );

        // Despite the error, on_entry should have received entries including a checkpoint
        let entries = collected.lock().unwrap();
        let checkpoint_entries: Vec<_> = entries
            .iter()
            .filter(|e| matches!(e, TranscriptEntry::Checkpoint { .. }))
            .collect();
        assert_eq!(
            checkpoint_entries.len(),
            1,
            "checkpoint should be emitted via on_entry before the API failure"
        );

        // Verify the checkpoint has valid snapshot data
        if let TranscriptEntry::Checkpoint { snapshots, .. } = &checkpoint_entries[0] {
            assert_eq!(snapshots.len(), 1);
            assert_ne!(snapshots[0].pre_hash, snapshots[0].post_hash);
        }

        // Verify the file was actually modified (tool executed successfully)
        assert_eq!(
            std::fs::read_to_string(&file_path).unwrap(),
            "modified content"
        );

        // Build a full history as main.rs would: user message + streamed entries
        let mut history: Vec<TranscriptEntry> = Vec::new();
        history.push(TranscriptEntry::user_message("edit"));
        history.extend(entries.iter().cloned());

        // /undo should be able to restore from the checkpoint
        let undo_result =
            crate::rewind::undo_last_turn(&history, repo.path(), crate::rewind::RewindMode::Both)
                .unwrap();

        assert_eq!(
            std::fs::read_to_string(&file_path).unwrap(),
            "original content"
        );
        assert!(!undo_result.file_results.is_empty());
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
                reasoning_content: None,
                encrypted_reasoning: None,
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
                reasoning_content: None,
                encrypted_reasoning: None,
            },
            // Iteration 3: model responds with text (no more tools)
            TurnResponse {
                text: "All done.".into(),
                tool_calls: vec![],
                usage: None,
                reasoning_content: None,
                encrypted_reasoning: None,
            },
        ]);

        let agent = Agent::new(&mock, std::path::Path::new("/tmp"));
        // Initial input: system prompt + user message = 2 messages
        let input = vec![
            json!({"role": "system", "content": "sys"}),
            json!({"role": "user", "content": "read twice"}),
        ];

        let result = agent
            .run(
                input,
                &mut noop_token,
                &mut noop_tool_call,
                &mut noop_tool_result,
                &mut allow_all,
                &mut no_refresh,
                &mut noop_entry,
                &mut noop_reasoning,
            )
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

    // --- Reasoning regression tests ---

    /// Helper that captures on_reasoning callback invocations.
    #[allow(clippy::type_complexity)]
    fn capturing_reasoning() -> (
        impl FnMut(Option<&str>, Option<&str>, Option<u64>) + Send,
        std::sync::Arc<std::sync::Mutex<Vec<(Option<String>, Option<String>, Option<u64>)>>>,
    ) {
        let records = std::sync::Arc::new(std::sync::Mutex::new(Vec::new()));
        let records_clone = records.clone();
        let callback =
            move |plaintext: Option<&str>, encrypted: Option<&str>, tokens: Option<u64>| {
                records_clone.lock().unwrap().push((
                    plaintext.map(|s| s.to_string()),
                    encrypted.map(|s| s.to_string()),
                    tokens,
                ));
            };
        (callback, records)
    }

    #[tokio::test]
    async fn intermediate_reasoning_round_tripped_in_messages() {
        // Verifies that when an intermediate tool-using turn includes reasoning
        // content, the reasoning fields are present in the messages sent to the
        // API on the next iteration.
        let tmp = tempfile::NamedTempFile::new().unwrap();
        std::io::Write::write_all(&mut &tmp, b"content").unwrap();
        let path = tmp.path().to_str().unwrap();

        let mock = CapturingMockGrokApi::new(vec![
            // Iteration 1: model reasons, then calls a tool
            TurnResponse {
                text: "Let me read that.".into(),
                tool_calls: vec![ToolCall {
                    call_id: "call_1".into(),
                    name: "file_read".into(),
                    arguments: format!(r#"{{"path": "{}"}}"#, path),
                }],
                usage: None,
                reasoning_content: Some("I should read the file first.".into()),
                encrypted_reasoning: None,
            },
            // Iteration 2: final response
            TurnResponse {
                text: "Done.".into(),
                tool_calls: vec![],
                usage: None,
                reasoning_content: None,
                encrypted_reasoning: None,
            },
        ]);

        let agent = Agent::new(&mock, std::path::Path::new("/tmp"));
        let input = vec![json!({"role": "user", "content": "read it"})];

        agent
            .run(
                input,
                &mut noop_token,
                &mut noop_tool_call,
                &mut noop_tool_result,
                &mut allow_all,
                &mut no_refresh,
                &mut noop_entry,
                &mut noop_reasoning,
            )
            .await
            .unwrap();

        // Check the second API call's input for reasoning in the assistant message
        let inputs = mock.captured_inputs.lock().unwrap();
        assert_eq!(inputs.len(), 2, "should have made 2 API calls");

        // Find the assistant message in the second call's input
        let second_call = &inputs[1];
        let assistant_msg = second_call
            .iter()
            .find(|m| m.get("role").and_then(|r| r.as_str()) == Some("assistant"))
            .expect("second call should include an assistant message from iteration 1");

        assert_eq!(
            assistant_msg
                .get("reasoning_content")
                .and_then(|v| v.as_str()),
            Some("I should read the file first."),
            "intermediate reasoning_content must be round-tripped"
        );
    }

    #[tokio::test]
    async fn intermediate_encrypted_reasoning_round_tripped_in_messages() {
        // Same as above but for encrypted reasoning (grok-4 style).
        let tmp = tempfile::NamedTempFile::new().unwrap();
        std::io::Write::write_all(&mut &tmp, b"content").unwrap();
        let path = tmp.path().to_str().unwrap();

        let mock = CapturingMockGrokApi::new(vec![
            TurnResponse {
                text: "Let me check.".into(),
                tool_calls: vec![ToolCall {
                    call_id: "call_1".into(),
                    name: "file_read".into(),
                    arguments: format!(r#"{{"path": "{}"}}"#, path),
                }],
                usage: None,
                reasoning_content: None,
                encrypted_reasoning: Some("opaque-blob-abc123".into()),
            },
            TurnResponse {
                text: "Done.".into(),
                tool_calls: vec![],
                usage: None,
                reasoning_content: None,
                encrypted_reasoning: None,
            },
        ]);

        let agent = Agent::new(&mock, std::path::Path::new("/tmp"));
        let input = vec![json!({"role": "user", "content": "read it"})];

        agent
            .run(
                input,
                &mut noop_token,
                &mut noop_tool_call,
                &mut noop_tool_result,
                &mut allow_all,
                &mut no_refresh,
                &mut noop_entry,
                &mut noop_reasoning,
            )
            .await
            .unwrap();

        let inputs = mock.captured_inputs.lock().unwrap();
        let second_call = &inputs[1];
        let assistant_msg = second_call
            .iter()
            .find(|m| m.get("role").and_then(|r| r.as_str()) == Some("assistant"))
            .expect("second call should include an assistant message");

        assert_eq!(
            assistant_msg
                .get("encrypted_reasoning")
                .and_then(|v| v.as_str()),
            Some("opaque-blob-abc123"),
            "intermediate encrypted_reasoning must be round-tripped"
        );
    }

    #[tokio::test]
    async fn on_reasoning_called_for_intermediate_tool_using_turns() {
        // Verifies the on_reasoning callback fires for intermediate turns
        // (not just the final response).
        let tmp = tempfile::NamedTempFile::new().unwrap();
        std::io::Write::write_all(&mut &tmp, b"content").unwrap();
        let path = tmp.path().to_str().unwrap();

        let mock = MockGrokApi::new(vec![
            // Intermediate turn with reasoning + tool call
            TurnResponse {
                text: String::new(),
                tool_calls: vec![ToolCall {
                    call_id: "call_1".into(),
                    name: "file_read".into(),
                    arguments: format!(r#"{{"path": "{}"}}"#, path),
                }],
                usage: Some(crate::api::Usage {
                    input_tokens: 100,
                    output_tokens: 50,
                    cached_input_tokens: None,
                    reasoning_tokens: Some(30),
                }),
                reasoning_content: Some("Thinking about intermediate step.".into()),
                encrypted_reasoning: None,
            },
            // Final turn with different reasoning
            TurnResponse {
                text: "All done.".into(),
                tool_calls: vec![],
                usage: Some(crate::api::Usage {
                    input_tokens: 200,
                    output_tokens: 75,
                    cached_input_tokens: None,
                    reasoning_tokens: Some(40),
                }),
                reasoning_content: Some("Final thoughts.".into()),
                encrypted_reasoning: None,
            },
        ]);

        let agent = Agent::new(&mock, std::path::Path::new("/tmp"));
        let input = vec![json!({"role": "user", "content": "read it"})];
        let (mut on_reasoning, records) = capturing_reasoning();

        agent
            .run(
                input,
                &mut noop_token,
                &mut noop_tool_call,
                &mut noop_tool_result,
                &mut allow_all,
                &mut no_refresh,
                &mut noop_entry,
                &mut on_reasoning,
            )
            .await
            .unwrap();
        drop(on_reasoning);

        let records = records.lock().unwrap();
        assert_eq!(
            records.len(),
            2,
            "on_reasoning should fire for both intermediate and final turns"
        );

        // First call: intermediate turn reasoning
        assert_eq!(
            records[0].0.as_deref(),
            Some("Thinking about intermediate step.")
        );
        assert_eq!(records[0].2, Some(30));

        // Second call: final turn reasoning
        assert_eq!(records[1].0.as_deref(), Some("Final thoughts."));
        assert_eq!(records[1].2, Some(40));
    }
}
