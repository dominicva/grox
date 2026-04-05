use std::path::{Path, PathBuf};

use anyhow::Result;
use serde_json::{Value, json};

use crate::api::GrokApi;
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

    /// Run the agent loop for a single user message.
    /// Returns the final text response and optional response_id for continuation.
    ///
    /// `on_authorize` is called before executing each tool. It receives the tool name
    /// and arguments, and returns true if the tool should be executed.
    pub async fn run(
        &self,
        input: Vec<Value>,
        previous_response_id: Option<&str>,
        on_token: &mut (dyn FnMut(String) + Send),
        on_tool_call: &mut (dyn FnMut(&str, &str) + Send),
        on_tool_result: &mut (dyn FnMut(&str, &str) + Send),
        on_authorize: &mut (dyn FnMut(&str, &str) -> bool + Send),
    ) -> Result<AgentResult> {
        let mut current_input = input;
        let mut current_response_id = previous_response_id.map(String::from);
        let mut final_text = String::new();

        for _turn in 0..MAX_TURNS {
            let response = self
                .api
                .send_turn(
                    current_input.clone(),
                    &self.tool_defs,
                    current_response_id.as_deref(),
                    on_token,
                )
                .await?;

            final_text = response.text.clone();
            current_response_id = response.response_id.clone();

            if response.tool_calls.is_empty() {
                return Ok(AgentResult {
                    text: final_text,
                    response_id: current_response_id,
                    usage: response.usage,
                });
            }

            // Execute tool calls and build next input
            let mut tool_results: Vec<Value> = Vec::new();

            for tc in &response.tool_calls {
                on_tool_call(&tc.name, &tc.arguments);

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

                tool_results.push(json!({
                    "type": "function_call_output",
                    "call_id": tc.call_id,
                    "output": output,
                }));
            }

            current_input = tool_results;
        }

        // Hit max turns
        Ok(AgentResult {
            text: final_text,
            response_id: current_response_id,
            usage: None,
        })
    }
}

#[derive(Debug)]
pub struct AgentResult {
    pub text: String,
    pub response_id: Option<String>,
    pub usage: Option<crate::api::Usage>,
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
            .run(input, None, &mut noop_token, &mut noop_tool_call, &mut noop_tool_result, &mut allow_all)
            .await
            .unwrap();

        assert_eq!(result.text, "The file contains a main function.");
        assert_eq!(result.response_id, Some("resp_2".into()));
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
            .run(input, None, &mut noop_token, &mut noop_tool_call, &mut noop_tool_result, &mut allow_all)
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
            .run(input, None, &mut noop_token, &mut noop_tool_call, &mut noop_tool_result, &mut allow_all)
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
            .run(input, None, &mut noop_token, &mut noop_tool_call, &mut noop_tool_result, &mut allow_all)
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
            .run(input, None, &mut noop_token, &mut noop_tool_call, &mut noop_tool_result, &mut allow_all)
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
            .run(input, None, &mut noop_token, &mut noop_tool_call, &mut noop_tool_result, &mut deny_all)
            .await
            .unwrap();

        assert_eq!(result.text, "Write was denied.");
    }
}
