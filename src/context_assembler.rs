use serde_json::{Value, json};

use crate::session::TranscriptEntry;

/// Builds the message array for the xAI Responses API from transcript entries.
///
/// Converts `TranscriptEntry` variants into the correct wire format:
/// - `UserMessage` → `{"role": "user", "content": "..."}`
/// - `AssistantMessage` → `{"role": "assistant", "content": "..."}`
/// - `ToolCall` → `{"type": "function_call", "name": "...", "arguments": "...", "call_id": "..."}`
/// - `ToolResult` → `{"type": "function_call_output", "call_id": "...", "output": "..."}`
/// - `CompactionSummary` → `{"role": "system", "content": "[Context summary]\n\n..."}`
/// - `SystemEvent` → skipped (internal bookkeeping, not sent to model)
pub struct ContextAssembler {
    system_prompt: Value,
    /// Estimated tokens for system-layer overhead (system prompt + repo context + GROX.md).
    system_overhead_estimate: usize,
}

impl ContextAssembler {
    /// Create a new assembler with the given system prompt.
    pub fn new(system_prompt: Value) -> Self {
        let overhead = system_prompt
            .get("content")
            .and_then(|c| c.as_str())
            .map(|s| s.len() / 4)
            .unwrap_or(0);
        Self {
            system_prompt,
            system_overhead_estimate: overhead,
        }
    }

    /// Update the system prompt (e.g. after repo context refresh).
    pub fn set_system_prompt(&mut self, system_prompt: Value) {
        self.system_overhead_estimate = system_prompt
            .get("content")
            .and_then(|c| c.as_str())
            .map(|s| s.len() / 4)
            .unwrap_or(0);
        self.system_prompt = system_prompt;
    }

    /// Build the message array for the API call.
    ///
    /// Order: system prompt → compaction summary (if any) → conversation history.
    pub fn build_messages(&self, entries: &[TranscriptEntry]) -> Vec<Value> {
        let mut messages = Vec::new();
        messages.push(self.system_prompt.clone());

        for entry in entries {
            match entry {
                TranscriptEntry::UserMessage { content, .. } => {
                    messages.push(json!({
                        "role": "user",
                        "content": content,
                    }));
                }
                TranscriptEntry::AssistantMessage { content, .. } => {
                    messages.push(json!({
                        "role": "assistant",
                        "content": content,
                    }));
                }
                TranscriptEntry::ToolCall {
                    call_id,
                    name,
                    arguments,
                    ..
                } => {
                    messages.push(json!({
                        "type": "function_call",
                        "name": name,
                        "arguments": arguments,
                        "call_id": call_id,
                    }));
                }
                TranscriptEntry::ToolResult {
                    call_id, output, ..
                } => {
                    messages.push(json!({
                        "type": "function_call_output",
                        "call_id": call_id,
                        "output": output,
                    }));
                }
                TranscriptEntry::CompactionSummary { summary, .. } => {
                    messages.push(json!({
                        "role": "system",
                        "content": format!("[Context summary from earlier in this conversation]\n\n{summary}"),
                    }));
                }
                TranscriptEntry::SystemEvent { .. } | TranscriptEntry::Checkpoint { .. } => {
                    // Internal bookkeeping — not sent to the model
                }
            }
        }

        messages
    }

    /// Estimate the total token count for a potential API request.
    ///
    /// Sums system-layer overhead + per-entry token estimates.
    pub fn estimate_tokens(&self, entries: &[TranscriptEntry]) -> usize {
        let transcript_tokens: usize = entries.iter().map(|e| e.token_estimate()).sum();
        self.system_overhead_estimate + transcript_tokens
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::session::TranscriptEntry;
    use serde_json::json;

    fn test_system_prompt() -> Value {
        json!({"role": "system", "content": "You are a helpful assistant."})
    }

    #[test]
    fn build_messages_starts_with_system_prompt() {
        let assembler = ContextAssembler::new(test_system_prompt());
        let messages = assembler.build_messages(&[]);
        assert_eq!(messages.len(), 1);
        assert_eq!(messages[0]["role"], "system");
        assert_eq!(messages[0]["content"], "You are a helpful assistant.");
    }

    #[test]
    fn user_message_wire_format() {
        let assembler = ContextAssembler::new(test_system_prompt());
        let entries = vec![TranscriptEntry::user_message("Hello!")];
        let messages = assembler.build_messages(&entries);

        assert_eq!(messages.len(), 2);
        assert_eq!(messages[1]["role"], "user");
        assert_eq!(messages[1]["content"], "Hello!");
    }

    #[test]
    fn assistant_message_wire_format() {
        let assembler = ContextAssembler::new(test_system_prompt());
        let entries = vec![TranscriptEntry::assistant_message("Hi there!")];
        let messages = assembler.build_messages(&entries);

        assert_eq!(messages[1]["role"], "assistant");
        assert_eq!(messages[1]["content"], "Hi there!");
    }

    #[test]
    fn tool_call_wire_format() {
        let assembler = ContextAssembler::new(test_system_prompt());
        let entries = vec![TranscriptEntry::tool_call(
            "call_abc",
            "file_read",
            r#"{"path": "src/main.rs"}"#,
        )];
        let messages = assembler.build_messages(&entries);

        assert_eq!(messages[1]["type"], "function_call");
        assert_eq!(messages[1]["name"], "file_read");
        assert_eq!(messages[1]["call_id"], "call_abc");
        assert_eq!(messages[1]["arguments"], r#"{"path": "src/main.rs"}"#);
    }

    #[test]
    fn tool_result_wire_format() {
        let assembler = ContextAssembler::new(test_system_prompt());
        let entries = vec![TranscriptEntry::tool_result(
            "call_abc",
            "file_read",
            "fn main() {}",
        )];
        let messages = assembler.build_messages(&entries);

        assert_eq!(messages[1]["type"], "function_call_output");
        assert_eq!(messages[1]["call_id"], "call_abc");
        assert_eq!(messages[1]["output"], "fn main() {}");
    }

    #[test]
    fn tool_call_result_linkage() {
        let assembler = ContextAssembler::new(test_system_prompt());
        let entries = vec![
            TranscriptEntry::tool_call("call_1", "file_read", r#"{"path":"a.rs"}"#),
            TranscriptEntry::tool_result("call_1", "file_read", "content of a.rs"),
        ];
        let messages = assembler.build_messages(&entries);

        let call_id_from_call = messages[1]["call_id"].as_str().unwrap();
        let call_id_from_result = messages[2]["call_id"].as_str().unwrap();
        assert_eq!(call_id_from_call, call_id_from_result);
        assert_eq!(call_id_from_call, "call_1");
    }

    #[test]
    fn compaction_summary_wire_format() {
        let assembler = ContextAssembler::new(test_system_prompt());
        let entries = vec![TranscriptEntry::compaction_summary(
            "User asked about auth. Files read: src/auth.rs",
        )];
        let messages = assembler.build_messages(&entries);

        assert_eq!(messages[1]["role"], "system");
        let content = messages[1]["content"].as_str().unwrap();
        assert!(content.starts_with("[Context summary"));
        assert!(content.contains("User asked about auth"));
    }

    #[test]
    fn system_event_is_skipped() {
        let assembler = ContextAssembler::new(test_system_prompt());
        let entries = vec![
            TranscriptEntry::system_event("session started"),
            TranscriptEntry::user_message("hello"),
        ];
        let messages = assembler.build_messages(&entries);

        // system prompt + user message only (no system event)
        assert_eq!(messages.len(), 2);
        assert_eq!(messages[1]["role"], "user");
    }

    #[test]
    fn message_ordering_system_then_summary_then_history() {
        let assembler = ContextAssembler::new(test_system_prompt());
        let entries = vec![
            TranscriptEntry::compaction_summary("Earlier conversation summary"),
            TranscriptEntry::user_message("What next?"),
            TranscriptEntry::assistant_message("Let me check."),
        ];
        let messages = assembler.build_messages(&entries);

        assert_eq!(messages.len(), 4);
        assert_eq!(messages[0]["role"], "system"); // system prompt
        assert_eq!(messages[1]["role"], "system"); // compaction summary
        assert_eq!(messages[2]["role"], "user");
        assert_eq!(messages[3]["role"], "assistant");
    }

    #[test]
    fn full_conversation_roundtrip() {
        let assembler = ContextAssembler::new(test_system_prompt());
        let entries = vec![
            TranscriptEntry::user_message("Read main.rs"),
            TranscriptEntry::assistant_message(""),
            TranscriptEntry::tool_call("c1", "file_read", r#"{"path":"src/main.rs"}"#),
            TranscriptEntry::tool_result("c1", "file_read", "fn main() { println!(\"hello\"); }"),
            TranscriptEntry::assistant_message("The main function prints hello."),
        ];
        let messages = assembler.build_messages(&entries);

        assert_eq!(messages.len(), 6); // system + 5 entries
        assert_eq!(messages[0]["role"], "system");
        assert_eq!(messages[1]["role"], "user");
        assert_eq!(messages[2]["role"], "assistant");
        assert_eq!(messages[3]["type"], "function_call");
        assert_eq!(messages[4]["type"], "function_call_output");
        assert_eq!(messages[5]["role"], "assistant");
    }

    #[test]
    fn estimate_tokens_empty_transcript() {
        let assembler = ContextAssembler::new(test_system_prompt());
        let estimate = assembler.estimate_tokens(&[]);
        // Should be just the system overhead
        let expected = "You are a helpful assistant.".len() / 4;
        assert_eq!(estimate, expected);
    }

    #[test]
    fn estimate_tokens_sums_entries_plus_overhead() {
        let assembler = ContextAssembler::new(test_system_prompt());
        let entries = vec![
            TranscriptEntry::user_message("hello world"), // 11/4 = 2
            TranscriptEntry::assistant_message("hi there!"), // 9/4 = 2
        ];
        let estimate = assembler.estimate_tokens(&entries);

        let overhead = "You are a helpful assistant.".len() / 4; // 28/4 = 7
        let entry_sum: usize = entries.iter().map(|e| e.token_estimate()).sum();
        assert_eq!(estimate, overhead + entry_sum);
    }

    #[test]
    fn estimate_tokens_no_double_counting() {
        let assembler = ContextAssembler::new(test_system_prompt());
        let entries = vec![
            TranscriptEntry::user_message("a"),
            TranscriptEntry::user_message("b"),
        ];
        let estimate = assembler.estimate_tokens(&entries);
        let overhead = "You are a helpful assistant.".len() / 4;
        let sum: usize = entries.iter().map(|e| e.token_estimate()).sum();
        assert_eq!(estimate, overhead + sum);
    }

    #[test]
    fn set_system_prompt_updates_overhead() {
        let mut assembler = ContextAssembler::new(test_system_prompt());
        let old_overhead = assembler.system_overhead_estimate;

        let long_prompt = json!({"role": "system", "content": "x".repeat(1000)});
        assembler.set_system_prompt(long_prompt);

        assert_eq!(assembler.system_overhead_estimate, 1000 / 4);
        assert_ne!(assembler.system_overhead_estimate, old_overhead);
    }

    #[test]
    fn build_messages_with_empty_entries() {
        let assembler = ContextAssembler::new(test_system_prompt());
        let messages = assembler.build_messages(&[]);
        assert_eq!(messages.len(), 1); // just system prompt
    }
}
