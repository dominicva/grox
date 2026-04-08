use serde_json::{Value, json};

use crate::session::TranscriptEntry;

/// The separator + header prepended to a compaction summary when inlining it
/// into the system prompt. Must match the format string in `build_messages`.
const SUMMARY_WRAPPER: &str = "\n\n---\n\n[Context summary from earlier in this conversation]\n\n";

/// Builds the message array for the xAI Responses API from transcript entries.
///
/// Converts `TranscriptEntry` variants into the correct wire format:
/// - `UserMessage` → `{"role": "user", "content": "..."}`
/// - `AssistantMessage` → `{"role": "assistant", "content": "..."}`
/// - `ToolCall` → `{"type": "function_call", "name": "...", "arguments": "...", "call_id": "..."}`
/// - `ToolResult` → `{"type": "function_call_output", "call_id": "...", "output": "..."}`
/// - `CompactionSummary` → inlined into the system prompt (only the most recent)
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
    /// The most recent CompactionSummary is appended to the system prompt content
    /// (separated by `---`), producing exactly one system message at position 0.
    /// Earlier summaries are ignored (they were already incorporated by compaction).
    /// CompactionSummary entries are never emitted as separate messages.
    pub fn build_messages(&self, entries: &[TranscriptEntry]) -> Vec<Value> {
        let mut messages = Vec::new();

        // Find the most recent CompactionSummary (if any) and inline it
        let latest_summary = entries.iter().rev().find_map(|e| match e {
            TranscriptEntry::CompactionSummary { summary, .. } => Some(summary.as_str()),
            _ => None,
        });

        if let Some(summary) = latest_summary {
            let base_content = self
                .system_prompt
                .get("content")
                .and_then(|c| c.as_str())
                .unwrap_or("");
            let combined = format!("{base_content}{SUMMARY_WRAPPER}{summary}");
            let mut prompt = self.system_prompt.clone();
            prompt["content"] = Value::String(combined);
            messages.push(prompt);
        } else {
            messages.push(self.system_prompt.clone());
        }

        for entry in entries {
            match entry {
                TranscriptEntry::UserMessage { content, .. } => {
                    messages.push(json!({
                        "role": "user",
                        "content": content,
                    }));
                }
                TranscriptEntry::AssistantMessage {
                    content,
                    reasoning_content,
                    encrypted_reasoning,
                    ..
                } => {
                    let mut msg = json!({
                        "role": "assistant",
                        "content": content,
                    });
                    // Round-trip reasoning payloads so retained turns preserve their
                    // reasoning when resent to the API.
                    if let Some(rc) = reasoning_content {
                        msg["reasoning_content"] = Value::String(rc.clone());
                    }
                    if let Some(er) = encrypted_reasoning {
                        msg["encrypted_reasoning"] = Value::String(er.clone());
                    }
                    messages.push(msg);
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
                TranscriptEntry::CompactionSummary { .. }
                | TranscriptEntry::SystemEvent { .. }
                | TranscriptEntry::Checkpoint { .. } => {
                    // CompactionSummary: inlined into system prompt above
                    // SystemEvent/Checkpoint: internal bookkeeping, not sent to model
                }
            }
        }

        messages
    }

    /// Estimate the total token count for a potential API request.
    ///
    /// Sums system-layer overhead + per-entry token estimates.
    /// CompactionSummary entries are excluded from the transcript sum because
    /// they are inlined into the system prompt (and counted via system overhead).
    pub fn estimate_tokens(&self, entries: &[TranscriptEntry]) -> usize {
        let transcript_tokens: usize = entries
            .iter()
            .filter(|e| !matches!(e, TranscriptEntry::CompactionSummary { .. }))
            .map(|e| e.token_estimate())
            .sum();

        // The most recent CompactionSummary is inlined into the system prompt,
        // so add its tokens plus the wrapper overhead to the system estimate.
        let summary_overhead = entries
            .iter()
            .rev()
            .find_map(|e| match e {
                TranscriptEntry::CompactionSummary { token_estimate, .. } => {
                    Some(*token_estimate + SUMMARY_WRAPPER.len() / 4)
                }
                _ => None,
            })
            .unwrap_or(0);

        self.system_overhead_estimate + summary_overhead + transcript_tokens
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
    fn compaction_summary_inlined_into_system_prompt() {
        let assembler = ContextAssembler::new(test_system_prompt());
        let entries = vec![TranscriptEntry::compaction_summary(
            "User asked about auth. Files read: src/auth.rs",
        )];
        let messages = assembler.build_messages(&entries);

        // Only one message: system prompt with summary inlined
        assert_eq!(messages.len(), 1);
        let content = messages[0]["content"].as_str().unwrap();
        assert!(content.starts_with("You are a helpful assistant."));
        assert!(content.contains("---"));
        assert!(content.contains("[Context summary from earlier in this conversation]"));
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
    fn message_ordering_summary_inlined_then_history() {
        let assembler = ContextAssembler::new(test_system_prompt());
        let entries = vec![
            TranscriptEntry::compaction_summary("Earlier conversation summary"),
            TranscriptEntry::user_message("What next?"),
            TranscriptEntry::assistant_message("Let me check."),
        ];
        let messages = assembler.build_messages(&entries);

        // 3 messages: system prompt (with summary inlined) + user + assistant
        assert_eq!(messages.len(), 3);
        assert_eq!(messages[0]["role"], "system");
        let content = messages[0]["content"].as_str().unwrap();
        assert!(content.contains("Earlier conversation summary"));
        assert_eq!(messages[1]["role"], "user");
        assert_eq!(messages[2]["role"], "assistant");
    }

    #[test]
    fn multiple_compaction_summaries_only_latest_used() {
        let assembler = ContextAssembler::new(test_system_prompt());
        let entries = vec![
            TranscriptEntry::compaction_summary("Old summary"),
            TranscriptEntry::user_message("middle"),
            TranscriptEntry::compaction_summary("Latest summary"),
            TranscriptEntry::user_message("What next?"),
        ];
        let messages = assembler.build_messages(&entries);

        // system prompt (with latest summary) + 2 user messages
        assert_eq!(messages.len(), 3);
        let content = messages[0]["content"].as_str().unwrap();
        assert!(content.contains("Latest summary"));
        assert!(!content.contains("Old summary"));
    }

    #[test]
    fn no_compaction_summary_leaves_system_prompt_unchanged() {
        let assembler = ContextAssembler::new(test_system_prompt());
        let entries = vec![TranscriptEntry::user_message("hello")];
        let messages = assembler.build_messages(&entries);

        assert_eq!(messages[0]["content"], "You are a helpful assistant.");
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

    #[test]
    fn estimate_tokens_excludes_compaction_summary_from_transcript() {
        let assembler = ContextAssembler::new(test_system_prompt());
        let summary_text = "x".repeat(400); // 400 chars = 100 tokens
        let entries = vec![
            TranscriptEntry::compaction_summary(&summary_text),
            TranscriptEntry::user_message("hello world"), // 11/4 = 2
        ];
        let estimate = assembler.estimate_tokens(&entries);

        let overhead = "You are a helpful assistant.".len() / 4; // 7
        let user_tokens = "hello world".len() / 4; // 2
        let summary_tokens = 400 / 4; // 100
        let wrapper_tokens = SUMMARY_WRAPPER.len() / 4;

        // Summary + wrapper tokens counted as system overhead, not transcript
        assert_eq!(
            estimate,
            overhead + summary_tokens + wrapper_tokens + user_tokens
        );
    }

    #[test]
    fn estimate_tokens_matches_actual_build_messages_size() {
        // Verify the estimate accounts for the same content as build_messages
        let assembler = ContextAssembler::new(test_system_prompt());
        let entries = vec![
            TranscriptEntry::compaction_summary("Summary of prior work"),
            TranscriptEntry::user_message("hello"),
        ];

        let messages = assembler.build_messages(&entries);
        let actual_system_len = messages[0]["content"].as_str().unwrap().len();
        let actual_user_len = messages[1]["content"].as_str().unwrap().len();
        let actual_tokens = (actual_system_len + actual_user_len) / 4;

        let estimate = assembler.estimate_tokens(&entries);
        assert_eq!(estimate, actual_tokens);
    }

    #[test]
    fn estimate_tokens_no_double_count_across_compaction_cycles() {
        let assembler = ContextAssembler::new(test_system_prompt());
        let entries = vec![
            TranscriptEntry::compaction_summary("old summary that was folded in"),
            TranscriptEntry::compaction_summary("latest summary"),
            TranscriptEntry::user_message("hello"),
        ];

        let estimate = assembler.estimate_tokens(&entries);
        let overhead = "You are a helpful assistant.".len() / 4;
        let user_tokens = "hello".len() / 4;
        // Only the latest summary + wrapper is counted
        let latest_summary_tokens = "latest summary".len() / 4;
        let wrapper_tokens = SUMMARY_WRAPPER.len() / 4;

        assert_eq!(
            estimate,
            overhead + latest_summary_tokens + wrapper_tokens + user_tokens
        );
    }

    #[test]
    fn reasoning_content_round_tripped_in_assistant_message() {
        let assembler = ContextAssembler::new(test_system_prompt());
        let entries = vec![TranscriptEntry::assistant_message_with_reasoning(
            "visible text",
            Some("thinking...".to_string()),
            None,
        )];
        let messages = assembler.build_messages(&entries);

        assert_eq!(messages[1]["role"], "assistant");
        assert_eq!(messages[1]["content"], "visible text");
        assert_eq!(messages[1]["reasoning_content"], "thinking...");
        assert!(messages[1].get("encrypted_reasoning").is_none());
    }

    #[test]
    fn encrypted_reasoning_round_tripped_in_assistant_message() {
        let assembler = ContextAssembler::new(test_system_prompt());
        let entries = vec![TranscriptEntry::assistant_message_with_reasoning(
            "answer",
            None,
            Some("encrypted_blob_data".to_string()),
        )];
        let messages = assembler.build_messages(&entries);

        assert_eq!(messages[1]["content"], "answer");
        assert!(messages[1].get("reasoning_content").is_none());
        assert_eq!(messages[1]["encrypted_reasoning"], "encrypted_blob_data");
    }

    #[test]
    fn no_reasoning_fields_when_none() {
        let assembler = ContextAssembler::new(test_system_prompt());
        let entries = vec![TranscriptEntry::assistant_message("plain text")];
        let messages = assembler.build_messages(&entries);

        assert_eq!(messages[1]["content"], "plain text");
        assert!(messages[1].get("reasoning_content").is_none());
        assert!(messages[1].get("encrypted_reasoning").is_none());
    }
}
