use std::collections::HashMap;

use crate::session::TranscriptEntry;

/// Number of recent user turns to preserve verbatim during heuristic compaction.
const RECENT_TURNS: usize = 5;

/// Result of running heuristic compaction.
pub struct CompactionResult {
    /// The compacted transcript entries.
    pub entries: Vec<TranscriptEntry>,
    /// Whether any entries were modified or removed.
    pub compacted: bool,
}

/// Run heuristic compaction on transcript entries (no LLM call).
///
/// Preserves the last `RECENT_TURNS` user turns verbatim. For older entries:
/// - Tool result content replaced with short placeholders
/// - Duplicate file reads removed (only most recent kept)
/// - Old shell output truncated to first 3 + last 3 lines
pub fn heuristic_compact(entries: &[TranscriptEntry]) -> CompactionResult {
    let boundary = find_recent_boundary(entries, RECENT_TURNS);

    // If everything is recent, nothing to compact
    if boundary == 0 {
        return CompactionResult {
            entries: entries.to_vec(),
            compacted: false,
        };
    }

    let old_entries = &entries[..boundary];
    let recent_entries = &entries[boundary..];

    // Find the last read of each file path across the entire transcript.
    // Key: file path, Value: index (in the full entries slice) of the last ToolCall for that path.
    let last_read_index = find_last_file_read_indices(entries);

    // Process old entries
    let mut result = Vec::new();
    let mut compacted = false;
    let mut i = 0;

    while i < old_entries.len() {
        match &old_entries[i] {
            TranscriptEntry::ToolCall { call_id, name, arguments, .. } => {
                // Check if this is a file_read that has a later duplicate
                if name == "file_read" {
                    if let Some(path) = extract_path(arguments) {
                        if let Some(&last_idx) = last_read_index.get(&path) {
                            if last_idx > i {
                                // This read has a later duplicate — skip both ToolCall and ToolResult
                                // Find and skip the matching ToolResult
                                let cid = call_id.clone();
                                i += 1;
                                while i < old_entries.len() {
                                    if matches!(&old_entries[i], TranscriptEntry::ToolResult { call_id, .. } if call_id == &cid) {
                                        i += 1; // skip the result too
                                        break;
                                    }
                                    // Keep non-matching entries
                                    result.push(old_entries[i].clone());
                                    i += 1;
                                }
                                compacted = true;
                                continue;
                            }
                        }
                    }
                }
                // Keep the tool call as-is
                result.push(old_entries[i].clone());
                i += 1;
            }
            TranscriptEntry::ToolResult { call_id, name, output, .. } => {
                // Replace tool result content with placeholder
                let placeholder = make_placeholder(name, output, &find_args_for_call_id(old_entries, call_id));
                if placeholder != *output {
                    compacted = true;
                }
                result.push(TranscriptEntry::tool_result(call_id, name, &placeholder));
                i += 1;
            }
            _ => {
                result.push(old_entries[i].clone());
                i += 1;
            }
        }
    }

    // Append recent entries unchanged
    result.extend_from_slice(recent_entries);

    CompactionResult {
        entries: result,
        compacted,
    }
}

/// Find the index in `entries` where the recent region begins.
/// Returns 0 if everything is within the recent window.
fn find_recent_boundary(entries: &[TranscriptEntry], recent_turns: usize) -> usize {
    let user_indices: Vec<usize> = entries
        .iter()
        .enumerate()
        .filter(|(_, e)| matches!(e, TranscriptEntry::UserMessage { .. }))
        .map(|(i, _)| i)
        .collect();

    if user_indices.len() <= recent_turns {
        return 0;
    }

    user_indices[user_indices.len() - recent_turns]
}

/// For each file path that was read (via file_read ToolCall), record the index of its last occurrence.
fn find_last_file_read_indices(entries: &[TranscriptEntry]) -> HashMap<String, usize> {
    let mut last_read: HashMap<String, usize> = HashMap::new();

    for (i, entry) in entries.iter().enumerate() {
        if let TranscriptEntry::ToolCall { name, arguments, .. } = entry {
            if name == "file_read" {
                if let Some(path) = extract_path(arguments) {
                    last_read.insert(path, i);
                }
            }
        }
    }

    last_read
}

/// Extract the `path` field from tool call arguments JSON.
fn extract_path(arguments: &str) -> Option<String> {
    serde_json::from_str::<serde_json::Value>(arguments)
        .ok()
        .and_then(|v| v.get("path").and_then(|p| p.as_str().map(String::from)))
}

/// Find the arguments string for a ToolCall with the given call_id.
fn find_args_for_call_id(entries: &[TranscriptEntry], target_call_id: &str) -> String {
    for entry in entries {
        if let TranscriptEntry::ToolCall { call_id, arguments, .. } = entry {
            if call_id == target_call_id {
                return arguments.clone();
            }
        }
    }
    String::new()
}

/// Create a compact placeholder for a tool result.
fn make_placeholder(tool_name: &str, output: &str, arguments: &str) -> String {
    let path = extract_path(arguments).unwrap_or_else(|| "?".to_string());
    let line_count = output.lines().count();

    match tool_name {
        "file_read" => {
            format!("[file_read: {path} — {line_count} lines]")
        }
        "file_write" => {
            format!("[file_write: {path}]")
        }
        "file_edit" => {
            format!("[file_edit: {path}]")
        }
        "list_files" => {
            let entry_count = output.lines().count();
            format!("[list_files: {path} — {entry_count} entries]")
        }
        "grep" => {
            let pattern = serde_json::from_str::<serde_json::Value>(arguments)
                .ok()
                .and_then(|v| v.get("pattern").and_then(|p| p.as_str().map(String::from)))
                .unwrap_or_else(|| "?".to_string());
            let match_count = output.lines().count();
            format!("[grep: \"{pattern}\" — {match_count} matches]")
        }
        "shell_exec" => {
            truncate_shell_output(output)
        }
        _ => {
            format!("[{tool_name}: {line_count} lines]")
        }
    }
}

/// Truncate shell output to first 3 + last 3 lines.
fn truncate_shell_output(output: &str) -> String {
    let lines: Vec<&str> = output.lines().collect();
    if lines.len() <= 6 {
        return output.to_string();
    }

    let first = &lines[..3];
    let last = &lines[lines.len() - 3..];
    let omitted = lines.len() - 6;

    format!(
        "{}\n... ({omitted} lines omitted)\n{}",
        first.join("\n"),
        last.join("\n")
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    // -----------------------------------------------------------------------
    // Helper: build a multi-turn transcript for testing
    // -----------------------------------------------------------------------

    /// Build a simple turn: user message + assistant response
    fn simple_turn(user_msg: &str, assistant_msg: &str) -> Vec<TranscriptEntry> {
        vec![
            TranscriptEntry::user_message(user_msg),
            TranscriptEntry::assistant_message(assistant_msg),
        ]
    }

    /// Build a turn with a tool call: user → tool_call → tool_result → assistant
    fn tool_turn(
        user_msg: &str,
        tool_name: &str,
        args: &str,
        call_id: &str,
        output: &str,
        assistant_msg: &str,
    ) -> Vec<TranscriptEntry> {
        vec![
            TranscriptEntry::user_message(user_msg),
            TranscriptEntry::tool_call(call_id, tool_name, args),
            TranscriptEntry::tool_result(call_id, tool_name, output),
            TranscriptEntry::assistant_message(assistant_msg),
        ]
    }

    // -----------------------------------------------------------------------
    // find_recent_boundary
    // -----------------------------------------------------------------------

    #[test]
    fn boundary_all_recent_when_fewer_than_5_turns() {
        let entries: Vec<TranscriptEntry> = (0..3)
            .flat_map(|i| simple_turn(&format!("q{i}"), &format!("a{i}")))
            .collect();
        assert_eq!(find_recent_boundary(&entries, 5), 0);
    }

    #[test]
    fn boundary_exactly_5_turns_all_recent() {
        let entries: Vec<TranscriptEntry> = (0..5)
            .flat_map(|i| simple_turn(&format!("q{i}"), &format!("a{i}")))
            .collect();
        assert_eq!(find_recent_boundary(&entries, 5), 0);
    }

    #[test]
    fn boundary_6_turns_first_is_old() {
        let entries: Vec<TranscriptEntry> = (0..6)
            .flat_map(|i| simple_turn(&format!("q{i}"), &format!("a{i}")))
            .collect();
        // Turn 0 is old (entries 0-1), turns 1-5 are recent (entries 2-11)
        let boundary = find_recent_boundary(&entries, 5);
        assert_eq!(boundary, 2); // Index of the 2nd UserMessage (turn 1)
    }

    #[test]
    fn boundary_with_tool_turns() {
        let mut entries = tool_turn("q0", "file_read", r#"{"path":"a.rs"}"#, "c0", "content", "a0");
        entries.extend(simple_turn("q1", "a1"));
        entries.extend(simple_turn("q2", "a2"));
        entries.extend(simple_turn("q3", "a3"));
        entries.extend(simple_turn("q4", "a4"));
        entries.extend(simple_turn("q5", "a5"));
        // 6 turns: first has 4 entries, rest have 2 each = 4 + 10 = 14 entries
        // Boundary should be at index 4 (start of turn 1)
        assert_eq!(find_recent_boundary(&entries, 5), 4);
    }

    // -----------------------------------------------------------------------
    // No compaction when everything is recent
    // -----------------------------------------------------------------------

    #[test]
    fn no_compaction_when_below_5_turns() {
        let entries: Vec<TranscriptEntry> = (0..3)
            .flat_map(|i| simple_turn(&format!("q{i}"), &format!("a{i}")))
            .collect();
        let result = heuristic_compact(&entries);
        assert!(!result.compacted);
        assert_eq!(result.entries.len(), entries.len());
    }

    // -----------------------------------------------------------------------
    // Tool result placeholders in old region
    // -----------------------------------------------------------------------

    #[test]
    fn old_file_read_result_replaced_with_placeholder() {
        let mut entries = tool_turn(
            "q0",
            "file_read",
            r#"{"path":"src/main.rs"}"#,
            "c0",
            "line1\nline2\nline3",
            "a0",
        );
        // Add 5 more simple turns to push turn 0 into the old region
        for i in 1..=5 {
            entries.extend(simple_turn(&format!("q{i}"), &format!("a{i}")));
        }

        let result = heuristic_compact(&entries);
        assert!(result.compacted);

        // Find the tool result in the compacted entries
        let tool_result = result.entries.iter().find(|e| {
            matches!(e, TranscriptEntry::ToolResult { call_id, .. } if call_id == "c0")
        });
        assert!(tool_result.is_some());
        if let TranscriptEntry::ToolResult { output, .. } = tool_result.unwrap() {
            assert_eq!(output, "[file_read: src/main.rs — 3 lines]");
        }
    }

    #[test]
    fn recent_turns_preserved_verbatim() {
        let mut entries = tool_turn(
            "q0",
            "file_read",
            r#"{"path":"old.rs"}"#,
            "c0",
            "old content",
            "a0",
        );
        // Recent turns include a tool use
        for i in 1..=4 {
            entries.extend(simple_turn(&format!("q{i}"), &format!("a{i}")));
        }
        entries.extend(tool_turn(
            "q5",
            "file_read",
            r#"{"path":"recent.rs"}"#,
            "c5",
            "recent file content that should be preserved",
            "a5",
        ));

        let result = heuristic_compact(&entries);

        // The recent file_read result should be unchanged
        let recent_result = result.entries.iter().find(|e| {
            matches!(e, TranscriptEntry::ToolResult { call_id, .. } if call_id == "c5")
        });
        assert!(recent_result.is_some());
        if let TranscriptEntry::ToolResult { output, .. } = recent_result.unwrap() {
            assert_eq!(output, "recent file content that should be preserved");
        }
    }

    // -----------------------------------------------------------------------
    // Deduplication of file reads
    // -----------------------------------------------------------------------

    #[test]
    fn duplicate_file_read_in_old_region_removed() {
        // Turn 0: read a.rs
        let mut entries = tool_turn(
            "q0",
            "file_read",
            r#"{"path":"a.rs"}"#,
            "c0",
            "old content",
            "a0",
        );
        // Turn 1: read a.rs again (more recent, still in old region)
        entries.extend(tool_turn(
            "q1",
            "file_read",
            r#"{"path":"a.rs"}"#,
            "c1",
            "newer content",
            "a1",
        ));
        // Turns 2-6 to push turns 0-1 into old region
        for i in 2..=6 {
            entries.extend(simple_turn(&format!("q{i}"), &format!("a{i}")));
        }

        let result = heuristic_compact(&entries);
        assert!(result.compacted);

        // The first read (c0) should be removed
        let c0_exists = result.entries.iter().any(|e| {
            matches!(e, TranscriptEntry::ToolCall { call_id, .. } if call_id == "c0")
                || matches!(e, TranscriptEntry::ToolResult { call_id, .. } if call_id == "c0")
        });
        assert!(!c0_exists, "duplicate file read should be removed");

        // The second read (c1) should still exist (as a placeholder)
        let c1_exists = result.entries.iter().any(|e| {
            matches!(e, TranscriptEntry::ToolResult { call_id, .. } if call_id == "c1")
        });
        assert!(c1_exists, "most recent file read should be kept");
    }

    #[test]
    fn file_read_in_old_region_removed_when_recent_has_same_file() {
        // Turn 0: read a.rs (old region)
        let mut entries = tool_turn(
            "q0",
            "file_read",
            r#"{"path":"a.rs"}"#,
            "c0",
            "old content",
            "a0",
        );
        // Turns 1-4: simple
        for i in 1..=4 {
            entries.extend(simple_turn(&format!("q{i}"), &format!("a{i}")));
        }
        // Turn 5: read a.rs again (recent region)
        entries.extend(tool_turn(
            "q5",
            "file_read",
            r#"{"path":"a.rs"}"#,
            "c5",
            "current content",
            "a5",
        ));

        let result = heuristic_compact(&entries);
        assert!(result.compacted);

        // Old read should be removed
        let c0_exists = result.entries.iter().any(|e| {
            matches!(e, TranscriptEntry::ToolCall { call_id, .. } if call_id == "c0")
        });
        assert!(!c0_exists);

        // Recent read should be untouched
        let c5_result = result.entries.iter().find(|e| {
            matches!(e, TranscriptEntry::ToolResult { call_id, .. } if call_id == "c5")
        });
        if let Some(TranscriptEntry::ToolResult { output, .. }) = c5_result {
            assert_eq!(output, "current content");
        }
    }

    // -----------------------------------------------------------------------
    // Shell output truncation
    // -----------------------------------------------------------------------

    #[test]
    fn old_shell_output_truncated() {
        let long_output = (0..20)
            .map(|i| format!("line {i}"))
            .collect::<Vec<_>>()
            .join("\n");

        let mut entries = tool_turn(
            "q0",
            "shell_exec",
            r#"{"command":"ls -la"}"#,
            "c0",
            &long_output,
            "a0",
        );
        for i in 1..=5 {
            entries.extend(simple_turn(&format!("q{i}"), &format!("a{i}")));
        }

        let result = heuristic_compact(&entries);
        assert!(result.compacted);

        let shell_result = result.entries.iter().find(|e| {
            matches!(e, TranscriptEntry::ToolResult { call_id, .. } if call_id == "c0")
        });
        if let Some(TranscriptEntry::ToolResult { output, .. }) = shell_result {
            assert!(output.contains("line 0"));
            assert!(output.contains("line 1"));
            assert!(output.contains("line 2"));
            assert!(output.contains("lines omitted"));
            assert!(output.contains("line 17"));
            assert!(output.contains("line 18"));
            assert!(output.contains("line 19"));
            assert!(!output.contains("line 10"));
        } else {
            panic!("shell_exec result not found");
        }
    }

    #[test]
    fn short_shell_output_not_truncated() {
        let short_output = "line 0\nline 1\nline 2";
        let mut entries = tool_turn(
            "q0",
            "shell_exec",
            r#"{"command":"echo hi"}"#,
            "c0",
            short_output,
            "a0",
        );
        for i in 1..=5 {
            entries.extend(simple_turn(&format!("q{i}"), &format!("a{i}")));
        }

        let result = heuristic_compact(&entries);
        // Shell output is short enough — no truncation needed
        let shell_result = result.entries.iter().find(|e| {
            matches!(e, TranscriptEntry::ToolResult { call_id, .. } if call_id == "c0")
        });
        if let Some(TranscriptEntry::ToolResult { output, .. }) = shell_result {
            assert_eq!(output, short_output);
        }
    }

    // -----------------------------------------------------------------------
    // Token estimates recalculated
    // -----------------------------------------------------------------------

    #[test]
    fn compacted_entries_have_recalculated_token_estimates() {
        let big_content = "x".repeat(4000); // 4000 chars = 1000 token estimate
        let mut entries = tool_turn(
            "q0",
            "file_read",
            r#"{"path":"big.rs"}"#,
            "c0",
            &big_content,
            "a0",
        );
        for i in 1..=5 {
            entries.extend(simple_turn(&format!("q{i}"), &format!("a{i}")));
        }

        let original_estimate: usize = entries.iter().map(|e| e.token_estimate()).sum();

        let result = heuristic_compact(&entries);
        let compacted_estimate: usize = result.entries.iter().map(|e| e.token_estimate()).sum();

        assert!(
            compacted_estimate < original_estimate,
            "compacted estimate ({compacted_estimate}) should be less than original ({original_estimate})"
        );
    }

    // -----------------------------------------------------------------------
    // Grep placeholder
    // -----------------------------------------------------------------------

    #[test]
    fn old_grep_result_replaced_with_placeholder() {
        let mut entries = tool_turn(
            "q0",
            "grep",
            r#"{"pattern":"TODO","path":"src"}"#,
            "c0",
            "src/a.rs:10: TODO fix\nsrc/b.rs:20: TODO cleanup",
            "a0",
        );
        for i in 1..=5 {
            entries.extend(simple_turn(&format!("q{i}"), &format!("a{i}")));
        }

        let result = heuristic_compact(&entries);
        let grep_result = result.entries.iter().find(|e| {
            matches!(e, TranscriptEntry::ToolResult { call_id, .. } if call_id == "c0")
        });
        if let Some(TranscriptEntry::ToolResult { output, .. }) = grep_result {
            assert_eq!(output, r#"[grep: "TODO" — 2 matches]"#);
        }
    }

    // -----------------------------------------------------------------------
    // truncate_shell_output unit tests
    // -----------------------------------------------------------------------

    #[test]
    fn truncate_shell_exactly_6_lines_no_truncation() {
        let input = "a\nb\nc\nd\ne\nf";
        assert_eq!(truncate_shell_output(input), input);
    }

    #[test]
    fn truncate_shell_7_lines() {
        let input = "a\nb\nc\nd\ne\nf\ng";
        let result = truncate_shell_output(input);
        assert_eq!(result, "a\nb\nc\n... (1 lines omitted)\ne\nf\ng");
    }

    // -----------------------------------------------------------------------
    // Empty transcript
    // -----------------------------------------------------------------------

    #[test]
    fn empty_transcript_returns_empty() {
        let result = heuristic_compact(&[]);
        assert!(!result.compacted);
        assert!(result.entries.is_empty());
    }

    // -----------------------------------------------------------------------
    // Mixed turn: file tools + shell_exec both compacted
    // -----------------------------------------------------------------------

    #[test]
    fn mixed_turn_both_tools_compacted() {
        let long_output = (0..20).map(|i| format!("line{i}")).collect::<Vec<_>>().join("\n");
        let mut entries = vec![
            TranscriptEntry::user_message("q0"),
            TranscriptEntry::tool_call("c0a", "file_read", r#"{"path":"a.rs"}"#),
            TranscriptEntry::tool_result("c0a", "file_read", "file content here\nsecond line"),
            TranscriptEntry::tool_call("c0b", "shell_exec", r#"{"command":"cargo test"}"#),
            TranscriptEntry::tool_result("c0b", "shell_exec", &long_output),
            TranscriptEntry::assistant_message("a0"),
        ];
        for i in 1..=5 {
            entries.extend(simple_turn(&format!("q{i}"), &format!("a{i}")));
        }

        let result = heuristic_compact(&entries);
        assert!(result.compacted);

        // file_read should be placeholder
        let fr = result.entries.iter().find(|e| {
            matches!(e, TranscriptEntry::ToolResult { call_id, .. } if call_id == "c0a")
        });
        if let Some(TranscriptEntry::ToolResult { output, .. }) = fr {
            assert!(output.starts_with("[file_read:"));
        }

        // shell_exec should be truncated
        let se = result.entries.iter().find(|e| {
            matches!(e, TranscriptEntry::ToolResult { call_id, .. } if call_id == "c0b")
        });
        if let Some(TranscriptEntry::ToolResult { output, .. }) = se {
            assert!(output.contains("lines omitted"));
        }
    }
}
