use std::collections::HashMap;
use std::path::{Path, PathBuf};

use anyhow::Result;
use serde_json::json;

use crate::api::GrokApi;
use crate::context_assembler::ContextAssembler;
use crate::model_profile::ModelProfile;
use crate::session::TranscriptEntry;

/// Number of recent user turns to preserve verbatim during compaction.
const RECENT_TURNS: usize = 5;

/// System prompt for LLM-based summarization.
const SUMMARIZATION_PROMPT: &str = "\
You are a conversation summarizer for a coding assistant session. \
Produce a structured summary of the conversation using exactly these 9 sections. \
Each section should be concise but capture all important details. \
If a section has nothing relevant, write \"None.\" under it. \
Do not add any sections beyond these 9.\n\
\n\
## Primary Request\n\
What the user originally asked for and the main goal of the conversation.\n\
\n\
## Technical Concepts\n\
Key technical concepts, patterns, or architectural decisions discussed.\n\
\n\
## Files & Code\n\
Files that were read, written, or edited, with brief notes on what was done to each.\n\
\n\
## Errors & Fixes\n\
Any errors encountered and how they were resolved.\n\
\n\
## Problem Solving\n\
Key decision points, trade-offs considered, and reasoning applied.\n\
\n\
## User Messages\n\
Important clarifications, preferences, or constraints the user communicated.\n\
\n\
## Pending Tasks\n\
Work that was discussed but not yet completed.\n\
\n\
## Current Work\n\
What was being worked on most recently.\n\
\n\
## Next Step\n\
What should happen next based on the conversation state.";

/// Result of running compaction (heuristic and/or LLM).
pub struct CompactionResult {
    /// The compacted transcript entries.
    pub entries: Vec<TranscriptEntry>,
    /// Whether any entries were modified or removed.
    pub compacted: bool,
    /// Token usage from LLM summarization (None if only heuristic compaction ran).
    pub llm_usage: Option<crate::api::Usage>,
}

/// Check whether compaction should fire and, if so, run it.
///
/// Runs heuristic compaction first (no LLM call). If the result is still
/// above the compaction threshold, escalates to LLM-based summarization.
///
/// Returns `Some(CompactionResult)` with the compacted entries if compaction
/// fired and produced changes. Returns `None` if the estimate is below the
/// threshold or if compaction had no effect.
pub async fn maybe_compact(
    history: &[TranscriptEntry],
    assembler: &ContextAssembler,
    model: &str,
    project_root: &Path,
    api: &dyn GrokApi,
) -> Option<CompactionResult> {
    let profile = ModelProfile::for_model(model);
    let estimated = assembler.estimate_tokens(history);
    if estimated <= profile.compaction_threshold() {
        return None;
    }

    // Layer 1: heuristic compaction
    let heuristic_result = heuristic_compact(history, project_root);

    // Check if heuristic was sufficient
    let after_heuristic = assembler.estimate_tokens(&heuristic_result.entries);
    if after_heuristic <= profile.compaction_threshold() {
        return if heuristic_result.compacted { Some(heuristic_result) } else { None };
    }

    // Layer 2: LLM compaction on the heuristic-compacted entries
    match llm_compact(&heuristic_result.entries, model, api).await {
        Ok(llm_result) if llm_result.compacted => Some(llm_result),
        Ok(llm_result) => {
            // LLM compaction didn't help (e.g. summary was too verbose).
            // Carry the LLM usage through — tokens were still spent.
            if heuristic_result.compacted {
                Some(CompactionResult {
                    llm_usage: llm_result.llm_usage,
                    ..heuristic_result
                })
            } else {
                // Nothing compacted, but if LLM tokens were spent, report them
                match llm_result.llm_usage {
                    Some(_) => Some(CompactionResult {
                        entries: history.to_vec(),
                        compacted: false,
                        llm_usage: llm_result.llm_usage,
                    }),
                    None => None,
                }
            }
        }
        Err(e) => {
            eprintln!("  warning: LLM compaction failed: {e}");
            if heuristic_result.compacted { Some(heuristic_result) } else { None }
        }
    }
}

/// Run heuristic compaction on transcript entries (no LLM call).
///
/// `project_root` is used to normalize file paths so that relative and absolute
/// references to the same file are correctly deduplicated.
///
/// Preserves the last `RECENT_TURNS` user turns verbatim. For older entries:
/// - Tool result content replaced with short placeholders
/// - Duplicate file reads removed (only most recent kept)
/// - Old shell output truncated to first 3 + last 3 lines
pub fn heuristic_compact(entries: &[TranscriptEntry], project_root: &Path) -> CompactionResult {
    let boundary = find_recent_boundary(entries, RECENT_TURNS);

    // If everything is recent, nothing to compact
    if boundary == 0 {
        return CompactionResult {
            entries: entries.to_vec(),
            compacted: false,
            llm_usage: None,
        };
    }

    let old_entries = &entries[..boundary];
    let recent_entries = &entries[boundary..];

    // Find the last read of each file path across the entire transcript.
    // Key: normalized file path, Value: index (in the full entries slice) of the last ToolCall.
    let last_read_index = find_last_file_read_indices(entries, project_root);

    // Process old entries
    let mut result = Vec::new();
    let mut compacted = false;
    let mut i = 0;

    while i < old_entries.len() {
        match &old_entries[i] {
            TranscriptEntry::ToolCall { call_id, name, arguments, .. }
                if name == "file_read"
                    && extract_path(arguments)
                        .map(|p| normalize_path(&p, project_root))
                        .and_then(|norm| last_read_index.get(&norm).copied())
                        .is_some_and(|last_idx| last_idx > i) =>
            {
                // This file_read has a later duplicate — skip both ToolCall and ToolResult
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
            TranscriptEntry::ToolCall { .. } => {
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
        llm_usage: None,
    }
}

/// Run LLM-based compaction on transcript entries.
///
/// Sends the old entries (before the recent boundary) to the model for
/// structured summarization. The summary replaces all older messages,
/// with recent turns preserved verbatim.
///
/// If the old entries exceed 80% of the model's context window, uses
/// chunked summarization: summarize the older half first, then combine
/// with the newer half in a second pass.
pub async fn llm_compact(
    entries: &[TranscriptEntry],
    model: &str,
    api: &dyn GrokApi,
) -> Result<CompactionResult> {
    let profile = ModelProfile::for_model(model);
    let boundary = find_recent_boundary(entries, RECENT_TURNS);

    if boundary == 0 {
        return Ok(CompactionResult {
            entries: entries.to_vec(),
            compacted: false,
            llm_usage: None,
        });
    }

    let old_entries = &entries[..boundary];
    let recent_entries = &entries[boundary..];

    // Check if we need chunked summarization (old entries exceed 80% of context)
    let old_tokens: usize = old_entries.iter().map(|e| e.token_estimate()).sum();
    let overflow_threshold = profile.context_window * 80 / 100;

    let (summary, usage) = if old_tokens > overflow_threshold {
        chunked_summarize(old_entries, api).await?
    } else {
        single_summarize(old_entries, api).await?
    };

    // Validate that the summary actually reduces context size.
    // A verbose or malformed summary could be larger than the original.
    let summary_tokens = summary.len() / 4;
    if summary_tokens >= old_tokens {
        return Ok(CompactionResult {
            entries: entries.to_vec(),
            compacted: false,
            llm_usage: usage,
        });
    }

    let mut result = vec![TranscriptEntry::compaction_summary(&summary)];
    result.extend_from_slice(recent_entries);

    Ok(CompactionResult {
        entries: result,
        compacted: true,
        llm_usage: usage,
    })
}

/// Summarize entries in a single API call.
async fn single_summarize(
    entries: &[TranscriptEntry],
    api: &dyn GrokApi,
) -> Result<(String, Option<crate::api::Usage>)> {
    let formatted = format_entries_for_summarization(entries);
    let messages = vec![
        json!({"role": "system", "content": SUMMARIZATION_PROMPT}),
        json!({"role": "user", "content": format!("Summarize this conversation:\n\n{formatted}")}),
    ];

    let response = api.send_turn(messages, &[], &mut |_| {}).await?;
    Ok((response.text, response.usage))
}

/// Summarize entries in two passes when they exceed context limits.
///
/// First pass: summarize the older half of the entries.
/// Second pass: combine the first summary with the newer half.
///
/// Splits at the nearest UserMessage boundary to the midpoint so that
/// turns are never bisected (avoiding orphaned ToolCall/ToolResult pairs).
async fn chunked_summarize(
    entries: &[TranscriptEntry],
    api: &dyn GrokApi,
) -> Result<(String, Option<crate::api::Usage>)> {
    let split = find_turn_boundary_near_midpoint(entries);
    let older_half = &entries[..split];
    let newer_half = &entries[split..];

    // First pass: summarize the older half
    let (first_summary, first_usage) = single_summarize(older_half, api).await?;

    // Second pass: combine first summary with newer half
    let newer_formatted = format_entries_for_summarization(newer_half);
    let messages = vec![
        json!({"role": "system", "content": SUMMARIZATION_PROMPT}),
        json!({"role": "user", "content": format!(
            "Combine this earlier summary with the newer conversation into a single summary.\n\n\
             Earlier summary:\n{first_summary}\n\n\
             Newer conversation:\n{newer_formatted}"
        )}),
    ];

    let second_response = api.send_turn(messages, &[], &mut |_| {}).await?;

    // Combine usage from both passes
    let combined_usage = match (first_usage, second_response.usage) {
        (Some(u1), Some(u2)) => Some(crate::api::Usage {
            input_tokens: u1.input_tokens + u2.input_tokens,
            output_tokens: u1.output_tokens + u2.output_tokens,
        }),
        (Some(u), None) | (None, Some(u)) => Some(u),
        (None, None) => None,
    };

    Ok((second_response.text, combined_usage))
}

/// Find the UserMessage boundary nearest to the midpoint of entries.
///
/// Used by chunked summarization to split at a turn boundary rather than
/// at an arbitrary entry index, preventing orphaned ToolCall/ToolResult pairs.
/// Falls back to raw midpoint if no valid UserMessage boundary exists
/// (no UserMessages, or the only candidate is index 0 which would produce
/// an empty first half).
fn find_turn_boundary_near_midpoint(entries: &[TranscriptEntry]) -> usize {
    let midpoint = entries.len() / 2;

    // Collect indices of UserMessage entries, excluding index 0
    // (splitting at 0 produces an empty first half which defeats chunking)
    let user_indices: Vec<usize> = entries
        .iter()
        .enumerate()
        .filter(|(i, e)| *i > 0 && matches!(e, TranscriptEntry::UserMessage { .. }))
        .map(|(i, _)| i)
        .collect();

    if user_indices.is_empty() {
        return midpoint;
    }

    // Find the UserMessage index closest to the midpoint
    *user_indices
        .iter()
        .min_by_key(|&&idx| (idx as isize - midpoint as isize).unsigned_abs())
        .unwrap()
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
/// Paths are normalized so that relative and absolute references to the same file match.
fn find_last_file_read_indices(entries: &[TranscriptEntry], project_root: &Path) -> HashMap<String, usize> {
    let mut last_read: HashMap<String, usize> = HashMap::new();

    for (i, entry) in entries.iter().enumerate() {
        if let TranscriptEntry::ToolCall { name, arguments, .. } = entry
            && name == "file_read"
            && let Some(path) = extract_path(arguments)
        {
            let normalized = normalize_path(&path, project_root);
            last_read.insert(normalized, i);
        }
    }

    last_read
}

/// Normalize a file path for dedup comparison.
///
/// Makes relative paths absolute using `project_root`, then cleans `.` and `..`
/// components without touching the filesystem. This ensures `src/main.rs` and
/// `/abs/repo/src/main.rs` resolve to the same key.
fn normalize_path(raw: &str, project_root: &Path) -> String {
    let p = PathBuf::from(raw);
    let absolute = if p.is_absolute() {
        p
    } else {
        project_root.join(p)
    };
    clean_path(&absolute).to_string_lossy().into_owned()
}

/// Lexically clean a path: resolve `.` and `..` components without filesystem access.
fn clean_path(path: &Path) -> PathBuf {
    let mut out = PathBuf::new();
    for component in path.components() {
        match component {
            std::path::Component::CurDir => {} // skip `.`
            std::path::Component::ParentDir => { out.pop(); }
            other => out.push(other),
        }
    }
    out
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
        if let TranscriptEntry::ToolCall { call_id, arguments, .. } = entry
            && call_id == target_call_id
        {
            return arguments.clone();
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

/// Format transcript entries as human-readable text for LLM summarization.
fn format_entries_for_summarization(entries: &[TranscriptEntry]) -> String {
    let mut text = String::new();
    for entry in entries {
        match entry {
            TranscriptEntry::UserMessage { content, .. } => {
                text.push_str(&format!("User: {content}\n\n"));
            }
            TranscriptEntry::AssistantMessage { content, .. } => {
                text.push_str(&format!("Assistant: {content}\n\n"));
            }
            TranscriptEntry::ToolCall { name, arguments, .. } => {
                text.push_str(&format!("Tool call: {name}({arguments})\n\n"));
            }
            TranscriptEntry::ToolResult { name, output, .. } => {
                let display = if output.len() > 500 {
                    format!("{}... (truncated)", &output[..500])
                } else {
                    output.clone()
                };
                text.push_str(&format!("Tool result ({name}): {display}\n\n"));
            }
            TranscriptEntry::CompactionSummary { summary, .. } => {
                text.push_str(&format!("Previous summary: {summary}\n\n"));
            }
            TranscriptEntry::SystemEvent { .. } => {}
        }
    }
    text
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

    /// Build a turn with substantial content (for LLM compaction tests where
    /// old entries need to be larger than any summary).
    fn big_turn(i: usize) -> Vec<TranscriptEntry> {
        let content = format!("message {i}: {}", "x".repeat(200));
        simple_turn(&content, &content)
    }

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
        let result = heuristic_compact(&entries, Path::new("/project"));
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

        let result = heuristic_compact(&entries, Path::new("/project"));
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

        let result = heuristic_compact(&entries, Path::new("/project"));

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

        let result = heuristic_compact(&entries, Path::new("/project"));
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
    fn dedup_normalizes_relative_vs_absolute_paths() {
        // Turn 0: read via relative path (old region)
        let mut entries = tool_turn(
            "q0",
            "file_read",
            r#"{"path":"src/main.rs"}"#,
            "c0",
            "old content",
            "a0",
        );
        // Turn 1: read same file via absolute path (still in old region)
        entries.extend(tool_turn(
            "q1",
            "file_read",
            r#"{"path":"/project/src/main.rs"}"#,
            "c1",
            "newer content",
            "a1",
        ));
        for i in 2..=6 {
            entries.extend(simple_turn(&format!("q{i}"), &format!("a{i}")));
        }

        let result = heuristic_compact(&entries, Path::new("/project"));
        assert!(result.compacted);

        // The relative read (c0) should be removed — it resolves to the same file
        let c0_exists = result.entries.iter().any(|e| {
            matches!(e, TranscriptEntry::ToolCall { call_id, .. } if call_id == "c0")
        });
        assert!(!c0_exists, "relative path read should be deduped against absolute path");

        // The absolute read (c1) should be kept (most recent)
        let c1_exists = result.entries.iter().any(|e| {
            matches!(e, TranscriptEntry::ToolResult { call_id, .. } if call_id == "c1")
        });
        assert!(c1_exists);
    }

    #[test]
    fn dedup_normalizes_dot_dot_components() {
        // Turn 0: read with ../ in path
        let mut entries = tool_turn(
            "q0",
            "file_read",
            r#"{"path":"src/../src/main.rs"}"#,
            "c0",
            "old content",
            "a0",
        );
        // Turn 1: read clean path
        entries.extend(tool_turn(
            "q1",
            "file_read",
            r#"{"path":"src/main.rs"}"#,
            "c1",
            "newer content",
            "a1",
        ));
        for i in 2..=6 {
            entries.extend(simple_turn(&format!("q{i}"), &format!("a{i}")));
        }

        let result = heuristic_compact(&entries, Path::new("/project"));
        let c0_exists = result.entries.iter().any(|e| {
            matches!(e, TranscriptEntry::ToolCall { call_id, .. } if call_id == "c0")
        });
        assert!(!c0_exists, "path with ../ should be deduped against clean path");
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

        let result = heuristic_compact(&entries, Path::new("/project"));
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

        let result = heuristic_compact(&entries, Path::new("/project"));
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

        let result = heuristic_compact(&entries, Path::new("/project"));
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

        let result = heuristic_compact(&entries, Path::new("/project"));
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

        let result = heuristic_compact(&entries, Path::new("/project"));
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
        let result = heuristic_compact(&[], Path::new("/project"));
        assert!(!result.compacted);
        assert!(result.entries.is_empty());
    }

    // -----------------------------------------------------------------------
    // Mixed turn: file tools + shell_exec both compacted
    // -----------------------------------------------------------------------

    // -----------------------------------------------------------------------
    // Preflight integration: compaction reduces token estimate
    // -----------------------------------------------------------------------

    #[test]
    fn preflight_compaction_reduces_estimate_below_threshold() {
        use crate::context_assembler::ContextAssembler;
        use crate::model_profile::ModelProfile;
        use serde_json::json;

        let assembler = ContextAssembler::new(json!({"role": "system", "content": "short"}));
        let profile = ModelProfile::for_model("grok-3");
        let threshold = profile.compaction_threshold(); // 48,000

        // Build a transcript with large tool results that exceed the threshold
        let big_output = "x".repeat(threshold * 4 + 1000); // well over threshold in chars
        let mut entries = tool_turn(
            "q0",
            "file_read",
            r#"{"path":"huge.rs"}"#,
            "c0",
            &big_output,
            "a0",
        );
        for i in 1..=5 {
            entries.extend(simple_turn(&format!("q{i}"), &format!("a{i}")));
        }

        let before = assembler.estimate_tokens(&entries);
        assert!(before > threshold, "setup: should exceed threshold");

        let result = heuristic_compact(&entries, Path::new("/project"));
        assert!(result.compacted);

        let after = assembler.estimate_tokens(&result.entries);
        assert!(after < before, "compaction should reduce token estimate");
    }

    // -----------------------------------------------------------------------
    // file_write and file_edit placeholders
    // -----------------------------------------------------------------------

    #[test]
    fn old_file_write_result_replaced_with_placeholder() {
        let mut entries = tool_turn(
            "q0",
            "file_write",
            r#"{"path":"out.txt","content":"hello world"}"#,
            "c0",
            "File written successfully (11 bytes).",
            "a0",
        );
        for i in 1..=5 {
            entries.extend(simple_turn(&format!("q{i}"), &format!("a{i}")));
        }

        let result = heuristic_compact(&entries, Path::new("/project"));
        let tr = result.entries.iter().find(|e| {
            matches!(e, TranscriptEntry::ToolResult { call_id, .. } if call_id == "c0")
        });
        if let Some(TranscriptEntry::ToolResult { output, .. }) = tr {
            assert_eq!(output, "[file_write: out.txt]");
        }
    }

    #[test]
    fn old_file_edit_result_replaced_with_placeholder() {
        let mut entries = tool_turn(
            "q0",
            "file_edit",
            r#"{"path":"src/lib.rs","old_string":"foo","new_string":"bar"}"#,
            "c0",
            "Edit applied. Context:\n  bar\n  baz",
            "a0",
        );
        for i in 1..=5 {
            entries.extend(simple_turn(&format!("q{i}"), &format!("a{i}")));
        }

        let result = heuristic_compact(&entries, Path::new("/project"));
        let tr = result.entries.iter().find(|e| {
            matches!(e, TranscriptEntry::ToolResult { call_id, .. } if call_id == "c0")
        });
        if let Some(TranscriptEntry::ToolResult { output, .. }) = tr {
            assert_eq!(output, "[file_edit: src/lib.rs]");
        }
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

        let result = heuristic_compact(&entries, Path::new("/project"));
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

    // -----------------------------------------------------------------------
    // maybe_compact: preflight decision logic
    // -----------------------------------------------------------------------

    #[tokio::test]
    async fn maybe_compact_returns_none_below_threshold() {
        use crate::api::mock::MockGrokApi;
        use serde_json::json;

        let assembler = ContextAssembler::new(json!({"role": "system", "content": "short"}));
        let mock = MockGrokApi::new(vec![]);
        // Small transcript — well below any threshold
        let mut entries: Vec<TranscriptEntry> = Vec::new();
        for i in 0..6 {
            entries.extend(simple_turn(&format!("q{i}"), &format!("a{i}")));
        }

        let result = maybe_compact(&entries, &assembler, "grok-3", Path::new("/project"), &mock).await;
        assert!(result.is_none(), "should not compact when below threshold");
    }

    #[tokio::test]
    async fn maybe_compact_returns_some_above_threshold() {
        use crate::api::mock::MockGrokApi;
        use serde_json::json;

        let assembler = ContextAssembler::new(json!({"role": "system", "content": "short"}));
        let threshold = ModelProfile::for_model("grok-3").compaction_threshold();

        // Build a transcript that exceeds the threshold
        let big_output = "x".repeat(threshold * 4 + 1000);
        let mut entries = tool_turn(
            "q0",
            "file_read",
            r#"{"path":"big.rs"}"#,
            "c0",
            &big_output,
            "a0",
        );
        for i in 1..=5 {
            entries.extend(simple_turn(&format!("q{i}"), &format!("a{i}")));
        }

        let before_estimate = assembler.estimate_tokens(&entries);
        assert!(before_estimate > threshold, "setup: should exceed threshold");

        // Heuristic alone should handle this (big file_read → placeholder)
        let mock = MockGrokApi::new(vec![]);
        let result = maybe_compact(&entries, &assembler, "grok-3", Path::new("/project"), &mock).await;
        assert!(result.is_some(), "should compact when above threshold");

        let compacted = result.unwrap();
        let after_estimate = assembler.estimate_tokens(&compacted.entries);
        assert!(after_estimate < before_estimate, "compaction should reduce estimate");
    }

    #[tokio::test]
    async fn maybe_compact_returns_none_when_above_threshold_but_nothing_to_compact() {
        use crate::api::mock::MockGrokApi;
        use serde_json::json;

        // Use a system prompt large enough to push us over the threshold on its own
        let threshold = ModelProfile::for_model("grok-3").compaction_threshold();
        let big_system = "x".repeat(threshold * 4 + 1000);
        let assembler = ContextAssembler::new(json!({"role": "system", "content": big_system}));
        let mock = MockGrokApi::new(vec![]);

        // Only 3 turns — everything is recent, nothing to compact
        let mut entries: Vec<TranscriptEntry> = Vec::new();
        for i in 0..3 {
            entries.extend(simple_turn(&format!("q{i}"), &format!("a{i}")));
        }

        let estimate = assembler.estimate_tokens(&entries);
        assert!(estimate > threshold, "setup: system overhead should exceed threshold");

        let result = maybe_compact(&entries, &assembler, "grok-3", Path::new("/project"), &mock).await;
        assert!(result.is_none(), "should return None when above threshold but nothing compactable");
    }

    #[tokio::test]
    async fn maybe_compact_escalates_to_llm_when_heuristic_insufficient() {
        use crate::api::TurnResponse;
        use crate::api::mock::MockGrokApi;
        use serde_json::json;

        let threshold = ModelProfile::for_model("grok-3").compaction_threshold();

        // Use a large system prompt so we're over threshold even after heuristic
        // compaction. The heuristic can only compact tool results — user/assistant
        // messages are untouched. So big user messages stay big.
        let big_msg = "x".repeat(threshold * 2); // big enough to stay over threshold
        let assembler = ContextAssembler::new(json!({"role": "system", "content": "short"}));

        let mock = MockGrokApi::new(vec![TurnResponse {
            text: "LLM summary of conversation.".into(),
            tool_calls: vec![],
            usage: Some(crate::api::Usage { input_tokens: 2000, output_tokens: 500 }),
        }]);

        // Build transcript with big user messages that heuristic can't compact
        let mut entries = vec![
            TranscriptEntry::user_message(&big_msg),
            TranscriptEntry::assistant_message(&big_msg),
        ];
        for i in 1..=5 {
            entries.extend(simple_turn(&format!("q{i}"), &format!("a{i}")));
        }

        let result = maybe_compact(&entries, &assembler, "grok-3", Path::new("/project"), &mock).await;
        assert!(result.is_some(), "should compact via LLM");

        let compacted = result.unwrap();
        // Should have LLM usage since it escalated
        assert!(compacted.llm_usage.is_some(), "should have LLM usage");
        // First entry should be a CompactionSummary
        assert!(matches!(&compacted.entries[0], TranscriptEntry::CompactionSummary { .. }));
    }

    #[tokio::test]
    async fn maybe_compact_propagates_usage_from_rejected_llm() {
        use crate::api::TurnResponse;
        use crate::api::mock::MockGrokApi;
        use serde_json::json;

        let threshold = ModelProfile::for_model("grok-3").compaction_threshold();

        // Big user messages that heuristic can't compact
        let big_msg = "x".repeat(threshold * 2);
        let assembler = ContextAssembler::new(json!({"role": "system", "content": "short"}));

        // LLM returns a verbose summary that's LARGER than old entries
        let verbose_summary = "y".repeat(threshold * 4);
        let mock = MockGrokApi::new(vec![TurnResponse {
            text: verbose_summary,
            tool_calls: vec![],
            usage: Some(crate::api::Usage { input_tokens: 5000, output_tokens: 3000 }),
        }]);

        let mut entries = vec![
            TranscriptEntry::user_message(&big_msg),
            TranscriptEntry::assistant_message(&big_msg),
        ];
        for i in 1..=5 {
            entries.extend(simple_turn(&format!("q{i}"), &format!("a{i}")));
        }

        let result = maybe_compact(&entries, &assembler, "grok-3", Path::new("/project"), &mock).await;
        assert!(result.is_some(), "should return Some to carry LLM usage");

        let compacted = result.unwrap();
        // Compaction was rejected (summary too large), but usage is preserved
        assert!(!compacted.compacted, "should not accept verbose summary");
        let usage = compacted.llm_usage.expect("LLM usage should be propagated");
        assert_eq!(usage.input_tokens, 5000);
        assert_eq!(usage.output_tokens, 3000);
    }

    // -----------------------------------------------------------------------
    // llm_compact
    // -----------------------------------------------------------------------

    #[tokio::test]
    async fn llm_compact_replaces_old_entries_with_summary() {
        use crate::api::TurnResponse;
        use crate::api::mock::MockGrokApi;

        let mock = MockGrokApi::new(vec![TurnResponse {
            text: "## Primary Request\nUser wanted help.\n\n## Technical Concepts\nNone.".into(),
            tool_calls: vec![],
            usage: Some(crate::api::Usage { input_tokens: 500, output_tokens: 200 }),
        }]);

        let mut entries: Vec<TranscriptEntry> = Vec::new();
        for i in 0..8 {
            entries.extend(big_turn(i));
        }

        let result = llm_compact(&entries, "grok-3", &mock).await.unwrap();
        assert!(result.compacted);

        // Should have CompactionSummary + recent 5 turns (10 entries)
        assert!(matches!(&result.entries[0], TranscriptEntry::CompactionSummary { summary, .. } if summary.contains("Primary Request")));

        // Recent entries should be preserved verbatim
        let recent_user_msgs: Vec<_> = result.entries.iter().filter(|e| {
            matches!(e, TranscriptEntry::UserMessage { .. })
        }).collect();
        assert_eq!(recent_user_msgs.len(), 5);
    }

    #[tokio::test]
    async fn llm_compact_returns_usage() {
        use crate::api::TurnResponse;
        use crate::api::mock::MockGrokApi;

        let mock = MockGrokApi::new(vec![TurnResponse {
            text: "Summary.".into(),
            tool_calls: vec![],
            usage: Some(crate::api::Usage { input_tokens: 1000, output_tokens: 300 }),
        }]);

        let mut entries: Vec<TranscriptEntry> = Vec::new();
        for i in 0..8 {
            entries.extend(big_turn(i));
        }

        let result = llm_compact(&entries, "grok-3", &mock).await.unwrap();
        let usage = result.llm_usage.unwrap();
        assert_eq!(usage.input_tokens, 1000);
        assert_eq!(usage.output_tokens, 300);
    }

    #[tokio::test]
    async fn llm_compact_no_op_when_all_recent() {
        use crate::api::mock::MockGrokApi;

        let mock = MockGrokApi::new(vec![]);

        let entries: Vec<TranscriptEntry> = (0..3)
            .flat_map(|i| simple_turn(&format!("q{i}"), &format!("a{i}")))
            .collect();

        let result = llm_compact(&entries, "grok-3", &mock).await.unwrap();
        assert!(!result.compacted);
        assert!(result.llm_usage.is_none());
        assert_eq!(result.entries.len(), entries.len());
    }

    #[tokio::test]
    async fn llm_compact_summary_token_estimate_matches_content() {
        use crate::api::TurnResponse;
        use crate::api::mock::MockGrokApi;

        let summary_text = "A detailed summary of the conversation.";
        let mock = MockGrokApi::new(vec![TurnResponse {
            text: summary_text.into(),
            tool_calls: vec![],
            usage: None,
        }]);

        let mut entries: Vec<TranscriptEntry> = Vec::new();
        for i in 0..8 {
            entries.extend(big_turn(i));
        }

        let result = llm_compact(&entries, "grok-3", &mock).await.unwrap();
        if let TranscriptEntry::CompactionSummary { token_estimate, .. } = &result.entries[0] {
            assert_eq!(*token_estimate, summary_text.len() / 4);
        } else {
            panic!("First entry should be CompactionSummary");
        }
    }

    #[tokio::test]
    async fn llm_compact_rejects_verbose_summary() {
        use crate::api::TurnResponse;
        use crate::api::mock::MockGrokApi;

        // The model returns a summary that's LARGER than the old entries
        let verbose_summary = "x".repeat(10_000);
        let mock = MockGrokApi::new(vec![TurnResponse {
            text: verbose_summary,
            tool_calls: vec![],
            usage: Some(crate::api::Usage { input_tokens: 500, output_tokens: 5000 }),
        }]);

        // Small old entries — summary will be bigger
        let mut entries: Vec<TranscriptEntry> = Vec::new();
        for i in 0..8 {
            entries.extend(simple_turn(&format!("q{i}"), &format!("a{i}")));
        }

        let result = llm_compact(&entries, "grok-3", &mock).await.unwrap();
        // Should reject the summary since it's larger
        assert!(!result.compacted, "should reject verbose summary");
        // Usage should still be reported (tokens were consumed)
        assert!(result.llm_usage.is_some());
    }

    // -----------------------------------------------------------------------
    // turn boundary splitting
    // -----------------------------------------------------------------------

    #[test]
    fn turn_boundary_splits_at_user_message() {
        // 3 turns = 6 entries. Midpoint is 3.
        // UserMessages are at indices 0, 2, 4
        let entries: Vec<TranscriptEntry> = (0..3)
            .flat_map(|i| simple_turn(&format!("q{i}"), &format!("a{i}")))
            .collect();

        let split = find_turn_boundary_near_midpoint(&entries);
        // Nearest UserMessage to midpoint 3 is index 2 or 4 (both distance 1)
        assert!(
            matches!(&entries[split], TranscriptEntry::UserMessage { .. }),
            "split should be at a UserMessage, got index {split}"
        );
    }

    #[test]
    fn turn_boundary_does_not_orphan_tool_calls() {
        // Turn with tool calls: User, ToolCall, ToolResult, Assistant = 4 entries
        let mut entries = tool_turn("q0", "file_read", r#"{"path":"a.rs"}"#, "c0", "content", "a0");
        entries.extend(simple_turn("q1", "a1"));
        entries.extend(simple_turn("q2", "a2"));
        // 8 entries total. Midpoint is 4. UserMessages at indices 0, 4, 6.
        // Index 4 is a UserMessage (start of turn q1) — should split there.
        let split = find_turn_boundary_near_midpoint(&entries);
        assert!(
            matches!(&entries[split], TranscriptEntry::UserMessage { .. }),
            "split at {split} should be a UserMessage"
        );
        // Verify neither half has an orphaned ToolCall without its ToolResult
        let older = &entries[..split];
        let newer = &entries[split..];
        for half in [older, newer] {
            let calls: Vec<&str> = half.iter().filter_map(|e| match e {
                TranscriptEntry::ToolCall { call_id, .. } => Some(call_id.as_str()),
                _ => None,
            }).collect();
            let results: Vec<&str> = half.iter().filter_map(|e| match e {
                TranscriptEntry::ToolResult { call_id, .. } => Some(call_id.as_str()),
                _ => None,
            }).collect();
            for call_id in &calls {
                assert!(results.contains(call_id), "orphaned ToolCall {call_id}");
            }
        }
    }

    #[test]
    fn turn_boundary_single_turn_falls_back_to_midpoint() {
        // One big turn: User + ToolCall + ToolResult + Assistant = 4 entries
        let entries = vec![
            TranscriptEntry::user_message("big question"),
            TranscriptEntry::tool_call("c0", "file_read", r#"{"path":"a.rs"}"#),
            TranscriptEntry::tool_result("c0", "file_read", "content"),
            TranscriptEntry::assistant_message("answer"),
        ];

        let split = find_turn_boundary_near_midpoint(&entries);
        // Only UserMessage is at index 0 which is excluded — falls back to midpoint (2)
        assert_eq!(split, 2, "single turn should fall back to raw midpoint");
        assert!(split > 0, "should not produce empty first half");
    }

    #[test]
    fn turn_boundary_fallback_when_no_user_messages() {
        let entries = vec![
            TranscriptEntry::assistant_message("a0"),
            TranscriptEntry::assistant_message("a1"),
            TranscriptEntry::assistant_message("a2"),
            TranscriptEntry::assistant_message("a3"),
        ];
        let split = find_turn_boundary_near_midpoint(&entries);
        assert_eq!(split, 2, "should fall back to raw midpoint");
    }

    // -----------------------------------------------------------------------
    // chunked summarization
    // -----------------------------------------------------------------------

    #[tokio::test]
    async fn llm_compact_uses_chunked_when_old_entries_exceed_80_percent() {
        use crate::api::TurnResponse;
        use crate::api::mock::CapturingMockGrokApi;

        // 80% of 131_072 = 104_857 tokens. We need old entries exceeding this.
        // Each entry token_estimate = content.len() / 4
        // So we need total content > 104_857 * 4 = 419_428 chars in old entries.
        let big_content = "x".repeat(250_000);

        let mock = CapturingMockGrokApi::new(vec![
            // First pass: summarize older half
            TurnResponse {
                text: "First half summary.".into(),
                tool_calls: vec![],
                usage: Some(crate::api::Usage { input_tokens: 5000, output_tokens: 500 }),
            },
            // Second pass: combine with newer half
            TurnResponse {
                text: "Combined summary.".into(),
                tool_calls: vec![],
                usage: Some(crate::api::Usage { input_tokens: 3000, output_tokens: 400 }),
            },
        ]);

        let mut entries = vec![
            TranscriptEntry::user_message(&big_content),
            TranscriptEntry::assistant_message(&big_content),
        ];
        // Add 5 more turns to push the big ones into old region
        for i in 1..=5 {
            entries.extend(simple_turn(&format!("q{i}"), &format!("a{i}")));
        }

        let result = llm_compact(&entries, "grok-3", &mock).await.unwrap();
        assert!(result.compacted);

        // Should have made 2 API calls (chunked)
        let counts = mock.captured_input_counts.lock().unwrap();
        assert_eq!(counts.len(), 2, "chunked summarization should make 2 API calls");

        // Usage should be combined
        let usage = result.llm_usage.unwrap();
        assert_eq!(usage.input_tokens, 8000);
        assert_eq!(usage.output_tokens, 900);

        // Result should have the combined summary
        if let TranscriptEntry::CompactionSummary { summary, .. } = &result.entries[0] {
            assert_eq!(summary, "Combined summary.");
        } else {
            panic!("First entry should be CompactionSummary");
        }
    }

    #[tokio::test]
    async fn llm_compact_uses_single_when_below_overflow() {
        use crate::api::TurnResponse;
        use crate::api::mock::CapturingMockGrokApi;

        let mock = CapturingMockGrokApi::new(vec![
            TurnResponse {
                text: "Single pass summary.".into(),
                tool_calls: vec![],
                usage: Some(crate::api::Usage { input_tokens: 1000, output_tokens: 200 }),
            },
        ]);

        // Entries with enough content so summary is smaller, but below 80% of context window
        let mut entries: Vec<TranscriptEntry> = Vec::new();
        for i in 0..8 {
            entries.extend(big_turn(i));
        }

        let result = llm_compact(&entries, "grok-3", &mock).await.unwrap();
        assert!(result.compacted);

        // Should have made only 1 API call (not chunked)
        let counts = mock.captured_input_counts.lock().unwrap();
        assert_eq!(counts.len(), 1, "should use single pass for small transcripts");
    }

    // -----------------------------------------------------------------------
    // format_entries_for_summarization
    // -----------------------------------------------------------------------

    #[test]
    fn format_entries_includes_all_message_types() {
        let entries = vec![
            TranscriptEntry::user_message("hello"),
            TranscriptEntry::assistant_message("hi there"),
            TranscriptEntry::tool_call("c1", "file_read", r#"{"path":"a.rs"}"#),
            TranscriptEntry::tool_result("c1", "file_read", "fn main() {}"),
            TranscriptEntry::compaction_summary("earlier context"),
            TranscriptEntry::system_event("session started"),
        ];

        let text = format_entries_for_summarization(&entries);
        assert!(text.contains("User: hello"));
        assert!(text.contains("Assistant: hi there"));
        assert!(text.contains("Tool call: file_read"));
        assert!(text.contains("Tool result (file_read): fn main() {}"));
        assert!(text.contains("Previous summary: earlier context"));
        // SystemEvent should be skipped
        assert!(!text.contains("session started"));
    }

    #[test]
    fn format_entries_truncates_long_tool_output() {
        let long_output = "x".repeat(1000);
        let entries = vec![
            TranscriptEntry::tool_result("c1", "file_read", &long_output),
        ];

        let text = format_entries_for_summarization(&entries);
        assert!(text.contains("(truncated)"));
        assert!(text.len() < long_output.len() + 100);
    }

    #[test]
    fn format_entries_empty_transcript() {
        let text = format_entries_for_summarization(&[]);
        assert!(text.is_empty());
    }

    // -----------------------------------------------------------------------
    // SUMMARIZATION_PROMPT sanity
    // -----------------------------------------------------------------------

    #[test]
    fn summarization_prompt_has_all_9_sections() {
        let sections = [
            "## Primary Request",
            "## Technical Concepts",
            "## Files & Code",
            "## Errors & Fixes",
            "## Problem Solving",
            "## User Messages",
            "## Pending Tasks",
            "## Current Work",
            "## Next Step",
        ];
        for section in &sections {
            assert!(
                SUMMARIZATION_PROMPT.contains(section),
                "missing section: {section}"
            );
        }
    }
}
