use std::path::Path;

use crate::checkpoint::{self, RestoreResult};
use crate::session::TranscriptEntry;

/// Result of an undo/rewind operation.
#[derive(Debug)]
pub struct RewindResult {
    /// The transcript entries after rewind (to be persisted).
    pub entries: Vec<TranscriptEntry>,
    /// Per-file restore outcomes.
    pub file_results: Vec<RestoreResult>,
    /// Whether the rewound turn included shell_exec (needs warning).
    pub had_shell_exec: bool,
    /// Number of transcript entries removed.
    pub entries_removed: usize,
    /// Whether we're outside a git repo (code changes can't be restored).
    pub not_in_git: bool,
}

/// Mode for rewind operations.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum RewindMode {
    /// Restore both conversation and code.
    Both,
    /// Restore only code (files), keep conversation.
    CodeOnly,
    /// Restore only conversation (truncate transcript), don't touch files.
    ConversationOnly,
}

/// Find the index of the Nth-from-last UserMessage in the transcript.
/// `n = 1` means the most recent user message, `n = 2` means the one before that, etc.
fn find_nth_user_message_from_end(entries: &[TranscriptEntry], n: usize) -> Option<usize> {
    let mut count = 0;
    for (i, entry) in entries.iter().enumerate().rev() {
        if matches!(entry, TranscriptEntry::UserMessage { .. }) {
            count += 1;
            if count == n {
                return Some(i);
            }
        }
    }
    None
}

/// Undo the most recent agent turn.
///
/// Finds the last checkpoint, restores files (if mode includes code),
/// and truncates transcript to the previous user message boundary (if mode includes conversation).
///
/// **Crash-safe ordering**: files are restored first, transcript is rewritten last.
pub fn undo_last_turn(
    entries: &[TranscriptEntry],
    project_root: &Path,
    mode: RewindMode,
) -> Result<RewindResult, String> {
    if entries.is_empty() {
        return Err("no entries to undo".to_string());
    }

    // Find the last user message — this is where we truncate to (exclusive of everything after)
    let last_user_idx = find_nth_user_message_from_end(entries, 1)
        .ok_or_else(|| "no user messages found".to_string())?;

    // We want to remove the last user message and everything after it
    let truncate_to = last_user_idx;

    let mut file_results = Vec::new();
    let mut had_shell_exec = false;

    // Track whether we're outside a git repo (for warning)
    let not_in_git = !checkpoint::is_git_repo(project_root);

    // Restore files if mode includes code
    if mode != RewindMode::ConversationOnly {
        if mode == RewindMode::CodeOnly && not_in_git {
            return Err(
                "not in a git repo — cannot restore code changes".to_string(),
            );
        }

        // Collect ALL checkpoints in the removed range (there can be multiple
        // if the agent made tool calls across several inner iterations).
        // Restore in reverse order so later edits are undone first.
        let checkpoints_in_range: Vec<usize> = (truncate_to..entries.len())
            .filter(|&i| matches!(&entries[i], TranscriptEntry::Checkpoint { .. }))
            .collect();

        for &cp_idx in checkpoints_in_range.iter().rev() {
            if let TranscriptEntry::Checkpoint {
                snapshots,
                has_shell_exec,
                ..
            } = &entries[cp_idx]
            {
                if *has_shell_exec {
                    had_shell_exec = true;
                }
                for snapshot in snapshots {
                    file_results.push(checkpoint::restore_file(snapshot, project_root));
                }
            }
        }
    }

    // Truncate transcript if mode includes conversation
    let new_entries = if mode != RewindMode::CodeOnly {
        entries[..truncate_to].to_vec()
    } else {
        entries.to_vec()
    };

    let entries_removed = entries.len() - new_entries.len();

    Ok(RewindResult {
        entries: new_entries,
        file_results,
        had_shell_exec,
        entries_removed,
        not_in_git,
    })
}

/// Rewind to a specific user message turn number.
///
/// Turn numbers are 1-indexed: turn 1 is the first user message, turn 2 is the second, etc.
/// The rewind removes everything from that turn's user message onward.
///
/// Restores files from all checkpoints within the removed range (in reverse order).
pub fn rewind_to_turn(
    entries: &[TranscriptEntry],
    turn_number: usize,
    project_root: &Path,
    mode: RewindMode,
) -> Result<RewindResult, String> {
    if turn_number == 0 {
        return Err("turn number must be >= 1".to_string());
    }

    // Find the Nth user message (1-indexed)
    let mut user_msg_count = 0;
    let mut target_idx = None;
    for (i, entry) in entries.iter().enumerate() {
        if matches!(entry, TranscriptEntry::UserMessage { .. }) {
            user_msg_count += 1;
            if user_msg_count == turn_number {
                target_idx = Some(i);
                break;
            }
        }
    }

    let target_idx = target_idx
        .ok_or_else(|| format!("turn {turn_number} not found (only {user_msg_count} turns exist)"))?;

    let truncate_to = target_idx;

    let mut file_results = Vec::new();
    let mut had_shell_exec = false;
    let not_in_git = !checkpoint::is_git_repo(project_root);

    // Restore files from all checkpoints in the removed range (reverse order)
    if mode != RewindMode::ConversationOnly {
        if mode == RewindMode::CodeOnly && not_in_git {
            return Err(
                "not in a git repo — cannot restore code changes".to_string(),
            );
        }

        let checkpoints_in_range: Vec<usize> = (truncate_to..entries.len())
            .filter(|&i| matches!(&entries[i], TranscriptEntry::Checkpoint { .. }))
            .collect();

        for &cp_idx in checkpoints_in_range.iter().rev() {
            if let TranscriptEntry::Checkpoint {
                snapshots,
                has_shell_exec,
                ..
            } = &entries[cp_idx]
            {
                if *has_shell_exec {
                    had_shell_exec = true;
                }
                for snapshot in snapshots {
                    file_results.push(checkpoint::restore_file(snapshot, project_root));
                }
            }
        }
    }

    // Truncate transcript if mode includes conversation
    let new_entries = if mode != RewindMode::CodeOnly {
        entries[..truncate_to].to_vec()
    } else {
        entries.to_vec()
    };

    let entries_removed = entries.len() - new_entries.len();

    Ok(RewindResult {
        entries: new_entries,
        file_results,
        had_shell_exec,
        entries_removed,
        not_in_git,
    })
}

/// Count the number of user message turns in the transcript.
#[allow(dead_code)] // Used in tests and available for UI display
pub fn count_turns(entries: &[TranscriptEntry]) -> usize {
    entries
        .iter()
        .filter(|e| matches!(e, TranscriptEntry::UserMessage { .. }))
        .count()
}

/// Format a rewind result for display to the user.
pub fn format_rewind_result(result: &RewindResult) -> String {
    let mut lines = Vec::new();

    if result.entries_removed > 0 {
        lines.push(format!(
            "  rewound {} transcript entries",
            result.entries_removed
        ));
    }

    for fr in &result.file_results {
        match fr {
            RestoreResult::Restored { path, action } => {
                lines.push(format!("  restored: {} ({})", short_path(path), action));
            }
            RestoreResult::Skipped { path, reason } => {
                lines.push(format!("  skipped: {} — {}", short_path(path), reason));
            }
            RestoreResult::Failed { path, reason } => {
                lines.push(format!("  FAILED: {} — {}", short_path(path), reason));
            }
        }
    }

    if result.not_in_git {
        lines.push(
            "  warning: not in a git repo — code changes cannot be restored".to_string(),
        );
    }

    if result.had_shell_exec {
        lines.push(
            "  warning: this turn included shell_exec — side effects cannot be automatically undone"
                .to_string(),
        );
    }

    if lines.is_empty() {
        "  nothing to undo".to_string()
    } else {
        lines.join("\n")
    }
}

/// Shorten a path for display (last 3 components).
fn short_path(path: &str) -> String {
    let components: Vec<&str> = path.split('/').filter(|s| !s.is_empty()).collect();
    if components.len() <= 3 {
        path.to_string()
    } else {
        format!(".../{}", components[components.len() - 3..].join("/"))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::checkpoint::FileSnapshot;

    // --- Helper ---

    fn simple_turn(user_msg: &str, assistant_msg: &str) -> Vec<TranscriptEntry> {
        vec![
            TranscriptEntry::user_message(user_msg),
            TranscriptEntry::assistant_message(assistant_msg),
        ]
    }

    fn tool_turn_with_checkpoint(
        user_msg: &str,
        snapshots: Vec<FileSnapshot>,
        has_shell_exec: bool,
    ) -> Vec<TranscriptEntry> {
        vec![
            TranscriptEntry::user_message(user_msg),
            TranscriptEntry::tool_call("c1", "file_write", r#"{"path":"test.txt"}"#),
            TranscriptEntry::tool_result("c1", "file_write", "ok"),
            TranscriptEntry::checkpoint(snapshots, has_shell_exec),
            TranscriptEntry::assistant_message("Done"),
        ]
    }

    // --- count_turns ---

    #[test]
    fn count_turns_empty() {
        assert_eq!(count_turns(&[]), 0);
    }

    #[test]
    fn count_turns_multiple() {
        let mut entries = Vec::new();
        entries.extend(simple_turn("q1", "a1"));
        entries.extend(simple_turn("q2", "a2"));
        entries.extend(simple_turn("q3", "a3"));
        assert_eq!(count_turns(&entries), 3);
    }

    // --- undo_last_turn (conversation only) ---

    #[test]
    fn undo_last_turn_removes_last_user_msg_and_after() {
        let mut entries = Vec::new();
        entries.extend(simple_turn("q1", "a1"));
        entries.extend(simple_turn("q2", "a2"));

        let result = undo_last_turn(&entries, Path::new("/tmp"), RewindMode::ConversationOnly)
            .unwrap();

        assert_eq!(result.entries.len(), 2); // only q1 + a1
        assert_eq!(result.entries_removed, 2); // q2 + a2
        assert!(matches!(
            &result.entries[0],
            TranscriptEntry::UserMessage { content, .. } if content == "q1"
        ));
    }

    #[test]
    fn undo_last_turn_empty_transcript() {
        let result = undo_last_turn(&[], Path::new("/tmp"), RewindMode::ConversationOnly);
        assert!(result.is_err());
        assert!(result.unwrap_err().contains("no entries"));
    }

    #[test]
    fn undo_single_turn() {
        let entries = simple_turn("only question", "only answer");
        let result = undo_last_turn(&entries, Path::new("/tmp"), RewindMode::ConversationOnly)
            .unwrap();
        assert!(result.entries.is_empty());
        assert_eq!(result.entries_removed, 2);
    }

    // --- undo_last_turn with checkpoints ---

    #[test]
    fn undo_with_checkpoint_restores_files() {
        // Setup: git repo with a file
        let repo = setup_git_repo();
        let file = repo.path().join("test.txt");
        std::fs::write(&file, "original").unwrap();
        let pre = checkpoint::git_hash_object(&file, repo.path()).unwrap();

        std::fs::write(&file, "modified by agent").unwrap();
        let post = checkpoint::git_hash_object(&file, repo.path()).unwrap();

        let mut entries = simple_turn("q1", "a1");
        entries.extend(tool_turn_with_checkpoint(
            "edit the file",
            vec![FileSnapshot {
                path: file.display().to_string(),
                pre_hash: pre,
                post_hash: post,
            }],
            false,
        ));

        let result = undo_last_turn(&entries, repo.path(), RewindMode::Both).unwrap();

        // File should be restored
        assert_eq!(std::fs::read_to_string(&file).unwrap(), "original");
        assert_eq!(result.entries.len(), 2); // only q1 + a1
        assert_eq!(result.file_results.len(), 1);
        assert!(matches!(&result.file_results[0], RestoreResult::Restored { .. }));
    }

    #[test]
    fn undo_shell_exec_warning() {
        let repo = setup_git_repo();
        let file = repo.path().join("test.txt");
        std::fs::write(&file, "original").unwrap();
        let pre = checkpoint::git_hash_object(&file, repo.path()).unwrap();
        std::fs::write(&file, "modified").unwrap();
        let post = checkpoint::git_hash_object(&file, repo.path()).unwrap();

        let entries = tool_turn_with_checkpoint(
            "do stuff",
            vec![FileSnapshot {
                path: file.display().to_string(),
                pre_hash: pre,
                post_hash: post,
            }],
            true, // has_shell_exec
        );

        let result = undo_last_turn(&entries, repo.path(), RewindMode::Both).unwrap();
        assert!(result.had_shell_exec);
    }

    #[test]
    fn undo_code_only_keeps_transcript() {
        let repo = setup_git_repo();
        let file = repo.path().join("test.txt");
        std::fs::write(&file, "original").unwrap();
        let pre = checkpoint::git_hash_object(&file, repo.path()).unwrap();
        std::fs::write(&file, "modified").unwrap();
        let post = checkpoint::git_hash_object(&file, repo.path()).unwrap();

        let entries = tool_turn_with_checkpoint(
            "edit the file",
            vec![FileSnapshot {
                path: file.display().to_string(),
                pre_hash: pre,
                post_hash: post,
            }],
            false,
        );

        let result = undo_last_turn(&entries, repo.path(), RewindMode::CodeOnly).unwrap();

        // File restored but transcript unchanged
        assert_eq!(std::fs::read_to_string(&file).unwrap(), "original");
        assert_eq!(result.entries.len(), entries.len()); // transcript unchanged
        assert_eq!(result.entries_removed, 0);
    }

    #[test]
    fn undo_multi_checkpoint_single_turn() {
        // Simulates a user turn where the model makes tool calls across two
        // inner iterations, producing two separate Checkpoint entries.
        let repo = setup_git_repo();
        let file1 = repo.path().join("a.txt");
        let file2 = repo.path().join("b.txt");

        std::fs::write(&file1, "a_original").unwrap();
        let pre1 = checkpoint::git_hash_object(&file1, repo.path()).unwrap();
        std::fs::write(&file1, "a_modified").unwrap();
        let post1 = checkpoint::git_hash_object(&file1, repo.path()).unwrap();

        std::fs::write(&file2, "b_original").unwrap();
        let pre2 = checkpoint::git_hash_object(&file2, repo.path()).unwrap();
        std::fs::write(&file2, "b_modified").unwrap();
        let post2 = checkpoint::git_hash_object(&file2, repo.path()).unwrap();

        // One user turn with two checkpoints (two model iterations)
        let entries = vec![
            TranscriptEntry::user_message("edit both files"),
            TranscriptEntry::tool_call("c1", "file_write", r#"{"path":"a.txt"}"#),
            TranscriptEntry::tool_result("c1", "file_write", "ok"),
            TranscriptEntry::checkpoint(
                vec![FileSnapshot {
                    path: file1.display().to_string(),
                    pre_hash: pre1,
                    post_hash: post1,
                }],
                false,
            ),
            TranscriptEntry::assistant_message("Edited a.txt, now editing b.txt"),
            TranscriptEntry::tool_call("c2", "file_write", r#"{"path":"b.txt"}"#),
            TranscriptEntry::tool_result("c2", "file_write", "ok"),
            TranscriptEntry::checkpoint(
                vec![FileSnapshot {
                    path: file2.display().to_string(),
                    pre_hash: pre2,
                    post_hash: post2,
                }],
                false,
            ),
            TranscriptEntry::assistant_message("Done"),
        ];

        let result = undo_last_turn(&entries, repo.path(), RewindMode::Both).unwrap();

        // Both files should be restored
        assert_eq!(std::fs::read_to_string(&file1).unwrap(), "a_original");
        assert_eq!(std::fs::read_to_string(&file2).unwrap(), "b_original");
        assert_eq!(result.file_results.len(), 2);
        assert!(result.entries.is_empty()); // no prior turns
    }

    #[test]
    fn undo_multi_checkpoint_shell_exec_in_earlier_iteration() {
        // shell_exec happens in first iteration, file_write in second.
        // undo should still see the shell_exec warning.
        let repo = setup_git_repo();
        let file = repo.path().join("test.txt");
        std::fs::write(&file, "original").unwrap();
        let pre = checkpoint::git_hash_object(&file, repo.path()).unwrap();
        std::fs::write(&file, "modified").unwrap();
        let post = checkpoint::git_hash_object(&file, repo.path()).unwrap();

        let entries = vec![
            TranscriptEntry::user_message("do stuff"),
            // First iteration: shell_exec + file_write
            TranscriptEntry::tool_call("c1", "file_write", r#"{"path":"test.txt"}"#),
            TranscriptEntry::tool_result("c1", "file_write", "ok"),
            TranscriptEntry::checkpoint(
                vec![FileSnapshot {
                    path: file.display().to_string(),
                    pre_hash: pre,
                    post_hash: post,
                }],
                true, // has_shell_exec from this iteration
            ),
            TranscriptEntry::assistant_message("Done"),
        ];

        let result = undo_last_turn(&entries, repo.path(), RewindMode::Both).unwrap();
        assert!(result.had_shell_exec);
    }

    // --- rewind_to_turn ---

    #[test]
    fn rewind_to_turn_1_removes_all() {
        let mut entries = Vec::new();
        entries.extend(simple_turn("q1", "a1"));
        entries.extend(simple_turn("q2", "a2"));
        entries.extend(simple_turn("q3", "a3"));

        let result =
            rewind_to_turn(&entries, 1, Path::new("/tmp"), RewindMode::ConversationOnly).unwrap();
        assert!(result.entries.is_empty());
        assert_eq!(result.entries_removed, 6);
    }

    #[test]
    fn rewind_to_turn_2_keeps_first() {
        let mut entries = Vec::new();
        entries.extend(simple_turn("q1", "a1"));
        entries.extend(simple_turn("q2", "a2"));
        entries.extend(simple_turn("q3", "a3"));

        let result =
            rewind_to_turn(&entries, 2, Path::new("/tmp"), RewindMode::ConversationOnly).unwrap();
        assert_eq!(result.entries.len(), 2); // q1 + a1
        assert_eq!(result.entries_removed, 4);
    }

    #[test]
    fn rewind_to_nonexistent_turn() {
        let entries = simple_turn("q1", "a1");
        let result = rewind_to_turn(&entries, 5, Path::new("/tmp"), RewindMode::ConversationOnly);
        assert!(result.is_err());
        assert!(result.unwrap_err().contains("turn 5 not found"));
    }

    #[test]
    fn rewind_to_turn_zero() {
        let result = rewind_to_turn(&[], 0, Path::new("/tmp"), RewindMode::ConversationOnly);
        assert!(result.is_err());
        assert!(result.unwrap_err().contains("must be >= 1"));
    }

    #[test]
    fn rewind_to_turn_non_git_both_mode() {
        let dir = tempfile::tempdir().unwrap(); // not a git repo
        let mut entries = Vec::new();
        entries.extend(simple_turn("q1", "a1"));
        entries.extend(simple_turn("q2", "a2"));
        entries.extend(simple_turn("q3", "a3"));

        let result = rewind_to_turn(&entries, 2, dir.path(), RewindMode::Both).unwrap();
        assert_eq!(result.entries.len(), 2); // q1 + a1
        assert!(result.not_in_git);
    }

    #[test]
    fn rewind_to_turn_non_git_code_only_errors() {
        let dir = tempfile::tempdir().unwrap();
        let mut entries = Vec::new();
        entries.extend(simple_turn("q1", "a1"));
        entries.extend(simple_turn("q2", "a2"));

        let result = rewind_to_turn(&entries, 1, dir.path(), RewindMode::CodeOnly);
        assert!(result.is_err());
        assert!(result.unwrap_err().contains("not in a git repo"));
    }

    #[test]
    fn rewind_to_turn_non_git_conversation_only_works() {
        let dir = tempfile::tempdir().unwrap();
        let mut entries = Vec::new();
        entries.extend(simple_turn("q1", "a1"));
        entries.extend(simple_turn("q2", "a2"));

        let result =
            rewind_to_turn(&entries, 2, dir.path(), RewindMode::ConversationOnly).unwrap();
        assert_eq!(result.entries.len(), 2);
        // not_in_git is true (we're not in a git repo), but operation succeeds
        assert!(result.not_in_git);
    }

    #[test]
    fn rewind_restores_multiple_checkpoints() {
        let repo = setup_git_repo();

        // File 1: turn 2
        let file1 = repo.path().join("a.txt");
        std::fs::write(&file1, "a_original").unwrap();
        let pre1 = checkpoint::git_hash_object(&file1, repo.path()).unwrap();
        std::fs::write(&file1, "a_modified").unwrap();
        let post1 = checkpoint::git_hash_object(&file1, repo.path()).unwrap();

        // File 2: turn 3
        let file2 = repo.path().join("b.txt");
        std::fs::write(&file2, "b_original").unwrap();
        let pre2 = checkpoint::git_hash_object(&file2, repo.path()).unwrap();
        std::fs::write(&file2, "b_modified").unwrap();
        let post2 = checkpoint::git_hash_object(&file2, repo.path()).unwrap();

        let mut entries = Vec::new();
        entries.extend(simple_turn("q1", "a1"));
        entries.extend(tool_turn_with_checkpoint(
            "edit a",
            vec![FileSnapshot {
                path: file1.display().to_string(),
                pre_hash: pre1,
                post_hash: post1,
            }],
            false,
        ));
        entries.extend(tool_turn_with_checkpoint(
            "edit b",
            vec![FileSnapshot {
                path: file2.display().to_string(),
                pre_hash: pre2,
                post_hash: post2,
            }],
            false,
        ));

        // Rewind to turn 2 — should restore both files
        let result = rewind_to_turn(&entries, 2, repo.path(), RewindMode::Both).unwrap();

        assert_eq!(std::fs::read_to_string(&file1).unwrap(), "a_original");
        assert_eq!(std::fs::read_to_string(&file2).unwrap(), "b_original");
        assert_eq!(result.file_results.len(), 2);
        assert_eq!(result.entries.len(), 2); // only q1 + a1
    }

    // --- format_rewind_result ---

    #[test]
    fn format_empty_result() {
        let result = RewindResult {
            entries: vec![],
            file_results: vec![],
            had_shell_exec: false,
            entries_removed: 0,
            not_in_git: false,
        };
        assert_eq!(format_rewind_result(&result), "  nothing to undo");
    }

    #[test]
    fn format_with_shell_warning() {
        let result = RewindResult {
            entries: vec![],
            file_results: vec![],
            had_shell_exec: true,
            entries_removed: 3,
            not_in_git: false,
        };
        let formatted = format_rewind_result(&result);
        assert!(formatted.contains("shell_exec"));
        assert!(formatted.contains("3 transcript entries"));
    }

    // --- non-git repo ---

    #[test]
    fn undo_non_git_conversation_only_works() {
        let dir = tempfile::tempdir().unwrap(); // not a git repo
        let mut entries = Vec::new();
        entries.extend(simple_turn("q1", "a1"));
        entries.extend(simple_turn("q2", "a2"));

        let result =
            undo_last_turn(&entries, dir.path(), RewindMode::ConversationOnly).unwrap();
        assert_eq!(result.entries.len(), 2);
        assert_eq!(result.entries_removed, 2);
    }

    #[test]
    fn undo_non_git_both_mode_graceful() {
        // In non-git repo with Both mode: should still undo conversation,
        // just can't restore code (sets not_in_git flag)
        let dir = tempfile::tempdir().unwrap();
        let mut entries = Vec::new();
        entries.extend(simple_turn("q1", "a1"));
        entries.extend(simple_turn("q2", "a2"));

        let result = undo_last_turn(&entries, dir.path(), RewindMode::Both).unwrap();
        assert_eq!(result.entries.len(), 2);
        assert!(result.not_in_git);
    }

    #[test]
    fn undo_non_git_code_only_errors() {
        let dir = tempfile::tempdir().unwrap();
        let entries = simple_turn("q1", "a1");

        let result = undo_last_turn(&entries, dir.path(), RewindMode::CodeOnly);
        assert!(result.is_err());
        assert!(result.unwrap_err().contains("not in a git repo"));
    }

    // --- no checkpoints ---

    #[test]
    fn undo_no_checkpoint_conversation_still_works() {
        // Turns without file modifications have no checkpoint entry
        let repo = setup_git_repo();
        let mut entries = Vec::new();
        entries.extend(simple_turn("q1", "a1"));
        entries.extend(simple_turn("q2", "a2"));

        let result = undo_last_turn(&entries, repo.path(), RewindMode::Both).unwrap();
        assert_eq!(result.entries.len(), 2);
        assert!(result.file_results.is_empty()); // no files to restore
    }

    // --- format with not_in_git warning ---

    #[test]
    fn format_with_not_in_git_warning() {
        let result = RewindResult {
            entries: vec![],
            file_results: vec![],
            had_shell_exec: false,
            entries_removed: 2,
            not_in_git: true,
        };
        let formatted = format_rewind_result(&result);
        assert!(formatted.contains("not in a git repo"));
    }

    // --- short_path ---

    #[test]
    fn short_path_short() {
        assert_eq!(short_path("src/main.rs"), "src/main.rs");
    }

    #[test]
    fn short_path_long() {
        assert_eq!(
            short_path("/home/user/projects/grox/src/main.rs"),
            ".../grox/src/main.rs"
        );
    }

    // --- Helper to create git repos for tests ---

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
}
