use std::io::{BufRead, Write};
use std::path::{Path, PathBuf};

use anyhow::{Context, Result};

use crate::checkpoint::FileSnapshot;

/// Fsync a directory to ensure rename durability on crash.
fn sync_directory(dir: &Path) -> Result<()> {
    let f = std::fs::File::open(dir)
        .with_context(|| format!("Failed to open directory for sync: {}", dir.display()))?;
    f.sync_all()
        .with_context(|| format!("Failed to sync directory: {}", dir.display()))?;
    Ok(())
}
use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};
use uuid::Uuid;

// ---------------------------------------------------------------------------
// TranscriptEntry
// ---------------------------------------------------------------------------

/// A single entry in the conversation transcript.
///
/// Each variant carries a `token_estimate` field (chars / 4 heuristic)
/// used for preflight budget checks.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(tag = "type")]
pub enum TranscriptEntry {
    UserMessage {
        content: String,
        token_estimate: usize,
    },
    AssistantMessage {
        content: String,
        token_estimate: usize,
    },
    ToolCall {
        call_id: String,
        name: String,
        arguments: String,
        token_estimate: usize,
    },
    ToolResult {
        call_id: String,
        name: String,
        output: String,
        token_estimate: usize,
    },
    CompactionSummary {
        summary: String,
        token_estimate: usize,
    },
    SystemEvent {
        event: String,
        token_estimate: usize,
    },
    Checkpoint {
        snapshots: Vec<FileSnapshot>,
        has_shell_exec: bool,
        token_estimate: usize,
    },
}

impl TranscriptEntry {
    pub fn token_estimate(&self) -> usize {
        match self {
            Self::UserMessage { token_estimate, .. }
            | Self::AssistantMessage { token_estimate, .. }
            | Self::ToolCall { token_estimate, .. }
            | Self::ToolResult { token_estimate, .. }
            | Self::CompactionSummary { token_estimate, .. }
            | Self::SystemEvent { token_estimate, .. }
            | Self::Checkpoint { token_estimate, .. } => *token_estimate,
        }
    }

    /// Create a UserMessage with auto-calculated token estimate.
    pub fn user_message(content: impl Into<String>) -> Self {
        let content = content.into();
        let token_estimate = content.len() / 4;
        Self::UserMessage { content, token_estimate }
    }

    /// Create an AssistantMessage with auto-calculated token estimate.
    pub fn assistant_message(content: impl Into<String>) -> Self {
        let content = content.into();
        let token_estimate = content.len() / 4;
        Self::AssistantMessage { content, token_estimate }
    }

    /// Create a ToolCall with auto-calculated token estimate.
    pub fn tool_call(call_id: impl Into<String>, name: impl Into<String>, arguments: impl Into<String>) -> Self {
        let call_id = call_id.into();
        let name = name.into();
        let arguments = arguments.into();
        let token_estimate = (call_id.len() + name.len() + arguments.len()) / 4;
        Self::ToolCall { call_id, name, arguments, token_estimate }
    }

    /// Create a ToolResult with auto-calculated token estimate.
    pub fn tool_result(call_id: impl Into<String>, name: impl Into<String>, output: impl Into<String>) -> Self {
        let call_id = call_id.into();
        let name = name.into();
        let output = output.into();
        let token_estimate = (call_id.len() + name.len() + output.len()) / 4;
        Self::ToolResult { call_id, name, output, token_estimate }
    }

    /// Create a CompactionSummary with auto-calculated token estimate.
    #[allow(dead_code)] // Used in Phase 6 (LLM compaction)
    pub fn compaction_summary(summary: impl Into<String>) -> Self {
        let summary = summary.into();
        let token_estimate = summary.len() / 4;
        Self::CompactionSummary { summary, token_estimate }
    }

    /// Create a SystemEvent with auto-calculated token estimate.
    #[allow(dead_code)] // Used in Phase 7 (rewind checkpoints)
    pub fn system_event(event: impl Into<String>) -> Self {
        let event = event.into();
        let token_estimate = event.len() / 4;
        Self::SystemEvent { event, token_estimate }
    }

    /// Create a Checkpoint with auto-calculated token estimate.
    pub fn checkpoint(snapshots: Vec<FileSnapshot>, has_shell_exec: bool) -> Self {
        // Estimate based on serialized snapshot data
        let content_len: usize = snapshots.iter().map(|s| {
            s.path.len() + s.pre_hash.len() + s.post_hash.len()
        }).sum();
        let token_estimate = content_len / 4;
        Self::Checkpoint { snapshots, has_shell_exec, token_estimate }
    }
}

// ---------------------------------------------------------------------------
// Transcript (JSONL persistence)
// ---------------------------------------------------------------------------

/// Manages reading and writing transcript entries to a JSONL file.
pub struct Transcript {
    path: PathBuf,
}

impl Transcript {
    /// Open (or create) a transcript at the given path.
    pub fn new(path: impl Into<PathBuf>) -> Self {
        Self { path: path.into() }
    }

    /// Create the transcript file on disk (and parent directories) if it doesn't exist.
    /// This ensures the file exists at session startup, not just on first append.
    pub fn create(&self) -> Result<()> {
        if let Some(parent) = self.path.parent() {
            std::fs::create_dir_all(parent)
                .with_context(|| format!("Failed to create directory: {}", parent.display()))?;
        }
        std::fs::OpenOptions::new()
            .create(true)
            .append(true)
            .open(&self.path)
            .with_context(|| format!("Failed to create transcript: {}", self.path.display()))?;
        Ok(())
    }

    pub fn path(&self) -> &Path {
        &self.path
    }

    /// Append a single entry to the transcript file.
    /// Creates the file (and parent directories) if it doesn't exist.
    pub fn append(&self, entry: &TranscriptEntry) -> Result<()> {
        if let Some(parent) = self.path.parent() {
            std::fs::create_dir_all(parent)
                .with_context(|| format!("Failed to create directory: {}", parent.display()))?;
        }
        let mut file = std::fs::OpenOptions::new()
            .create(true)
            .append(true)
            .open(&self.path)
            .with_context(|| format!("Failed to open transcript: {}", self.path.display()))?;

        let line = serde_json::to_string(entry)
            .context("Failed to serialize transcript entry")?;
        writeln!(file, "{line}")
            .with_context(|| format!("Failed to write to transcript: {}", self.path.display()))?;
        file.sync_all()
            .with_context(|| format!("Failed to sync transcript: {}", self.path.display()))?;
        Ok(())
    }

    /// Read all entries from the transcript file.
    /// Crash recovery: trailing incomplete JSON line is silently discarded.
    /// Empty or missing file returns an empty vec.
    pub fn read_all(&self) -> Result<Vec<TranscriptEntry>> {
        if !self.path.exists() {
            return Ok(Vec::new());
        }

        let file = std::fs::File::open(&self.path)
            .with_context(|| format!("Failed to open transcript: {}", self.path.display()))?;

        let reader = std::io::BufReader::new(file);
        let mut entries = Vec::new();

        for line in reader.lines() {
            let line = line.with_context(|| "Failed to read line from transcript")?;
            let line = line.trim();
            if line.is_empty() {
                continue;
            }
            match serde_json::from_str::<TranscriptEntry>(line) {
                Ok(entry) => entries.push(entry),
                Err(_) => {
                    // Trailing incomplete line — silently discard (crash recovery)
                    break;
                }
            }
        }

        Ok(entries)
    }

    /// Atomically rewrite the transcript with a new set of entries.
    /// Writes to a temp file first, then renames over the original.
    pub fn atomic_rewrite(&self, entries: &[TranscriptEntry]) -> Result<()> {
        if let Some(parent) = self.path.parent() {
            std::fs::create_dir_all(parent)
                .with_context(|| format!("Failed to create directory: {}", parent.display()))?;
        }

        let tmp_path = self.path.with_extension("jsonl.tmp");
        {
            let mut file = std::fs::File::create(&tmp_path)
                .with_context(|| format!("Failed to create temp file: {}", tmp_path.display()))?;
            for entry in entries {
                let line = serde_json::to_string(entry)
                    .context("Failed to serialize transcript entry")?;
                writeln!(file, "{line}")?;
            }
            file.sync_all()?;
        }

        std::fs::rename(&tmp_path, &self.path)
            .with_context(|| format!("Failed to rename temp file to: {}", self.path.display()))?;

        // Fsync the parent directory to ensure the rename is durable
        if let Some(parent) = self.path.parent() {
            sync_directory(parent)?;
        }

        Ok(())
    }
}

// ---------------------------------------------------------------------------
// SessionMeta
// ---------------------------------------------------------------------------

/// Metadata sidecar for a session. Stored as `{uuid}.meta.json`.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct SessionMeta {
    pub session_id: String,
    pub start_time: DateTime<Utc>,
    pub model: String,
    pub project_root: String,
    pub cumulative_input_tokens: u64,
    pub cumulative_output_tokens: u64,
    pub last_active: DateTime<Utc>,
    pub summary: String,
}

impl SessionMeta {
    /// Create a new session metadata record.
    pub fn new(model: impl Into<String>, project_root: impl Into<String>) -> Self {
        let now = Utc::now();
        Self {
            session_id: Uuid::new_v4().to_string(),
            start_time: now,
            model: model.into(),
            project_root: project_root.into(),
            cumulative_input_tokens: 0,
            cumulative_output_tokens: 0,
            last_active: now,
            summary: String::new(),
        }
    }

    /// Path for this session's metadata file within a sessions directory.
    pub fn meta_path(sessions_dir: &Path, session_id: &str) -> PathBuf {
        sessions_dir.join(format!("{session_id}.meta.json"))
    }

    /// Path for this session's transcript file within a sessions directory.
    pub fn transcript_path(sessions_dir: &Path, session_id: &str) -> PathBuf {
        sessions_dir.join(format!("{session_id}.jsonl"))
    }

    /// Atomically save this metadata to disk (write-to-temp-then-rename).
    pub fn save(&self, sessions_dir: &Path) -> Result<()> {
        std::fs::create_dir_all(sessions_dir)
            .with_context(|| format!("Failed to create sessions dir: {}", sessions_dir.display()))?;

        let path = Self::meta_path(sessions_dir, &self.session_id);
        let tmp_path = path.with_extension("meta.json.tmp");

        let json = serde_json::to_string_pretty(self)
            .context("Failed to serialize session metadata")?;

        {
            let file = std::fs::File::create(&tmp_path)
                .with_context(|| format!("Failed to create temp file: {}", tmp_path.display()))?;
            let mut writer = std::io::BufWriter::new(file);
            writer.write_all(json.as_bytes())
                .with_context(|| format!("Failed to write temp file: {}", tmp_path.display()))?;
            writer.into_inner()
                .map_err(|e| anyhow::anyhow!("Failed to flush temp file: {}", e))?
                .sync_all()
                .with_context(|| format!("Failed to sync temp file: {}", tmp_path.display()))?;
        }

        std::fs::rename(&tmp_path, &path)
            .with_context(|| format!("Failed to rename temp file to: {}", path.display()))?;

        // Fsync the parent directory to ensure the rename is durable
        sync_directory(sessions_dir)?;

        Ok(())
    }

    /// Load session metadata from disk.
    pub fn load(sessions_dir: &Path, session_id: &str) -> Result<Self> {
        let path = Self::meta_path(sessions_dir, session_id);
        let content = std::fs::read_to_string(&path)
            .with_context(|| format!("Failed to read session metadata: {}", path.display()))?;
        let meta: Self = serde_json::from_str(&content)
            .with_context(|| format!("Failed to parse session metadata: {}", path.display()))?;
        Ok(meta)
    }
}

// ---------------------------------------------------------------------------
// SessionIndex
// ---------------------------------------------------------------------------

/// Lists and filters sessions from the sessions directory.
pub struct SessionIndex;

impl SessionIndex {
    /// Default sessions directory: `~/.grox/sessions/`
    pub fn default_sessions_dir() -> Result<PathBuf> {
        let home = std::env::var("HOME")
            .or_else(|_| std::env::var("USERPROFILE"))
            .context("Could not determine home directory")?;
        Ok(PathBuf::from(home).join(".grox").join("sessions"))
    }

    /// List all sessions, sorted by last_active (most recent first).
    /// Creates the sessions directory if it doesn't exist.
    pub fn list(sessions_dir: &Path) -> Result<Vec<SessionMeta>> {
        std::fs::create_dir_all(sessions_dir)
            .with_context(|| format!("Failed to create sessions dir: {}", sessions_dir.display()))?;

        let mut sessions = Vec::new();

        for entry in std::fs::read_dir(sessions_dir)
            .with_context(|| format!("Failed to read sessions dir: {}", sessions_dir.display()))?
        {
            let entry = entry?;
            let path = entry.path();
            if path.extension().is_some_and(|ext| ext == "json")
                && path.to_string_lossy().contains(".meta.")
            {
                match std::fs::read_to_string(&path) {
                    Ok(content) => {
                        if let Ok(meta) = serde_json::from_str::<SessionMeta>(&content) {
                            sessions.push(meta);
                        }
                    }
                    Err(_) => continue,
                }
            }
        }

        sessions.sort_by(|a, b| b.last_active.cmp(&a.last_active));
        Ok(sessions)
    }

    /// List sessions filtered to a specific project root.
    pub fn list_for_project(sessions_dir: &Path, project_root: &str) -> Result<Vec<SessionMeta>> {
        let all = Self::list(sessions_dir)?;
        Ok(all.into_iter().filter(|s| s.project_root == project_root).collect())
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use crate::checkpoint::FileSnapshot;
    use tempfile::tempdir;

    // --- TranscriptEntry ---

    #[test]
    fn user_message_token_estimate() {
        let entry = TranscriptEntry::user_message("hello world"); // 11 chars
        assert_eq!(entry.token_estimate(), 11 / 4);
    }

    #[test]
    fn assistant_message_token_estimate() {
        let entry = TranscriptEntry::assistant_message("I can help with that.");
        assert_eq!(entry.token_estimate(), 21 / 4);
    }

    #[test]
    fn tool_call_token_estimate() {
        let entry = TranscriptEntry::tool_call("call_1", "file_read", r#"{"path":"src/main.rs"}"#);
        // (6 + 9 + 22) / 4 = 37 / 4 = 9
        assert_eq!(entry.token_estimate(), 37 / 4);
    }

    #[test]
    fn tool_result_token_estimate() {
        let entry = TranscriptEntry::tool_result("call_1", "file_read", "fn main() {}");
        // "call_1" (6) + "file_read" (9) + "fn main() {}" (13) = 28 / 4 = 7
        let expected = ("call_1".len() + "file_read".len() + "fn main() {}".len()) / 4;
        assert_eq!(entry.token_estimate(), expected);
    }

    #[test]
    fn compaction_summary_token_estimate() {
        let text = "x".repeat(400);
        let entry = TranscriptEntry::compaction_summary(&text);
        assert_eq!(entry.token_estimate(), 100);
    }

    #[test]
    fn system_event_token_estimate() {
        let entry = TranscriptEntry::system_event("session started");
        assert_eq!(entry.token_estimate(), 15 / 4);
    }

    #[test]
    fn all_variants_serialize_roundtrip() {
        let entries = vec![
            TranscriptEntry::user_message("hello"),
            TranscriptEntry::assistant_message("hi there"),
            TranscriptEntry::tool_call("c1", "grep", r#"{"pattern":"foo"}"#),
            TranscriptEntry::tool_result("c1", "grep", "src/lib.rs:10: foo"),
            TranscriptEntry::compaction_summary("Summary of conversation so far."),
            TranscriptEntry::system_event("session started"),
            TranscriptEntry::checkpoint(
                vec![FileSnapshot {
                    path: "src/main.rs".to_string(),
                    pre_hash: "abc123".to_string(),
                    post_hash: "def456".to_string(),
                }],
                false,
            ),
        ];

        for entry in &entries {
            let json = serde_json::to_string(entry).unwrap();
            let parsed: TranscriptEntry = serde_json::from_str(&json).unwrap();
            assert_eq!(&parsed, entry);
        }
    }

    #[test]
    fn serde_tags_distinguish_variants() {
        let user = TranscriptEntry::user_message("test");
        let json = serde_json::to_string(&user).unwrap();
        assert!(json.contains(r#""type":"UserMessage""#));

        let tool = TranscriptEntry::tool_call("c1", "grep", "{}");
        let json = serde_json::to_string(&tool).unwrap();
        assert!(json.contains(r#""type":"ToolCall""#));
    }

    // --- Checkpoint ---

    #[test]
    fn checkpoint_token_estimate() {
        let entry = TranscriptEntry::checkpoint(
            vec![FileSnapshot {
                path: "src/main.rs".to_string(),      // 11
                pre_hash: "a".repeat(40),              // 40
                post_hash: "b".repeat(40),             // 40
            }],
            false,
        );
        assert_eq!(entry.token_estimate(), (11 + 40 + 40) / 4);
    }

    #[test]
    fn checkpoint_serde_tag() {
        let entry = TranscriptEntry::checkpoint(vec![], false);
        let json = serde_json::to_string(&entry).unwrap();
        assert!(json.contains(r#""type":"Checkpoint""#));
    }

    #[test]
    fn checkpoint_with_shell_exec_flag() {
        let entry = TranscriptEntry::checkpoint(vec![], true);
        if let TranscriptEntry::Checkpoint { has_shell_exec, .. } = entry {
            assert!(has_shell_exec);
        } else {
            panic!("Expected Checkpoint variant");
        }
    }

    #[test]
    fn checkpoint_jsonl_roundtrip() {
        let dir = tempdir().unwrap();
        let path = dir.path().join("checkpoint.jsonl");
        let transcript = Transcript::new(&path);

        let entry = TranscriptEntry::checkpoint(
            vec![
                FileSnapshot {
                    path: "src/a.rs".to_string(),
                    pre_hash: "aaa".to_string(),
                    post_hash: "bbb".to_string(),
                },
                FileSnapshot {
                    path: "src/b.rs".to_string(),
                    pre_hash: "created".to_string(),
                    post_hash: "ccc".to_string(),
                },
            ],
            true,
        );

        transcript.append(&entry).unwrap();
        let loaded = transcript.read_all().unwrap();
        assert_eq!(loaded.len(), 1);
        assert_eq!(loaded[0], entry);
    }

    // --- Transcript JSONL ---

    #[test]
    fn append_and_read_roundtrip() {
        let dir = tempdir().unwrap();
        let path = dir.path().join("test.jsonl");
        let transcript = Transcript::new(&path);

        let entries = vec![
            TranscriptEntry::user_message("hello"),
            TranscriptEntry::assistant_message("hi"),
            TranscriptEntry::tool_call("c1", "file_read", r#"{"path":"a.rs"}"#),
        ];

        for entry in &entries {
            transcript.append(entry).unwrap();
        }

        let loaded = transcript.read_all().unwrap();
        assert_eq!(loaded, entries);
    }

    #[test]
    fn read_empty_file_returns_empty_vec() {
        let dir = tempdir().unwrap();
        let path = dir.path().join("empty.jsonl");
        std::fs::write(&path, "").unwrap();

        let transcript = Transcript::new(&path);
        let entries = transcript.read_all().unwrap();
        assert!(entries.is_empty());
    }

    #[test]
    fn read_missing_file_returns_empty_vec() {
        let dir = tempdir().unwrap();
        let path = dir.path().join("nonexistent.jsonl");

        let transcript = Transcript::new(&path);
        let entries = transcript.read_all().unwrap();
        assert!(entries.is_empty());
    }

    #[test]
    fn crash_recovery_trailing_incomplete_line() {
        let dir = tempdir().unwrap();
        let path = dir.path().join("crash.jsonl");

        let entry = TranscriptEntry::user_message("valid entry");
        let valid_line = serde_json::to_string(&entry).unwrap();

        // Write a valid line followed by an incomplete line (simulating crash)
        std::fs::write(&path, format!("{valid_line}\n{{\"type\":\"UserMess")).unwrap();

        let transcript = Transcript::new(&path);
        let entries = transcript.read_all().unwrap();
        assert_eq!(entries.len(), 1);
        assert_eq!(entries[0], entry);
    }

    #[test]
    fn crash_recovery_only_incomplete_line() {
        let dir = tempdir().unwrap();
        let path = dir.path().join("crash2.jsonl");
        std::fs::write(&path, r#"{"type":"broken"#).unwrap();

        let transcript = Transcript::new(&path);
        let entries = transcript.read_all().unwrap();
        assert!(entries.is_empty());
    }

    #[test]
    fn atomic_rewrite_replaces_content() {
        let dir = tempdir().unwrap();
        let path = dir.path().join("rewrite.jsonl");
        let transcript = Transcript::new(&path);

        // Write initial entries
        transcript.append(&TranscriptEntry::user_message("old1")).unwrap();
        transcript.append(&TranscriptEntry::user_message("old2")).unwrap();
        transcript.append(&TranscriptEntry::user_message("old3")).unwrap();

        // Atomic rewrite with fewer entries
        let new_entries = vec![
            TranscriptEntry::compaction_summary("summary of old conversation"),
            TranscriptEntry::user_message("latest"),
        ];
        transcript.atomic_rewrite(&new_entries).unwrap();

        let loaded = transcript.read_all().unwrap();
        assert_eq!(loaded, new_entries);
    }

    #[test]
    fn atomic_rewrite_interrupted_leaves_original() {
        let dir = tempdir().unwrap();
        let path = dir.path().join("safe.jsonl");
        let transcript = Transcript::new(&path);

        let original = vec![TranscriptEntry::user_message("original")];
        transcript.append(&original[0]).unwrap();

        // Simulate interrupted rewrite: temp file exists but rename didn't happen
        let tmp_path = path.with_extension("jsonl.tmp");
        std::fs::write(&tmp_path, "garbage").unwrap();

        // Original should still be readable
        let loaded = transcript.read_all().unwrap();
        assert_eq!(loaded, original);
    }

    #[test]
    fn atomic_rewrite_cleans_up_temp_file() {
        let dir = tempdir().unwrap();
        let path = dir.path().join("cleanup.jsonl");
        let transcript = Transcript::new(&path);

        transcript.atomic_rewrite(&[TranscriptEntry::user_message("test")]).unwrap();

        let tmp_path = path.with_extension("jsonl.tmp");
        assert!(!tmp_path.exists(), "temp file should not exist after successful rewrite");
    }

    #[test]
    fn append_creates_parent_directories() {
        let dir = tempdir().unwrap();
        let path = dir.path().join("deep").join("nested").join("test.jsonl");
        let transcript = Transcript::new(&path);

        transcript.append(&TranscriptEntry::user_message("hello")).unwrap();
        assert!(path.exists());
    }

    #[test]
    fn create_makes_empty_file() {
        let dir = tempdir().unwrap();
        let path = dir.path().join("sessions").join("new.jsonl");
        let transcript = Transcript::new(&path);

        assert!(!path.exists());
        transcript.create().unwrap();
        assert!(path.exists());

        // File should be empty and readable
        let entries = transcript.read_all().unwrap();
        assert!(entries.is_empty());
    }

    #[test]
    fn create_is_idempotent() {
        let dir = tempdir().unwrap();
        let path = dir.path().join("idem.jsonl");
        let transcript = Transcript::new(&path);

        transcript.create().unwrap();
        transcript.append(&TranscriptEntry::user_message("hello")).unwrap();
        transcript.create().unwrap(); // should not truncate

        let entries = transcript.read_all().unwrap();
        assert_eq!(entries.len(), 1);
    }

    // --- SessionMeta ---

    #[test]
    fn session_meta_new_has_uuid_and_timestamps() {
        let meta = SessionMeta::new("grok-3", "/home/user/project");
        assert!(!meta.session_id.is_empty());
        // UUID v4 format: 8-4-4-4-12
        assert_eq!(meta.session_id.len(), 36);
        assert_eq!(meta.model, "grok-3");
        assert_eq!(meta.project_root, "/home/user/project");
        assert_eq!(meta.cumulative_input_tokens, 0);
        assert_eq!(meta.cumulative_output_tokens, 0);
        assert!(meta.summary.is_empty());
        assert_eq!(meta.start_time, meta.last_active);
    }

    #[test]
    fn session_meta_serialization_roundtrip() {
        let meta = SessionMeta::new("grok-3-fast", "/test/project");
        let json = serde_json::to_string(&meta).unwrap();
        let parsed: SessionMeta = serde_json::from_str(&json).unwrap();
        assert_eq!(parsed, meta);
    }

    #[test]
    fn session_meta_save_and_load() {
        let dir = tempdir().unwrap();
        let sessions_dir = dir.path().join("sessions");

        let mut meta = SessionMeta::new("grok-3", "/test");
        meta.cumulative_input_tokens = 1500;
        meta.cumulative_output_tokens = 800;
        meta.summary = "Discussed file structure.".to_string();

        meta.save(&sessions_dir).unwrap();
        let loaded = SessionMeta::load(&sessions_dir, &meta.session_id).unwrap();
        assert_eq!(loaded, meta);
    }

    #[test]
    fn session_meta_atomic_write_no_temp_residue() {
        let dir = tempdir().unwrap();
        let sessions_dir = dir.path().join("sessions");

        let meta = SessionMeta::new("grok-3", "/test");
        meta.save(&sessions_dir).unwrap();

        // Check no temp file lingers
        let tmp_path = SessionMeta::meta_path(&sessions_dir, &meta.session_id)
            .with_extension("meta.json.tmp");
        assert!(!tmp_path.exists());
    }

    #[test]
    fn session_meta_load_missing_returns_error() {
        let dir = tempdir().unwrap();
        let result = SessionMeta::load(dir.path(), "nonexistent-id");
        assert!(result.is_err());
    }

    // --- SessionIndex ---

    #[test]
    fn session_index_list_empty_dir() {
        let dir = tempdir().unwrap();
        let sessions_dir = dir.path().join("sessions");

        let sessions = SessionIndex::list(&sessions_dir).unwrap();
        assert!(sessions.is_empty());
        assert!(sessions_dir.exists(), "should create directory");
    }

    #[test]
    fn session_index_list_sorted_by_last_active() {
        let dir = tempdir().unwrap();
        let sessions_dir = dir.path().join("sessions");

        let mut meta1 = SessionMeta::new("grok-3", "/project");
        meta1.last_active = DateTime::parse_from_rfc3339("2025-01-01T00:00:00Z").unwrap().into();
        meta1.save(&sessions_dir).unwrap();

        let mut meta2 = SessionMeta::new("grok-3", "/project");
        meta2.last_active = DateTime::parse_from_rfc3339("2025-06-01T00:00:00Z").unwrap().into();
        meta2.save(&sessions_dir).unwrap();

        let mut meta3 = SessionMeta::new("grok-3", "/project");
        meta3.last_active = DateTime::parse_from_rfc3339("2025-03-15T00:00:00Z").unwrap().into();
        meta3.save(&sessions_dir).unwrap();

        let sessions = SessionIndex::list(&sessions_dir).unwrap();
        assert_eq!(sessions.len(), 3);
        assert_eq!(sessions[0].session_id, meta2.session_id); // most recent
        assert_eq!(sessions[1].session_id, meta3.session_id);
        assert_eq!(sessions[2].session_id, meta1.session_id); // oldest
    }

    #[test]
    fn session_index_filter_by_project() {
        let dir = tempdir().unwrap();
        let sessions_dir = dir.path().join("sessions");

        let meta1 = SessionMeta::new("grok-3", "/project-a");
        meta1.save(&sessions_dir).unwrap();

        let meta2 = SessionMeta::new("grok-3", "/project-b");
        meta2.save(&sessions_dir).unwrap();

        let meta3 = SessionMeta::new("grok-3", "/project-a");
        meta3.save(&sessions_dir).unwrap();

        let filtered = SessionIndex::list_for_project(&sessions_dir, "/project-a").unwrap();
        assert_eq!(filtered.len(), 2);
        assert!(filtered.iter().all(|s| s.project_root == "/project-a"));
    }

    #[test]
    fn session_index_ignores_corrupt_files() {
        let dir = tempdir().unwrap();
        let sessions_dir = dir.path().join("sessions");
        std::fs::create_dir_all(&sessions_dir).unwrap();

        // Write a valid session
        let meta = SessionMeta::new("grok-3", "/test");
        meta.save(&sessions_dir).unwrap();

        // Write a corrupt .meta.json file
        std::fs::write(sessions_dir.join("corrupt.meta.json"), "not valid json").unwrap();

        let sessions = SessionIndex::list(&sessions_dir).unwrap();
        assert_eq!(sessions.len(), 1);
        assert_eq!(sessions[0].session_id, meta.session_id);
    }

    #[test]
    fn session_index_creates_directory_if_missing() {
        let dir = tempdir().unwrap();
        let sessions_dir = dir.path().join("deep").join("nested").join("sessions");

        let sessions = SessionIndex::list(&sessions_dir).unwrap();
        assert!(sessions.is_empty());
        assert!(sessions_dir.exists());
    }

    #[test]
    fn session_meta_paths_are_correct() {
        let dir = Path::new("/home/user/.grox/sessions");
        let id = "abc-123";

        assert_eq!(
            SessionMeta::meta_path(dir, id),
            PathBuf::from("/home/user/.grox/sessions/abc-123.meta.json")
        );
        assert_eq!(
            SessionMeta::transcript_path(dir, id),
            PathBuf::from("/home/user/.grox/sessions/abc-123.jsonl")
        );
    }

    // --- Session resume (Phase 8) ---

    #[test]
    fn resume_loads_transcript_and_rebuilds_history() {
        let dir = tempdir().unwrap();
        let sessions_dir = dir.path().join("sessions");

        // Create a session with some history
        let meta = SessionMeta::new("grok-3", "/test/project");
        meta.save(&sessions_dir).unwrap();

        let transcript = Transcript::new(SessionMeta::transcript_path(&sessions_dir, &meta.session_id));
        let entries = vec![
            TranscriptEntry::user_message("hello"),
            TranscriptEntry::assistant_message("hi there"),
            TranscriptEntry::tool_call("c1", "file_read", r#"{"path":"main.rs"}"#),
            TranscriptEntry::tool_result("c1", "file_read", "fn main() {}"),
            TranscriptEntry::assistant_message("Here is the file."),
            TranscriptEntry::user_message("thanks"),
            TranscriptEntry::assistant_message("You're welcome!"),
        ];
        for entry in &entries {
            transcript.append(entry).unwrap();
        }

        // Simulate resume: load metadata, read transcript
        let loaded_meta = SessionMeta::load(&sessions_dir, &meta.session_id).unwrap();
        let loaded_transcript = Transcript::new(
            SessionMeta::transcript_path(&sessions_dir, &loaded_meta.session_id),
        );
        let loaded_entries = loaded_transcript.read_all().unwrap();

        assert_eq!(loaded_meta.session_id, meta.session_id);
        assert_eq!(loaded_entries.len(), entries.len());
        assert_eq!(loaded_entries, entries);
    }

    #[test]
    fn resume_new_messages_append_to_existing_transcript() {
        let dir = tempdir().unwrap();
        let sessions_dir = dir.path().join("sessions");

        let meta = SessionMeta::new("grok-3", "/test");
        meta.save(&sessions_dir).unwrap();

        let transcript = Transcript::new(SessionMeta::transcript_path(&sessions_dir, &meta.session_id));
        transcript.append(&TranscriptEntry::user_message("first")).unwrap();
        transcript.append(&TranscriptEntry::assistant_message("response")).unwrap();

        // Simulate resume: read existing entries
        let mut history = transcript.read_all().unwrap();
        assert_eq!(history.len(), 2);

        // New turn appended after resume
        let new_entry = TranscriptEntry::user_message("second");
        transcript.append(&new_entry).unwrap();
        history.push(new_entry);

        // Verify all entries persist
        let final_entries = transcript.read_all().unwrap();
        assert_eq!(final_entries.len(), 3);
        assert_eq!(final_entries, history);
    }

    #[test]
    fn resume_most_recent_for_project() {
        let dir = tempdir().unwrap();
        let sessions_dir = dir.path().join("sessions");

        let mut old = SessionMeta::new("grok-3", "/my/project");
        old.last_active = DateTime::parse_from_rfc3339("2025-01-01T00:00:00Z").unwrap().into();
        old.save(&sessions_dir).unwrap();

        let mut recent = SessionMeta::new("grok-3", "/my/project");
        recent.last_active = DateTime::parse_from_rfc3339("2025-06-01T00:00:00Z").unwrap().into();
        recent.save(&sessions_dir).unwrap();

        // Different project — should not be selected
        let mut other = SessionMeta::new("grok-3", "/other/project");
        other.last_active = DateTime::parse_from_rfc3339("2025-12-01T00:00:00Z").unwrap().into();
        other.save(&sessions_dir).unwrap();

        let project_sessions = SessionIndex::list_for_project(&sessions_dir, "/my/project").unwrap();
        let most_recent = project_sessions.first().unwrap();
        assert_eq!(most_recent.session_id, recent.session_id);
    }

    #[test]
    fn resume_prefix_match_finds_session() {
        let dir = tempdir().unwrap();
        let sessions_dir = dir.path().join("sessions");

        let meta = SessionMeta::new("grok-3", "/test");
        meta.save(&sessions_dir).unwrap();

        let prefix = &meta.session_id[..8];
        let all = SessionIndex::list(&sessions_dir).unwrap();
        let matches: Vec<_> = all.into_iter()
            .filter(|s| s.session_id.starts_with(prefix))
            .collect();
        assert_eq!(matches.len(), 1);
        assert_eq!(matches[0].session_id, meta.session_id);
    }

    #[test]
    fn resume_graceful_error_on_missing_transcript() {
        let dir = tempdir().unwrap();
        let sessions_dir = dir.path().join("sessions");

        // Create metadata but no transcript file
        let meta = SessionMeta::new("grok-3", "/test");
        meta.save(&sessions_dir).unwrap();

        // Reading a missing transcript should return empty, not error
        let transcript = Transcript::new(SessionMeta::transcript_path(&sessions_dir, &meta.session_id));
        let entries = transcript.read_all().unwrap();
        assert!(entries.is_empty());
    }

    #[test]
    fn resume_graceful_error_on_corrupt_transcript() {
        let dir = tempdir().unwrap();
        let sessions_dir = dir.path().join("sessions");

        let meta = SessionMeta::new("grok-3", "/test");
        meta.save(&sessions_dir).unwrap();

        // Write corrupt data to transcript
        let transcript_path = SessionMeta::transcript_path(&sessions_dir, &meta.session_id);
        std::fs::write(&transcript_path, "not valid json at all\nmore garbage\n").unwrap();

        // Should recover gracefully — discard unparseable lines
        let transcript = Transcript::new(&transcript_path);
        let entries = transcript.read_all().unwrap();
        assert!(entries.is_empty());
    }

    #[test]
    fn resume_partial_corrupt_transcript_recovers_valid_entries() {
        let dir = tempdir().unwrap();
        let sessions_dir = dir.path().join("sessions");

        let meta = SessionMeta::new("grok-3", "/test");
        meta.save(&sessions_dir).unwrap();

        let valid_entry = TranscriptEntry::user_message("hello");
        let valid_json = serde_json::to_string(&valid_entry).unwrap();

        // Valid entry followed by corrupt data
        let transcript_path = SessionMeta::transcript_path(&sessions_dir, &meta.session_id);
        std::fs::write(&transcript_path, format!("{valid_json}\ncorrupt data here\n")).unwrap();

        let transcript = Transcript::new(&transcript_path);
        let entries = transcript.read_all().unwrap();
        assert_eq!(entries.len(), 1);
        assert_eq!(entries[0], valid_entry);
    }

    #[test]
    fn session_meta_last_active_updates_on_save() {
        let dir = tempdir().unwrap();
        let sessions_dir = dir.path().join("sessions");

        let mut meta = SessionMeta::new("grok-3", "/test");
        let original_time = meta.last_active;
        meta.save(&sessions_dir).unwrap();

        // Simulate a turn: update last_active and save
        std::thread::sleep(std::time::Duration::from_millis(10));
        meta.last_active = Utc::now();
        meta.save(&sessions_dir).unwrap();

        let loaded = SessionMeta::load(&sessions_dir, &meta.session_id).unwrap();
        assert!(loaded.last_active > original_time);
    }
}
