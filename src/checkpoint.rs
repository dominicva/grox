use std::path::{Path, PathBuf};
use std::process::Command;

use anyhow::{Context, Result, bail};
use serde::{Deserialize, Serialize};

/// A snapshot of a single file's state before and after a tool execution.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct FileSnapshot {
    /// Absolute path to the file.
    pub path: String,
    /// Git blob hash of the file before modification, or "created" if the file didn't exist.
    pub pre_hash: String,
    /// Git blob hash of the file after modification.
    pub post_hash: String,
}

/// Sentinel value for pre_hash when a file was newly created.
pub const CREATED_SENTINEL: &str = "created";

/// Store a file's content as a git blob and return its hash.
///
/// Runs `git hash-object -w <path>` in the given git repo root.
/// Returns the 40-character hex SHA-1 hash.
pub fn git_hash_object(file_path: &Path, repo_root: &Path) -> Result<String> {
    let output = Command::new("git")
        .args(["hash-object", "-w"])
        .arg(file_path)
        .current_dir(repo_root)
        .output()
        .with_context(|| format!("Failed to run git hash-object for {}", file_path.display()))?;

    if !output.status.success() {
        let stderr = String::from_utf8_lossy(&output.stderr);
        bail!(
            "git hash-object failed for {}: {}",
            file_path.display(),
            stderr.trim()
        );
    }

    Ok(String::from_utf8_lossy(&output.stdout).trim().to_string())
}

/// Retrieve a file's content from a git blob hash.
///
/// Runs `git cat-file blob <hash>` in the given git repo root.
/// Returns the raw bytes of the blob.
pub fn git_cat_file_blob(hash: &str, repo_root: &Path) -> Result<Vec<u8>> {
    let output = Command::new("git")
        .args(["cat-file", "blob", hash])
        .current_dir(repo_root)
        .output()
        .with_context(|| format!("Failed to run git cat-file blob {hash}"))?;

    if !output.status.success() {
        let stderr = String::from_utf8_lossy(&output.stderr);
        bail!("git cat-file blob {hash} failed: {}", stderr.trim());
    }

    Ok(output.stdout)
}

/// Compute a git blob hash for a file without storing it.
///
/// Runs `git hash-object <path>` (without -w) — useful for checking current state
/// without polluting the object store.
pub fn git_hash_object_readonly(file_path: &Path, repo_root: &Path) -> Result<String> {
    let output = Command::new("git")
        .args(["hash-object"])
        .arg(file_path)
        .current_dir(repo_root)
        .output()
        .with_context(|| format!("Failed to run git hash-object for {}", file_path.display()))?;

    if !output.status.success() {
        let stderr = String::from_utf8_lossy(&output.stderr);
        bail!(
            "git hash-object failed for {}: {}",
            file_path.display(),
            stderr.trim()
        );
    }

    Ok(String::from_utf8_lossy(&output.stdout).trim().to_string())
}

/// Check whether a path is inside a git repository.
pub fn is_git_repo(path: &Path) -> bool {
    Command::new("git")
        .args(["rev-parse", "--git-dir"])
        .current_dir(path)
        .output()
        .map(|o| o.status.success())
        .unwrap_or(false)
}

/// Snapshot a file before modification. Returns the pre_hash.
///
/// If the file doesn't exist, returns `CREATED_SENTINEL`.
/// If the file exists, stores it as a git blob and returns the hash.
pub fn snapshot_pre(file_path: &Path, repo_root: &Path) -> Result<String> {
    if !file_path.exists() {
        return Ok(CREATED_SENTINEL.to_string());
    }
    git_hash_object(file_path, repo_root)
}

/// Snapshot a file after modification. Returns the post_hash.
pub fn snapshot_post(file_path: &Path, repo_root: &Path) -> Result<String> {
    git_hash_object(file_path, repo_root)
}

/// Restore a single file from a checkpoint snapshot.
///
/// Returns a `RestoreResult` indicating what happened.
pub fn restore_file(
    snapshot: &FileSnapshot,
    repo_root: &Path,
) -> RestoreResult {
    let file_path = PathBuf::from(&snapshot.path);

    // Check if file was modified since the agent's turn
    if file_path.exists() {
        match git_hash_object_readonly(&file_path, repo_root) {
            Ok(current_hash) => {
                if current_hash != snapshot.post_hash {
                    return RestoreResult::Skipped {
                        path: snapshot.path.clone(),
                        reason: "file was modified after agent edit".to_string(),
                    };
                }
            }
            Err(e) => {
                return RestoreResult::Failed {
                    path: snapshot.path.clone(),
                    reason: format!("failed to hash current file: {e}"),
                };
            }
        }
    } else if snapshot.post_hash != CREATED_SENTINEL {
        // File was deleted by user since agent's turn — skip
        return RestoreResult::Skipped {
            path: snapshot.path.clone(),
            reason: "file no longer exists (deleted after agent edit)".to_string(),
        };
    }

    // Restore the file
    if snapshot.pre_hash == CREATED_SENTINEL {
        // File was created by the agent — delete it
        match std::fs::remove_file(&file_path) {
            Ok(()) => RestoreResult::Restored {
                path: snapshot.path.clone(),
                action: "deleted (was created by agent)".to_string(),
            },
            Err(e) if e.kind() == std::io::ErrorKind::NotFound => {
                // Already gone — fine
                RestoreResult::Restored {
                    path: snapshot.path.clone(),
                    action: "already deleted".to_string(),
                }
            }
            Err(e) => RestoreResult::Failed {
                path: snapshot.path.clone(),
                reason: format!("failed to delete file: {e}"),
            },
        }
    } else {
        // Restore from git blob
        match git_cat_file_blob(&snapshot.pre_hash, repo_root) {
            Ok(content) => {
                // Ensure parent directory exists
                if let Some(parent) = file_path.parent() {
                    if let Err(e) = std::fs::create_dir_all(parent) {
                        return RestoreResult::Failed {
                            path: snapshot.path.clone(),
                            reason: format!("failed to create parent dir: {e}"),
                        };
                    }
                }
                match std::fs::write(&file_path, content) {
                    Ok(()) => RestoreResult::Restored {
                        path: snapshot.path.clone(),
                        action: "restored to previous state".to_string(),
                    },
                    Err(e) => RestoreResult::Failed {
                        path: snapshot.path.clone(),
                        reason: format!("failed to write file: {e}"),
                    },
                }
            }
            Err(e) => RestoreResult::Failed {
                path: snapshot.path.clone(),
                reason: format!("failed to retrieve blob: {e}"),
            },
        }
    }
}

/// Result of attempting to restore a single file.
#[derive(Debug, Clone, PartialEq)]
pub enum RestoreResult {
    Restored { path: String, action: String },
    Skipped { path: String, reason: String },
    Failed { path: String, reason: String },
}

/// Resolve a tool path argument to an absolute path for checkpoint storage.
/// Relative paths are joined to project_root.
pub fn resolve_checkpoint_path(path: &str, project_root: &Path) -> PathBuf {
    let p = Path::new(path);
    if p.is_absolute() {
        p.to_path_buf()
    } else {
        project_root.join(p)
    }
}

/// Extract the file path from a tool's JSON arguments.
///
/// Returns `None` if the arguments don't contain a "path" field.
pub fn extract_tool_path(arguments: &str) -> Option<String> {
    serde_json::from_str::<serde_json::Value>(arguments)
        .ok()
        .and_then(|v| v.get("path")?.as_str().map(String::from))
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::fs;
    use tempfile::tempdir;

    /// Create a temporary git repo for testing.
    fn setup_git_repo() -> tempfile::TempDir {
        let dir = tempdir().unwrap();
        Command::new("git")
            .args(["init"])
            .current_dir(dir.path())
            .output()
            .unwrap();
        // Configure git user for the test repo
        Command::new("git")
            .args(["config", "user.email", "test@test.com"])
            .current_dir(dir.path())
            .output()
            .unwrap();
        Command::new("git")
            .args(["config", "user.name", "Test"])
            .current_dir(dir.path())
            .output()
            .unwrap();
        dir
    }

    // --- git_hash_object ---

    #[test]
    fn hash_object_stores_and_returns_hash() {
        let repo = setup_git_repo();
        let file = repo.path().join("test.txt");
        fs::write(&file, "hello world\n").unwrap();

        let hash = git_hash_object(&file, repo.path()).unwrap();
        assert_eq!(hash.len(), 40, "should be a 40-char hex SHA-1");
        assert!(hash.chars().all(|c| c.is_ascii_hexdigit()));
    }

    #[test]
    fn hash_object_same_content_same_hash() {
        let repo = setup_git_repo();
        let file1 = repo.path().join("a.txt");
        let file2 = repo.path().join("b.txt");
        fs::write(&file1, "identical content").unwrap();
        fs::write(&file2, "identical content").unwrap();

        let hash1 = git_hash_object(&file1, repo.path()).unwrap();
        let hash2 = git_hash_object(&file2, repo.path()).unwrap();
        assert_eq!(hash1, hash2);
    }

    #[test]
    fn hash_object_different_content_different_hash() {
        let repo = setup_git_repo();
        let file1 = repo.path().join("a.txt");
        let file2 = repo.path().join("b.txt");
        fs::write(&file1, "content A").unwrap();
        fs::write(&file2, "content B").unwrap();

        let hash1 = git_hash_object(&file1, repo.path()).unwrap();
        let hash2 = git_hash_object(&file2, repo.path()).unwrap();
        assert_ne!(hash1, hash2);
    }

    #[test]
    fn hash_object_nonexistent_file_fails() {
        let repo = setup_git_repo();
        let result = git_hash_object(&repo.path().join("nope.txt"), repo.path());
        assert!(result.is_err());
    }

    // --- git_cat_file_blob ---

    #[test]
    fn cat_file_blob_roundtrip() {
        let repo = setup_git_repo();
        let file = repo.path().join("test.txt");
        let content = "hello world\n";
        fs::write(&file, content).unwrap();

        let hash = git_hash_object(&file, repo.path()).unwrap();
        let retrieved = git_cat_file_blob(&hash, repo.path()).unwrap();
        assert_eq!(String::from_utf8(retrieved).unwrap(), content);
    }

    #[test]
    fn cat_file_blob_invalid_hash_fails() {
        let repo = setup_git_repo();
        let result = git_cat_file_blob("deadbeef", repo.path());
        assert!(result.is_err());
    }

    // --- git_hash_object_readonly ---

    #[test]
    fn hash_object_readonly_matches_write_version() {
        let repo = setup_git_repo();
        let file = repo.path().join("test.txt");
        fs::write(&file, "test content").unwrap();

        let hash_rw = git_hash_object(&file, repo.path()).unwrap();
        let hash_ro = git_hash_object_readonly(&file, repo.path()).unwrap();
        assert_eq!(hash_rw, hash_ro);
    }

    // --- is_git_repo ---

    #[test]
    fn is_git_repo_true_for_git_dir() {
        let repo = setup_git_repo();
        assert!(is_git_repo(repo.path()));
    }

    #[test]
    fn is_git_repo_false_for_non_git_dir() {
        let dir = tempdir().unwrap();
        assert!(!is_git_repo(dir.path()));
    }

    // --- snapshot_pre ---

    #[test]
    fn snapshot_pre_existing_file() {
        let repo = setup_git_repo();
        let file = repo.path().join("existing.txt");
        fs::write(&file, "original content").unwrap();

        let hash = snapshot_pre(&file, repo.path()).unwrap();
        assert_eq!(hash.len(), 40);
    }

    #[test]
    fn snapshot_pre_nonexistent_file() {
        let repo = setup_git_repo();
        let file = repo.path().join("new.txt");

        let hash = snapshot_pre(&file, repo.path()).unwrap();
        assert_eq!(hash, CREATED_SENTINEL);
    }

    // --- snapshot_post ---

    #[test]
    fn snapshot_post_records_new_state() {
        let repo = setup_git_repo();
        let file = repo.path().join("test.txt");
        fs::write(&file, "original").unwrap();
        let pre = snapshot_pre(&file, repo.path()).unwrap();

        // Modify the file
        fs::write(&file, "modified").unwrap();
        let post = snapshot_post(&file, repo.path()).unwrap();

        assert_ne!(pre, post);
    }

    // --- restore_file ---

    #[test]
    fn restore_file_to_previous_state() {
        let repo = setup_git_repo();
        let file = repo.path().join("test.txt");
        fs::write(&file, "original").unwrap();
        let pre = git_hash_object(&file, repo.path()).unwrap();

        fs::write(&file, "modified").unwrap();
        let post = git_hash_object(&file, repo.path()).unwrap();

        let snapshot = FileSnapshot {
            path: file.display().to_string(),
            pre_hash: pre,
            post_hash: post,
        };

        let result = restore_file(&snapshot, repo.path());
        assert!(matches!(result, RestoreResult::Restored { .. }));
        assert_eq!(fs::read_to_string(&file).unwrap(), "original");
    }

    #[test]
    fn restore_file_skips_when_user_modified() {
        let repo = setup_git_repo();
        let file = repo.path().join("test.txt");
        fs::write(&file, "original").unwrap();
        let pre = git_hash_object(&file, repo.path()).unwrap();

        fs::write(&file, "agent modified").unwrap();
        let post = git_hash_object(&file, repo.path()).unwrap();

        // User modifies the file after the agent
        fs::write(&file, "user modified").unwrap();

        let snapshot = FileSnapshot {
            path: file.display().to_string(),
            pre_hash: pre,
            post_hash: post,
        };

        let result = restore_file(&snapshot, repo.path());
        assert!(matches!(result, RestoreResult::Skipped { .. }));
        // File should be unchanged (user's version preserved)
        assert_eq!(fs::read_to_string(&file).unwrap(), "user modified");
    }

    #[test]
    fn restore_file_deletes_newly_created() {
        let repo = setup_git_repo();
        let file = repo.path().join("new.txt");
        fs::write(&file, "created by agent").unwrap();
        let post = git_hash_object(&file, repo.path()).unwrap();

        let snapshot = FileSnapshot {
            path: file.display().to_string(),
            pre_hash: CREATED_SENTINEL.to_string(),
            post_hash: post,
        };

        let result = restore_file(&snapshot, repo.path());
        assert!(matches!(result, RestoreResult::Restored { .. }));
        assert!(!file.exists());
    }

    #[test]
    fn restore_file_skips_newly_created_if_user_modified() {
        let repo = setup_git_repo();
        let file = repo.path().join("new.txt");
        fs::write(&file, "created by agent").unwrap();
        let post = git_hash_object(&file, repo.path()).unwrap();

        // User modifies the file
        fs::write(&file, "user edited").unwrap();

        let snapshot = FileSnapshot {
            path: file.display().to_string(),
            pre_hash: CREATED_SENTINEL.to_string(),
            post_hash: post,
        };

        let result = restore_file(&snapshot, repo.path());
        assert!(matches!(result, RestoreResult::Skipped { .. }));
        assert!(file.exists());
    }

    // --- extract_tool_path ---

    #[test]
    fn extract_tool_path_from_valid_json() {
        let args = r#"{"path": "src/main.rs", "content": "hello"}"#;
        assert_eq!(extract_tool_path(args), Some("src/main.rs".to_string()));
    }

    #[test]
    fn extract_tool_path_no_path_field() {
        let args = r#"{"command": "ls"}"#;
        assert_eq!(extract_tool_path(args), None);
    }

    #[test]
    fn extract_tool_path_invalid_json() {
        assert_eq!(extract_tool_path("not json"), None);
    }

    // --- resolve_checkpoint_path ---

    #[test]
    fn resolve_relative_path() {
        let root = Path::new("/project");
        let resolved = resolve_checkpoint_path("src/main.rs", root);
        assert_eq!(resolved, PathBuf::from("/project/src/main.rs"));
    }

    #[test]
    fn resolve_absolute_path() {
        let root = Path::new("/project");
        let resolved = resolve_checkpoint_path("/other/file.txt", root);
        assert_eq!(resolved, PathBuf::from("/other/file.txt"));
    }

    // --- FileSnapshot serialization ---

    #[test]
    fn file_snapshot_serde_roundtrip() {
        let snapshot = FileSnapshot {
            path: "/project/src/main.rs".to_string(),
            pre_hash: "abc123".to_string(),
            post_hash: "def456".to_string(),
        };
        let json = serde_json::to_string(&snapshot).unwrap();
        let parsed: FileSnapshot = serde_json::from_str(&json).unwrap();
        assert_eq!(parsed, snapshot);
    }

    #[test]
    fn file_snapshot_created_sentinel_roundtrip() {
        let snapshot = FileSnapshot {
            path: "/project/new.txt".to_string(),
            pre_hash: CREATED_SENTINEL.to_string(),
            post_hash: "abc123".to_string(),
        };
        let json = serde_json::to_string(&snapshot).unwrap();
        let parsed: FileSnapshot = serde_json::from_str(&json).unwrap();
        assert_eq!(parsed.pre_hash, CREATED_SENTINEL);
    }
}
