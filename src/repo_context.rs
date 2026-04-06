use std::path::Path;
use std::process::Command;

const MAX_CHARS: usize = 10_000;

/// Gathered repository context: git state + directory tree.
#[derive(Debug, Clone)]
pub struct RepoContext {
    pub text: String,
    pub truncated: bool,
}

impl RepoContext {
    /// Gather repo context from the given path. Gracefully handles non-git repos.
    pub fn gather(root: &Path) -> RepoContext {
        // Collect sections: git info (optional) + directory tree (always)
        let mut sections: Vec<String> = Vec::with_capacity(4);

        if let Some(branch) = git_branch(root) {
            sections.push(format!("Branch: {branch}"));
        }

        if let Some(status) = git_status(root) {
            if status.is_empty() {
                sections.push("Status: clean".to_string());
            } else {
                sections.push(format!("Status:\n{status}"));
            }
        }

        if let Some(log) = git_log(root)
            && !log.is_empty()
        {
            sections.push(format!("Recent commits:\n{log}"));
        }

        let tree = dir_tree(root, 2);
        if !tree.is_empty() {
            sections.push(format!("Directory tree:\n{tree}"));
        }

        let mut text = sections.join("\n\n");
        let truncation_note = "\n\n... (repo context truncated at 10K characters)";
        let truncated = text.len() > MAX_CHARS;
        if truncated {
            // Reserve space for the truncation note so final size stays within MAX_CHARS
            let budget = MAX_CHARS - truncation_note.len();
            text.truncate(budget);
            // Find last newline to avoid cutting mid-line
            if let Some(pos) = text.rfind('\n') {
                text.truncate(pos);
            }
            text.push_str(truncation_note);
        }

        RepoContext { text, truncated }
    }
}

fn git_branch(root: &Path) -> Option<String> {
    let output = Command::new("git")
        .args(["branch", "--show-current"])
        .current_dir(root)
        .output()
        .ok()?;
    if !output.status.success() {
        return None;
    }
    let branch = String::from_utf8_lossy(&output.stdout).trim().to_string();
    if branch.is_empty() { None } else { Some(branch) }
}

fn git_status(root: &Path) -> Option<String> {
    let output = Command::new("git")
        .args(["status", "--short"])
        .current_dir(root)
        .output()
        .ok()?;
    if !output.status.success() {
        return None;
    }
    Some(String::from_utf8_lossy(&output.stdout).trim().to_string())
}

fn git_log(root: &Path) -> Option<String> {
    let output = Command::new("git")
        .args(["log", "--oneline", "-5"])
        .current_dir(root)
        .output()
        .ok()?;
    if !output.status.success() {
        return None;
    }
    Some(String::from_utf8_lossy(&output.stdout).trim().to_string())
}

/// Build a 2-level directory tree listing.
fn dir_tree(root: &Path, max_depth: usize) -> String {
    let mut lines = Vec::new();
    collect_tree(root, root, 0, max_depth, &mut lines);
    lines.join("\n")
}

fn collect_tree(
    _base: &Path,
    current: &Path,
    depth: usize,
    max_depth: usize,
    lines: &mut Vec<String>,
) {
    if depth > max_depth {
        return;
    }

    let entries = match std::fs::read_dir(current) {
        Ok(rd) => rd,
        Err(_) => return,
    };

    let mut items: Vec<_> = entries
        .filter_map(|e| e.ok())
        .collect();
    items.sort_by_key(|e| e.file_name());

    for entry in items {
        let name = entry.file_name().to_string_lossy().to_string();

        // Skip hidden files/dirs and common noise
        if name.starts_with('.') || name == "node_modules" || name == "target" || name == "__pycache__" {
            continue;
        }

        let indent = "  ".repeat(depth);
        let is_dir = entry.file_type().map(|ft| ft.is_dir()).unwrap_or(false);

        if is_dir {
            lines.push(format!("{indent}{name}/"));
            collect_tree(_base, &entry.path(), depth + 1, max_depth, lines);
        } else {
            lines.push(format!("{indent}{name}"));
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::fs;
    use tempfile::tempdir;

    #[test]
    fn gather_in_non_git_dir() {
        let dir = tempdir().unwrap();
        fs::write(dir.path().join("hello.txt"), "world").unwrap();
        fs::create_dir(dir.path().join("src")).unwrap();
        fs::write(dir.path().join("src/main.rs"), "fn main() {}").unwrap();

        let ctx = RepoContext::gather(dir.path());
        // Should not crash, should have directory tree at minimum
        assert!(ctx.text.contains("hello.txt"));
        assert!(ctx.text.contains("src/"));
        // Should NOT have git info
        assert!(!ctx.text.contains("Branch:"));
    }

    #[test]
    fn gather_includes_directory_tree() {
        let dir = tempdir().unwrap();
        fs::create_dir(dir.path().join("src")).unwrap();
        fs::write(dir.path().join("src/lib.rs"), "").unwrap();
        fs::create_dir(dir.path().join("tests")).unwrap();
        fs::write(dir.path().join("Cargo.toml"), "").unwrap();

        let ctx = RepoContext::gather(dir.path());
        assert!(ctx.text.contains("Cargo.toml"));
        assert!(ctx.text.contains("src/"));
        assert!(ctx.text.contains("lib.rs"));
        assert!(ctx.text.contains("tests/"));
    }

    #[test]
    fn gather_skips_hidden_and_noise_dirs() {
        let dir = tempdir().unwrap();
        fs::create_dir(dir.path().join(".git")).unwrap();
        fs::create_dir(dir.path().join("node_modules")).unwrap();
        fs::create_dir(dir.path().join("target")).unwrap();
        fs::create_dir(dir.path().join("__pycache__")).unwrap();
        fs::write(dir.path().join("real.txt"), "").unwrap();

        let ctx = RepoContext::gather(dir.path());
        assert!(ctx.text.contains("real.txt"));
        assert!(!ctx.text.contains(".git"));
        assert!(!ctx.text.contains("node_modules"));
        assert!(!ctx.text.contains("target"));
        assert!(!ctx.text.contains("__pycache__"));
    }

    #[test]
    fn gather_respects_depth_limit() {
        let dir = tempdir().unwrap();
        // Create 4 levels deep
        fs::create_dir_all(dir.path().join("a/b/c/d")).unwrap();
        fs::write(dir.path().join("a/b/c/d/deep.txt"), "").unwrap();
        fs::write(dir.path().join("a/b/level2.txt"), "").unwrap();

        let ctx = RepoContext::gather(dir.path());
        // depth 0: a/, depth 1: b/, depth 2: c/ and level2.txt — that's the max
        assert!(ctx.text.contains("level2.txt"));
        // d/ is at depth 3, should not appear
        assert!(!ctx.text.contains("deep.txt"));
    }

    #[test]
    fn gather_truncates_at_10k() {
        let dir = tempdir().unwrap();
        // Create lots of files to exceed 10K
        for i in 0..500 {
            fs::write(
                dir.path().join(format!("file_{i:04}_with_a_long_name_to_fill_space.txt")),
                "",
            )
            .unwrap();
        }

        let ctx = RepoContext::gather(dir.path());
        assert!(ctx.truncated);
        assert!(ctx.text.contains("truncated"));
        assert!(ctx.text.len() <= MAX_CHARS, "truncated text ({} bytes) should not exceed MAX_CHARS ({})", ctx.text.len(), MAX_CHARS);
    }

    #[test]
    fn gather_not_truncated_for_small_repo() {
        let dir = tempdir().unwrap();
        fs::write(dir.path().join("README.md"), "hello").unwrap();

        let ctx = RepoContext::gather(dir.path());
        assert!(!ctx.truncated);
        assert!(!ctx.text.contains("truncated"));
    }

    #[test]
    fn gather_in_git_repo() {
        let dir = tempdir().unwrap();

        // Initialize a git repo
        Command::new("git")
            .args(["init"])
            .current_dir(dir.path())
            .output()
            .unwrap();
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
        fs::write(dir.path().join("file.txt"), "content").unwrap();
        Command::new("git")
            .args(["add", "."])
            .current_dir(dir.path())
            .output()
            .unwrap();
        Command::new("git")
            .args(["commit", "-m", "initial"])
            .current_dir(dir.path())
            .output()
            .unwrap();

        let ctx = RepoContext::gather(dir.path());
        // Should have git info
        assert!(ctx.text.contains("Branch:"));
        assert!(ctx.text.contains("Status:"));
        assert!(ctx.text.contains("Recent commits:"));
        assert!(ctx.text.contains("initial"));
    }
}
