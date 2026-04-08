use std::path::{Path, PathBuf};
use std::sync::{Arc, Mutex};

/// An index of project files for path completion.
///
/// Built using the `ignore` crate to walk git-tracked and untracked files
/// while respecting `.gitignore`. The index stores paths relative to the
/// project root.
#[derive(Clone)]
pub struct FileIndex {
    inner: Arc<Mutex<IndexState>>,
    project_root: PathBuf,
}

struct IndexState {
    entries: Vec<IndexEntry>,
}

#[derive(Clone, Debug)]
struct IndexEntry {
    /// Path relative to project root (e.g. "src/main.rs", "docs/")
    rel_path: String,
    is_dir: bool,
}

impl FileIndex {
    /// Build a new file index by walking the project root.
    pub fn build(project_root: &Path) -> Self {
        let entries = walk_project(project_root);
        Self {
            inner: Arc::new(Mutex::new(IndexState { entries })),
            project_root: project_root.to_path_buf(),
        }
    }

    /// Rebuild the index (called after mutating tool calls).
    pub fn refresh(&self) {
        let entries = walk_project(&self.project_root);
        let mut state = self.inner.lock().unwrap();
        state.entries = entries;
    }

    /// Return completion candidates matching the given input token.
    ///
    /// The `token` is the path-like text the user has typed (with any trigger
    /// prefix like `@` already stripped). Matches by prefix or substring on
    /// the relative path.
    ///
    /// Returns `(display, replacement)` pairs. Directories get a `/` suffix;
    /// files get a trailing space.
    pub fn completions(&self, token: &str) -> Vec<(String, String)> {
        let state = self.inner.lock().unwrap();
        let query = token.strip_prefix("./").unwrap_or(token);

        let mut results: Vec<(String, String)> = state
            .entries
            .iter()
            .filter(|e| {
                e.rel_path.starts_with(query)
                    || e.rel_path
                        .rsplit('/')
                        .next()
                        .is_some_and(|basename| basename.starts_with(query))
            })
            .take(50)
            .map(|e| {
                let suffix = if e.is_dir { "/" } else { " " };
                let display = format!("{}{suffix}", e.rel_path);
                let replacement = format!("{}{suffix}", e.rel_path);
                (display, replacement)
            })
            .collect();

        results.sort_by(|a, b| a.0.cmp(&b.0));
        results
    }
}

/// Walk the project tree using the `ignore` crate, collecting relative paths.
fn walk_project(project_root: &Path) -> Vec<IndexEntry> {
    let mut entries = Vec::new();

    let walker = ignore::WalkBuilder::new(project_root)
        .hidden(false)
        .git_ignore(true)
        .git_global(true)
        .build();

    for result in walker {
        let Ok(dir_entry) = result else { continue };
        let path = dir_entry.path();

        // Skip the root itself
        if path == project_root {
            continue;
        }

        let Ok(rel) = path.strip_prefix(project_root) else {
            continue;
        };
        let rel_str = rel.to_string_lossy().to_string();
        if rel_str.is_empty() {
            continue;
        }

        // Skip .git directory contents
        if rel_str == ".git" || rel_str.starts_with(".git/") {
            continue;
        }

        let is_dir = path.is_dir();
        entries.push(IndexEntry {
            rel_path: rel_str,
            is_dir,
        });
    }

    entries.sort_by(|a, b| a.rel_path.cmp(&b.rel_path));
    entries
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::fs;

    #[test]
    fn build_index_from_tempdir() {
        let dir = tempfile::tempdir().unwrap();
        fs::write(dir.path().join("main.rs"), "fn main() {}").unwrap();
        fs::create_dir(dir.path().join("src")).unwrap();
        fs::write(dir.path().join("src/lib.rs"), "").unwrap();

        let idx = FileIndex::build(dir.path());
        let completions = idx.completions("");
        let paths: Vec<&str> = completions.iter().map(|(d, _)| d.as_str()).collect();
        assert!(paths.contains(&"main.rs "));
        assert!(paths.contains(&"src/"));
        assert!(paths.contains(&"src/lib.rs "));
    }

    #[test]
    fn gitignore_respected() {
        let dir = tempfile::tempdir().unwrap();

        // Initialize a git repo so .gitignore is respected
        std::process::Command::new("git")
            .args(["init"])
            .current_dir(dir.path())
            .output()
            .unwrap();

        fs::write(dir.path().join(".gitignore"), "target/\n*.log\n").unwrap();
        fs::create_dir(dir.path().join("target")).unwrap();
        fs::write(dir.path().join("target/debug"), "").unwrap();
        fs::write(dir.path().join("app.log"), "log data").unwrap();
        fs::write(dir.path().join("main.rs"), "fn main() {}").unwrap();

        let idx = FileIndex::build(dir.path());
        let completions = idx.completions("");
        let paths: Vec<&str> = completions.iter().map(|(d, _)| d.as_str()).collect();
        assert!(paths.contains(&"main.rs "));
        assert!(!paths.iter().any(|p| p.contains("target")));
        assert!(!paths.iter().any(|p| p.contains("app.log")));
    }

    #[test]
    fn prefix_matching() {
        let dir = tempfile::tempdir().unwrap();
        fs::write(dir.path().join("main.rs"), "").unwrap();
        fs::write(dir.path().join("model.rs"), "").unwrap();
        fs::write(dir.path().join("readme.md"), "").unwrap();

        let idx = FileIndex::build(dir.path());
        let completions = idx.completions("m");
        let paths: Vec<&str> = completions.iter().map(|(d, _)| d.as_str()).collect();
        assert!(paths.contains(&"main.rs "));
        assert!(paths.contains(&"model.rs "));
        assert!(!paths.contains(&"readme.md "));
    }

    #[test]
    fn basename_matching() {
        let dir = tempfile::tempdir().unwrap();
        fs::create_dir(dir.path().join("src")).unwrap();
        fs::write(dir.path().join("src/main.rs"), "").unwrap();

        let idx = FileIndex::build(dir.path());
        let completions = idx.completions("main");
        let paths: Vec<&str> = completions.iter().map(|(d, _)| d.as_str()).collect();
        assert!(paths.contains(&"src/main.rs "));
    }

    #[test]
    fn directory_suffix() {
        let dir = tempfile::tempdir().unwrap();
        fs::create_dir(dir.path().join("src")).unwrap();
        fs::write(dir.path().join("src/lib.rs"), "").unwrap();

        let idx = FileIndex::build(dir.path());
        let completions = idx.completions("src");
        let paths: Vec<&str> = completions.iter().map(|(d, _)| d.as_str()).collect();
        assert!(paths.contains(&"src/"));
    }

    #[test]
    fn replacement_includes_suffix() {
        let dir = tempfile::tempdir().unwrap();
        fs::create_dir(dir.path().join("src")).unwrap();
        fs::write(dir.path().join("src/lib.rs"), "").unwrap();
        fs::write(dir.path().join("main.rs"), "").unwrap();

        let idx = FileIndex::build(dir.path());
        let completions = idx.completions("");
        // Directory replacement ends with /
        let dir_entry = completions.iter().find(|(d, _)| d == "src/").unwrap();
        assert_eq!(dir_entry.1, "src/");
        // File replacement ends with space
        let file_entry = completions.iter().find(|(d, _)| d == "main.rs ").unwrap();
        assert_eq!(file_entry.1, "main.rs ");
    }

    #[test]
    fn refresh_picks_up_new_files() {
        let dir = tempfile::tempdir().unwrap();
        fs::write(dir.path().join("a.rs"), "").unwrap();

        let idx = FileIndex::build(dir.path());
        assert!(idx.completions("b").is_empty());

        fs::write(dir.path().join("b.rs"), "").unwrap();
        idx.refresh();

        let completions = idx.completions("b");
        let paths: Vec<&str> = completions.iter().map(|(d, _)| d.as_str()).collect();
        assert!(paths.contains(&"b.rs "));
    }

    #[test]
    fn dot_slash_prefix_stripped() {
        let dir = tempfile::tempdir().unwrap();
        fs::write(dir.path().join("main.rs"), "").unwrap();

        let idx = FileIndex::build(dir.path());
        let completions = idx.completions("./main");
        let paths: Vec<&str> = completions.iter().map(|(d, _)| d.as_str()).collect();
        assert!(paths.contains(&"main.rs "));
    }

    #[test]
    fn empty_query_returns_all() {
        let dir = tempfile::tempdir().unwrap();
        fs::write(dir.path().join("a.rs"), "").unwrap();
        fs::write(dir.path().join("b.rs"), "").unwrap();

        let idx = FileIndex::build(dir.path());
        let completions = idx.completions("");
        assert!(completions.len() >= 2);
    }

    #[test]
    fn results_capped_at_50() {
        let dir = tempfile::tempdir().unwrap();
        for i in 0..60 {
            fs::write(dir.path().join(format!("file_{i:03}.rs")), "").unwrap();
        }

        let idx = FileIndex::build(dir.path());
        let completions = idx.completions("");
        assert!(completions.len() <= 50);
    }
}
