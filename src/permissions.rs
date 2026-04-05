use std::collections::HashSet;
use std::io::{self, Write};
use std::path::{Path, PathBuf};

use crate::util;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PermissionMode {
    Default,
    Trust,
    ReadOnly,
    Yolo,
}

impl std::fmt::Display for PermissionMode {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Default => write!(f, "default (prompt for writes)"),
            Self::Trust => write!(f, "trust (auto-approve project writes)"),
            Self::ReadOnly => write!(f, "read-only"),
            Self::Yolo => write!(f, "yolo (no guardrails)"),
        }
    }
}

/// The tool category determines what permission rules apply.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ToolCategory {
    Read,
    WriteInProject,
    WriteOutsideProject,
    Shell,
    ShellDestructive,
}

/// Result of a permission check.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum PermissionCheck {
    Allow,
    Deny,
    Prompt { message: String, allow_always: bool },
}

/// Session-scoped permission state (tracks "always" grants per directory).
pub struct SessionPermissions {
    mode: PermissionMode,
    project_root: PathBuf,
    always_granted_dirs: HashSet<PathBuf>,
}

impl SessionPermissions {
    pub fn new(mode: PermissionMode, project_root: PathBuf) -> Self {
        Self {
            mode,
            project_root,
            always_granted_dirs: HashSet::new(),
        }
    }

    pub fn mode(&self) -> PermissionMode {
        self.mode
    }

    pub fn project_root(&self) -> &Path {
        &self.project_root
    }

    /// Classify a tool call into a category based on tool name and arguments.
    pub fn classify_tool(&self, tool_name: &str, arguments: &str) -> ToolCategory {
        match tool_name {
            "file_read" | "list_files" | "grep" => ToolCategory::Read,
            "file_write" => {
                let path = extract_path(arguments);
                if self.is_inside_project(&path) {
                    ToolCategory::WriteInProject
                } else {
                    ToolCategory::WriteOutsideProject
                }
            }
            "shell_exec" => {
                let cmd = extract_command(arguments);
                if is_destructive_command(&cmd) {
                    ToolCategory::ShellDestructive
                } else {
                    ToolCategory::Shell
                }
            }
            _ => ToolCategory::Shell,
        }
    }

    /// Check whether a tool call is allowed, denied, or requires a prompt.
    pub fn check(&self, tool_name: &str, arguments: &str) -> PermissionCheck {
        let category = self.classify_tool(tool_name, arguments);
        self.check_category(category, arguments)
    }

    fn check_category(&self, category: ToolCategory, arguments: &str) -> PermissionCheck {
        // 1. Yolo: allow everything
        if self.mode == PermissionMode::Yolo {
            return PermissionCheck::Allow;
        }

        // 2. Reads: always allowed
        if category == ToolCategory::Read {
            return PermissionCheck::Allow;
        }

        // 3. ReadOnly: deny all writes and shell
        if self.mode == PermissionMode::ReadOnly {
            return PermissionCheck::Deny;
        }

        // 4. Destructive shell: always prompt, no "always" option
        if category == ToolCategory::ShellDestructive {
            let cmd = extract_command(arguments);
            return PermissionCheck::Prompt {
                message: format!("Run destructive command: {cmd}"),
                allow_always: false,
            };
        }

        // 5. Check "always" grants for the target directory
        let target_dir = self.target_dir(arguments);
        if let Some(dir) = &target_dir {
            if self.always_granted_dirs.contains(dir) {
                return PermissionCheck::Allow;
            }
        }

        // 6. Trust mode: auto-approve writes inside project
        if self.mode == PermissionMode::Trust && category == ToolCategory::WriteInProject {
            return PermissionCheck::Allow;
        }

        // 7. Default: prompt
        let message = match category {
            ToolCategory::WriteInProject => {
                let path = extract_path(arguments);
                format!("Write to: {path}")
            }
            ToolCategory::WriteOutsideProject => {
                let path = extract_path(arguments);
                let full = std::path::Path::new(&path);
                format!("Write outside project: {}", full.display())
            }
            ToolCategory::Shell => {
                let cmd = extract_command(arguments);
                format!("Run: {cmd}")
            }
            _ => "Approve?".to_string(),
        };

        PermissionCheck::Prompt {
            message,
            allow_always: true,
        }
    }

    /// Prompt the user interactively and return whether the action is allowed.
    /// If the user chooses "always", records the grant for the target directory.
    pub fn prompt_user(&mut self, check: &PermissionCheck) -> bool {
        let (message, allow_always) = match check {
            PermissionCheck::Prompt { message, allow_always } => (message.as_str(), *allow_always),
            PermissionCheck::Allow => return true,
            PermissionCheck::Deny => return false,
        };

        let options = if allow_always { "[y/n/always]" } else { "[y/n]" };
        eprint!("  {} {} {} ", "?".yellow(), message, options.dimmed());
        let _ = io::stderr().flush();

        let mut answer = String::new();
        if io::stdin().read_line(&mut answer).is_err() {
            return false;
        }
        let answer = answer.trim().to_lowercase();

        match answer.as_str() {
            "y" | "yes" => true,
            "always" if allow_always => {
                // We don't have the target dir here — callers should use
                // prompt_for_tool which handles the grant.
                true
            }
            _ => false,
        }
    }

    /// Full permission flow for a tool call: check, prompt if needed, record grants.
    /// Returns true if allowed.
    pub fn authorize(&mut self, tool_name: &str, arguments: &str) -> bool {
        let check = self.check(tool_name, arguments);
        match &check {
            PermissionCheck::Allow => true,
            PermissionCheck::Deny => false,
            PermissionCheck::Prompt { message, allow_always } => {
                let options = if *allow_always { "[y/n/always]" } else { "[y/n]" };
                eprint!("  {} {} {} ", "?".yellow(), message, options.dimmed());
                let _ = io::stderr().flush();

                let mut answer = String::new();
                if io::stdin().read_line(&mut answer).is_err() {
                    return false;
                }
                let answer = answer.trim().to_lowercase();

                match answer.as_str() {
                    "y" | "yes" => true,
                    "always" if *allow_always => {
                        if let Some(dir) = self.target_dir(arguments) {
                            self.always_granted_dirs.insert(dir);
                        }
                        true
                    }
                    _ => false,
                }
            }
        }
    }

    /// Record an "always" grant for a directory.
    pub fn grant_always(&mut self, dir: PathBuf) {
        self.always_granted_dirs.insert(dir);
    }

    fn is_inside_project(&self, path: &str) -> bool {
        let p = Path::new(path);
        // Try to validate — if it resolves inside project root, it's inside
        util::validate_path(p, &self.project_root).is_ok()
    }

    fn target_dir(&self, arguments: &str) -> Option<PathBuf> {
        // For writes: parent directory of the target path
        // For shell: the cwd (defaults to project root)
        let parsed: serde_json::Value = serde_json::from_str(arguments).ok()?;
        if let Some(path) = parsed.get("path").and_then(|v| v.as_str()) {
            let p = Path::new(path);
            return p.parent().map(PathBuf::from);
        }
        if let Some(cwd) = parsed.get("cwd").and_then(|v| v.as_str()) {
            return Some(PathBuf::from(cwd));
        }
        Some(self.project_root.clone())
    }
}

fn extract_path(arguments: &str) -> String {
    serde_json::from_str::<serde_json::Value>(arguments)
        .ok()
        .and_then(|v| v.get("path").and_then(|p| p.as_str()).map(String::from))
        .unwrap_or_else(|| "?".to_string())
}

fn extract_command(arguments: &str) -> String {
    serde_json::from_str::<serde_json::Value>(arguments)
        .ok()
        .and_then(|v| v.get("command").and_then(|c| c.as_str()).map(String::from))
        .unwrap_or_else(|| "?".to_string())
}

/// Best-effort destructive command detection.
fn is_destructive_command(cmd: &str) -> bool {
    use regex::Regex;
    // Lazy-init would be better but this is called infrequently
    let patterns = [
        r"\brm\s+(-\w*[rR]\w*\s+|--recursive)",      // rm -r, rm -rf, rm --recursive
        r"\bgit\s+push\s+.*--force",                   // git push --force
        r"\bgit\s+push\s+-f\b",                        // git push -f
        r"\bgit\s+reset\s+--hard",                     // git reset --hard
        r"\bgit\s+clean\s+-\w*f",                      // git clean -f, -fd
        r"(?i)\bdrop\s+(table|database|schema)\b",     // DROP TABLE/DATABASE
        r"(?i)\btruncate\s+table\b",                   // TRUNCATE TABLE
        r"(?i)\bdelete\s+from\b",                      // DELETE FROM
        r"\bchmod\s+(-\w+\s+)*0?777\b",               // chmod 777
        r"\bmkfs\b",                                   // mkfs
        r"\bdd\s+.*\bof=/dev/",                        // dd of=/dev/...
        r">\s*/dev/sd[a-z]",                           // redirect to block device
    ];

    for pattern in &patterns {
        if let Ok(re) = Regex::new(pattern) {
            if re.is_match(cmd) {
                return true;
            }
        }
    }
    false
}

// Need colored for the prompt
use colored::Colorize;

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::tempdir;

    fn perms(mode: PermissionMode) -> SessionPermissions {
        let dir = tempdir().unwrap();
        SessionPermissions::new(mode, dir.path().to_path_buf())
    }

    fn perms_with_root(mode: PermissionMode, root: PathBuf) -> SessionPermissions {
        SessionPermissions::new(mode, root)
    }

    // --- Permission matrix tests ---

    #[test]
    fn yolo_allows_everything() {
        let p = perms(PermissionMode::Yolo);
        assert_eq!(p.check("file_read", r#"{"path":"x"}"#), PermissionCheck::Allow);
        assert_eq!(p.check("file_write", r#"{"path":"x"}"#), PermissionCheck::Allow);
        assert_eq!(p.check("shell_exec", r#"{"command":"rm -rf /"}"#), PermissionCheck::Allow);
    }

    #[test]
    fn readonly_allows_reads_denies_writes() {
        let p = perms(PermissionMode::ReadOnly);
        assert_eq!(p.check("file_read", r#"{"path":"x"}"#), PermissionCheck::Allow);
        assert_eq!(p.check("list_files", r#"{"path":"."}"#), PermissionCheck::Allow);
        assert_eq!(p.check("grep", r#"{"pattern":"foo"}"#), PermissionCheck::Allow);
        assert_eq!(p.check("file_write", r#"{"path":"x"}"#), PermissionCheck::Deny);
        assert_eq!(p.check("shell_exec", r#"{"command":"ls"}"#), PermissionCheck::Deny);
    }

    #[test]
    fn trust_auto_approves_in_project_writes() {
        let dir = tempdir().unwrap();
        let file_path = dir.path().join("test.txt");
        let args = format!(r#"{{"path":"{}"}}"#, file_path.display());
        let p = perms_with_root(PermissionMode::Trust, dir.path().to_path_buf());
        assert_eq!(p.check("file_write", &args), PermissionCheck::Allow);
    }

    #[test]
    fn trust_prompts_for_outside_project_writes() {
        let dir = tempdir().unwrap();
        let other = tempdir().unwrap();
        let file_path = other.path().join("escape.txt");
        let args = format!(r#"{{"path":"{}"}}"#, file_path.display());
        let p = perms_with_root(PermissionMode::Trust, dir.path().to_path_buf());
        let check = p.check("file_write", &args);
        assert!(matches!(check, PermissionCheck::Prompt { .. }));
    }

    #[test]
    fn default_prompts_for_writes() {
        let dir = tempdir().unwrap();
        let file_path = dir.path().join("test.txt");
        let args = format!(r#"{{"path":"{}"}}"#, file_path.display());
        let p = perms_with_root(PermissionMode::Default, dir.path().to_path_buf());
        let check = p.check("file_write", &args);
        assert!(matches!(check, PermissionCheck::Prompt { allow_always: true, .. }));
    }

    #[test]
    fn default_prompts_for_shell() {
        let p = perms(PermissionMode::Default);
        let check = p.check("shell_exec", r#"{"command":"ls -la"}"#);
        assert!(matches!(check, PermissionCheck::Prompt { allow_always: true, .. }));
    }

    #[test]
    fn destructive_command_no_always() {
        let p = perms(PermissionMode::Default);
        let check = p.check("shell_exec", r#"{"command":"rm -rf /tmp/stuff"}"#);
        assert!(matches!(check, PermissionCheck::Prompt { allow_always: false, .. }));
    }

    #[test]
    fn trust_prompts_for_destructive() {
        let p = perms(PermissionMode::Trust);
        let check = p.check("shell_exec", r#"{"command":"git push --force"}"#);
        assert!(matches!(check, PermissionCheck::Prompt { allow_always: false, .. }));
    }

    // --- "Always" grants ---

    #[test]
    fn always_grant_persists() {
        let dir = tempdir().unwrap();
        let mut p = perms_with_root(PermissionMode::Default, dir.path().to_path_buf());
        let file_path = dir.path().join("test.txt");
        let args = format!(r#"{{"path":"{}"}}"#, file_path.display());

        // Before grant: prompts
        assert!(matches!(p.check("file_write", &args), PermissionCheck::Prompt { .. }));

        // Grant "always" for the directory
        p.grant_always(dir.path().to_path_buf());

        // After grant: allowed
        assert_eq!(p.check("file_write", &args), PermissionCheck::Allow);
    }

    #[test]
    fn always_grant_is_per_directory() {
        let dir = tempdir().unwrap();
        let sub = dir.path().join("subdir");
        std::fs::create_dir(&sub).unwrap();

        let mut p = perms_with_root(PermissionMode::Default, dir.path().to_path_buf());

        // Grant for root dir only
        p.grant_always(dir.path().to_path_buf());

        // File in root dir: allowed (parent is the root)
        let args_root = format!(r#"{{"path":"{}"}}"#, dir.path().join("file.txt").display());
        assert_eq!(p.check("file_write", &args_root), PermissionCheck::Allow);

        // File in subdir: still prompts (different dir)
        let args_sub = format!(r#"{{"path":"{}"}}"#, sub.join("file.txt").display());
        assert!(matches!(p.check("file_write", &args_sub), PermissionCheck::Prompt { .. }));
    }

    // --- Destructive pattern detection ---

    #[test]
    fn destructive_patterns_match() {
        assert!(is_destructive_command("rm -rf /tmp"));
        assert!(is_destructive_command("rm -r /tmp"));
        assert!(is_destructive_command("rm --recursive /tmp"));
        assert!(is_destructive_command("git push --force origin main"));
        assert!(is_destructive_command("git push -f"));
        assert!(is_destructive_command("git reset --hard HEAD~1"));
        assert!(is_destructive_command("git clean -fd"));
        assert!(is_destructive_command("DROP TABLE users"));
        assert!(is_destructive_command("drop database mydb"));
        assert!(is_destructive_command("TRUNCATE TABLE logs"));
        assert!(is_destructive_command("DELETE FROM users"));
    }

    #[test]
    fn safe_commands_dont_match() {
        assert!(!is_destructive_command("git push origin main"));
        assert!(!is_destructive_command("rm file.txt"));
        assert!(!is_destructive_command("ls -la"));
        assert!(!is_destructive_command("cargo test"));
        assert!(!is_destructive_command("git status"));
        assert!(!is_destructive_command("echo hello"));
    }
}
