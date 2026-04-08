use std::path::{Path, PathBuf};

use anyhow::{Result, bail};
use serde_json::{Value, json};
use tokio::process::Command;
use tokio::time::{Duration, timeout};

use crate::util;

/// Resolve a tool path argument against the project root.
/// Relative paths are joined to project_root; absolute paths are used as-is.
fn resolve_tool_path(path: &str, project_root: &Path) -> PathBuf {
    let p = Path::new(path);
    if p.is_absolute() {
        p.to_path_buf()
    } else {
        project_root.join(p)
    }
}

/// Count occurrences of `needle` in `haystack`, including overlapping matches.
/// For example, "aa" in "aaa" returns 2 (positions 0 and 1).
/// Advances by one character (not one byte) to safely handle multibyte UTF-8.
fn count_occurrences(haystack: &str, needle: &str) -> usize {
    if needle.is_empty() {
        return 0;
    }
    let mut count = 0;
    let mut start = 0;
    while start + needle.len() <= haystack.len() {
        if let Some(pos) = haystack[start..].find(needle) {
            count += 1;
            // Advance past the start of this match by one character, not one byte,
            // to avoid slicing inside a multibyte character.
            let match_start = start + pos;
            let next = match_start
                + haystack[match_start..]
                    .chars()
                    .next()
                    .map_or(1, |c| c.len_utf8());
            start = next;
        } else {
            break;
        }
    }
    count
}

/// Typed result from tool execution, replacing bare strings.
/// `success` indicates whether the tool executed without error.
#[derive(Debug, Clone)]
pub struct ToolOutcome {
    /// Whether the tool executed successfully (file was written, command ran, etc.).
    /// False for errors, permission denials, and validation failures.
    /// Used by tests and future phases; checkpoint emission uses hash comparison.
    #[allow(dead_code)]
    pub success: bool,
    /// Human-readable output to send back to the model.
    pub output: String,
}

#[derive(Debug, Clone)]
pub enum Tool {
    FileRead,
    FileWrite,
    FileEdit,
    ListFiles,
    ShellExec,
    Grep,
}

#[derive(Debug, Clone, PartialEq)]
pub struct ToolCall {
    pub call_id: String,
    pub name: String,
    pub arguments: String,
}

impl Tool {
    pub fn all() -> Vec<Tool> {
        vec![
            Tool::FileRead,
            Tool::FileWrite,
            Tool::FileEdit,
            Tool::ListFiles,
            Tool::ShellExec,
            Tool::Grep,
        ]
    }

    pub fn definitions() -> Vec<Value> {
        Tool::all().iter().map(|t| t.definition()).collect()
    }

    pub fn from_name(name: &str) -> Option<Tool> {
        match name {
            "file_read" => Some(Tool::FileRead),
            "file_write" => Some(Tool::FileWrite),
            "file_edit" => Some(Tool::FileEdit),
            "list_files" => Some(Tool::ListFiles),
            "shell_exec" => Some(Tool::ShellExec),
            "grep" => Some(Tool::Grep),
            _ => None,
        }
    }

    pub fn definition(&self) -> Value {
        match self {
            Tool::FileRead => json!({
                "type": "function",
                "name": "file_read",
                "description": "Read the contents of a file at the given path. Returns the file contents as a string.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "path": {
                            "type": "string",
                            "description": "The absolute or relative path to the file to read"
                        }
                    },
                    "required": ["path"],
                    "additionalProperties": false
                }
            }),
            Tool::FileWrite => json!({
                "type": "function",
                "name": "file_write",
                "description": "Write content to a file. Creates the file if it doesn't exist, or overwrites if it does. Parent directories are created automatically.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "path": {
                            "type": "string",
                            "description": "The path to the file to write"
                        },
                        "content": {
                            "type": "string",
                            "description": "The content to write to the file"
                        }
                    },
                    "required": ["path", "content"],
                    "additionalProperties": false
                }
            }),
            Tool::FileEdit => json!({
                "type": "function",
                "name": "file_edit",
                "description": "Edit a file by replacing occurrences of a string. By default, old_string must match exactly one location (use enough context to ensure uniqueness). Set replace_all to true to replace every occurrence.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "path": {
                            "type": "string",
                            "description": "The path to the file to edit"
                        },
                        "old_string": {
                            "type": "string",
                            "description": "The exact string to find and replace"
                        },
                        "new_string": {
                            "type": "string",
                            "description": "The replacement string"
                        },
                        "replace_all": {
                            "type": "boolean",
                            "description": "Replace all occurrences (default: false, which requires exactly one match)"
                        }
                    },
                    "required": ["path", "old_string", "new_string"],
                    "additionalProperties": false
                }
            }),
            Tool::ListFiles => json!({
                "type": "function",
                "name": "list_files",
                "description": "List the contents of a directory. Returns file and directory names, one per line. Use this to explore project structure and discover files before reading them.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "path": {
                            "type": "string",
                            "description": "The directory path to list (default: current directory)"
                        }
                    },
                    "required": ["path"],
                    "additionalProperties": false
                }
            }),
            Tool::Grep => json!({
                "type": "function",
                "name": "grep",
                "description": "Search file contents using ripgrep. Returns matching lines with file paths and line numbers. Use this to find code patterns, function definitions, imports, and references across the codebase.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "pattern": {
                            "type": "string",
                            "description": "Regex pattern to search for"
                        },
                        "path": {
                            "type": "string",
                            "description": "Directory or file to search in (defaults to project root)"
                        },
                        "glob": {
                            "type": "string",
                            "description": "Glob pattern to filter files (e.g. '*.rs', '*.ts')"
                        },
                        "case_insensitive": {
                            "type": "boolean",
                            "description": "Case-insensitive search (default: false)"
                        },
                        "max_results": {
                            "type": "integer",
                            "description": "Maximum number of matching lines to return (default: 100)"
                        }
                    },
                    "required": ["pattern"],
                    "additionalProperties": false
                }
            }),
            Tool::ShellExec => json!({
                "type": "function",
                "name": "shell_exec",
                "description": "Execute a shell command and return its output (stdout + stderr). Commands run in the project root by default. Use this for build commands, git operations, running tests, and other shell tasks.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "command": {
                            "type": "string",
                            "description": "The shell command to execute"
                        },
                        "cwd": {
                            "type": "string",
                            "description": "Working directory for the command (defaults to project root)"
                        },
                        "timeout_secs": {
                            "type": "integer",
                            "description": "Timeout in seconds (default: 60, max: 300)"
                        }
                    },
                    "required": ["command"],
                    "additionalProperties": false
                }
            }),
        }
    }

    pub async fn execute(&self, arguments: &str, project_root: &std::path::Path) -> ToolOutcome {
        let result = match self {
            Tool::FileRead => execute_file_read(arguments, project_root),
            Tool::FileWrite => execute_file_write(arguments, project_root),
            Tool::FileEdit => execute_file_edit(arguments, project_root),
            Tool::ListFiles => execute_list_files(arguments, project_root),
            Tool::ShellExec => execute_shell_exec(arguments, project_root).await,
            Tool::Grep => execute_grep(arguments, project_root).await,
        };
        match result {
            Ok(output) => ToolOutcome {
                success: true,
                output,
            },
            Err(e) => ToolOutcome {
                success: false,
                output: format!("Error: {e}"),
            },
        }
    }
}

fn execute_file_read(arguments: &str, project_root: &std::path::Path) -> Result<String> {
    let args: Value = serde_json::from_str(arguments)?;
    let path = args["path"]
        .as_str()
        .ok_or_else(|| anyhow::anyhow!("Missing required parameter: path"))?;

    let resolved = resolve_tool_path(path, project_root);

    // Read as bytes first for binary detection
    let bytes = std::fs::read(&resolved)
        .map_err(|e| anyhow::anyhow!("Failed to read file '{}': {}", path, e))?;

    if util::is_binary(&bytes) {
        anyhow::bail!(
            "File '{}' appears to be binary — cannot display contents",
            path
        );
    }

    let content = String::from_utf8(bytes)
        .map_err(|_| anyhow::anyhow!("File '{}' contains invalid UTF-8", path))?;

    Ok(util::clip_for_model(&content))
}

fn execute_file_write(arguments: &str, project_root: &std::path::Path) -> Result<String> {
    let args: Value = serde_json::from_str(arguments)?;
    let path = args["path"]
        .as_str()
        .ok_or_else(|| anyhow::anyhow!("Missing required parameter: path"))?;
    let content = args["content"]
        .as_str()
        .ok_or_else(|| anyhow::anyhow!("Missing required parameter: content"))?;

    let target = resolve_tool_path(path, project_root);

    // For nested paths, create parent directories first so validate_path can resolve them.
    // We create within the project root to ensure safety before canonicalization.
    if let Some(parent) = target.parent()
        && !parent.exists()
    {
        // Ensure the parent chain stays within project root by checking the
        // closest existing ancestor resolves inside the root
        let mut ancestor = parent.to_path_buf();
        while !ancestor.exists() {
            if !ancestor.pop() {
                break;
            }
        }
        if ancestor.exists() {
            let canonical_ancestor = ancestor.canonicalize()?;
            let canonical_root = project_root.canonicalize()?;
            if !canonical_ancestor.starts_with(&canonical_root) {
                anyhow::bail!(
                    "Path '{}' is outside the project root '{}'",
                    target.display(),
                    project_root.display()
                );
            }
        }
        std::fs::create_dir_all(parent).map_err(|e| {
            anyhow::anyhow!("Failed to create directory '{}': {}", parent.display(), e)
        })?;
    }

    // Validate the final path is within project root (symlink-safe)
    let resolved = util::validate_path(&target, project_root)?;

    std::fs::write(&resolved, content)
        .map_err(|e| anyhow::anyhow!("Failed to write file '{}': {}", path, e))?;

    Ok(format!("Wrote {} bytes to {}", content.len(), path))
}

fn execute_file_edit(arguments: &str, project_root: &std::path::Path) -> Result<String> {
    let args: Value = serde_json::from_str(arguments)?;
    let path = args["path"]
        .as_str()
        .ok_or_else(|| anyhow::anyhow!("Missing required parameter: path"))?;
    let old_string = args["old_string"]
        .as_str()
        .ok_or_else(|| anyhow::anyhow!("Missing required parameter: old_string"))?;
    let new_string = args["new_string"]
        .as_str()
        .ok_or_else(|| anyhow::anyhow!("Missing required parameter: new_string"))?;

    let replace_all = args["replace_all"].as_bool().unwrap_or(false);

    if old_string.is_empty() {
        bail!("old_string must not be empty");
    }

    if old_string == new_string {
        bail!("old_string and new_string are identical — nothing to change");
    }

    let target = resolve_tool_path(path, project_root);
    let resolved = util::validate_path(&target, project_root)?;

    // Read as bytes first for binary detection
    let bytes = std::fs::read(&resolved)
        .map_err(|e| anyhow::anyhow!("Failed to read file '{}': {}", path, e))?;

    if util::is_binary(&bytes) {
        bail!("File '{}' appears to be binary — cannot edit", path);
    }

    let content = String::from_utf8(bytes)
        .map_err(|_| anyhow::anyhow!("File '{}' contains invalid UTF-8", path))?;

    if replace_all {
        // Non-overlapping left-to-right count, consistent with str::replace semantics
        let count = content.matches(old_string).count();
        if count == 0 {
            bail!("old_string not found in '{}'", path);
        }
        let new_content = content.replace(old_string, new_string);
        std::fs::write(&resolved, &new_content)
            .map_err(|e| anyhow::anyhow!("Failed to write file '{}': {}", path, e))?;
        Ok(format!("Replaced {} occurrences in {}", count, path))
    } else {
        // Count occurrences with overlapping detection.
        // str::matches is non-overlapping, so we scan manually to catch cases
        // like old_string="aa" in "aaa" (positions 0 and 1).
        let count = count_occurrences(&content, old_string);

        if count == 0 {
            bail!("old_string not found in '{}'", path);
        }

        if count > 1 {
            bail!(
                "old_string matches {} locations in '{}' — provide more context to match exactly once",
                count,
                path
            );
        }

        let new_content = content.replacen(old_string, new_string, 1);
        std::fs::write(&resolved, &new_content)
            .map_err(|e| anyhow::anyhow!("Failed to write file '{}': {}", path, e))?;

        // Return a few lines of surrounding context
        let replacement_start = content.find(old_string).unwrap();
        let byte_offset_in_new = replacement_start;
        let context_snippet =
            extract_edit_context(&new_content, byte_offset_in_new, new_string.len());

        Ok(format!("Edited {}\n\n{}", path, context_snippet))
    }
}

/// Extract a few lines of context around the edited region.
fn extract_edit_context(content: &str, byte_offset: usize, replacement_len: usize) -> String {
    let lines: Vec<&str> = content.lines().collect();
    let context_lines = 3;

    // Find which line the edit starts and ends on
    let mut byte_pos = 0;
    let mut start_line = 0;
    let mut end_line = 0;
    for (i, line) in lines.iter().enumerate() {
        let line_end = byte_pos + line.len() + 1; // +1 for newline
        if byte_pos <= byte_offset && byte_offset < line_end {
            start_line = i;
        }
        if byte_pos <= byte_offset + replacement_len && byte_offset + replacement_len <= line_end {
            end_line = i;
            break;
        }
        byte_pos = line_end;
    }

    let from = start_line.saturating_sub(context_lines);
    let to = (end_line + context_lines + 1).min(lines.len());

    lines[from..to]
        .iter()
        .enumerate()
        .map(|(i, line)| format!("{:>4} | {}", from + i + 1, line))
        .collect::<Vec<_>>()
        .join("\n")
}

async fn execute_shell_exec(arguments: &str, project_root: &Path) -> Result<String> {
    let args: Value = serde_json::from_str(arguments)?;
    let command = args["command"]
        .as_str()
        .ok_or_else(|| anyhow::anyhow!("Missing required parameter: command"))?;
    let cwd = match args["cwd"].as_str() {
        Some(cwd_str) => {
            // Resolve relative cwd against project_root
            let raw = Path::new(cwd_str);
            if raw.is_absolute() {
                raw.to_path_buf()
            } else {
                project_root.join(raw)
            }
        }
        None => project_root.to_path_buf(),
    };
    let timeout_secs = args["timeout_secs"].as_u64().unwrap_or(60).min(300);

    if !cwd.exists() {
        bail!("Working directory does not exist: {}", cwd.display());
    }

    // Containment check: cwd must resolve inside project root (symlink-safe)
    let canonical_cwd = cwd
        .canonicalize()
        .map_err(|e| anyhow::anyhow!("Failed to resolve cwd '{}': {e}", cwd.display()))?;
    let canonical_root = project_root
        .canonicalize()
        .map_err(|e| anyhow::anyhow!("Failed to resolve project root: {e}"))?;
    if !canonical_cwd.starts_with(&canonical_root) {
        bail!(
            "Working directory '{}' is outside the project root",
            cwd.display()
        );
    }

    let child = Command::new("sh")
        .arg("-c")
        .arg(command)
        .current_dir(&canonical_cwd)
        .stdout(std::process::Stdio::piped())
        .stderr(std::process::Stdio::piped())
        .kill_on_drop(true)
        .spawn()
        .map_err(|e| anyhow::anyhow!("Failed to spawn command: {e}"))?;

    let output = match timeout(Duration::from_secs(timeout_secs), child.wait_with_output()).await {
        Ok(Ok(output)) => output,
        Ok(Err(e)) => bail!("Failed to execute command: {e}"),
        Err(_) => bail!("Command timed out after {timeout_secs}s"),
    };

    let stdout = String::from_utf8_lossy(&output.stdout);
    let stderr = String::from_utf8_lossy(&output.stderr);

    let mut result = String::new();
    if !stdout.is_empty() {
        result.push_str(&stdout);
    }
    if !stderr.is_empty() {
        if !result.is_empty() {
            result.push('\n');
        }
        result.push_str("[stderr]\n");
        result.push_str(&stderr);
    }
    if !output.status.success() {
        let code = output.status.code().unwrap_or(-1);
        if !result.is_empty() {
            result.push('\n');
        }
        result.push_str(&format!("[exit code: {code}]"));
    }

    if result.is_empty() {
        result.push_str("(no output)");
    }

    Ok(util::clip_for_model(&result))
}

async fn execute_grep(arguments: &str, project_root: &Path) -> Result<String> {
    let args: Value = serde_json::from_str(arguments)?;
    let pattern = args["pattern"]
        .as_str()
        .ok_or_else(|| anyhow::anyhow!("Missing required parameter: pattern"))?;

    let search_path = args["path"]
        .as_str()
        .map(|p| resolve_tool_path(p, project_root))
        .unwrap_or_else(|| project_root.to_path_buf());

    let max_results = args["max_results"].as_u64().unwrap_or(100) as usize;

    let mut cmd = Command::new("rg");
    cmd.arg("--line-number")
        .arg("--no-heading")
        .arg("--color=never");

    if args["case_insensitive"].as_bool().unwrap_or(false) {
        cmd.arg("--ignore-case");
    }

    if let Some(glob) = args["glob"].as_str() {
        cmd.arg("--glob").arg(glob);
    }

    cmd.arg("--max-count").arg(max_results.to_string());

    cmd.arg(pattern).arg(&search_path);

    cmd.stdout(std::process::Stdio::piped())
        .stderr(std::process::Stdio::piped())
        .kill_on_drop(true);

    let output = match timeout(Duration::from_secs(30), cmd.output()).await {
        Ok(Ok(output)) => output,
        Ok(Err(e)) => {
            let msg = e.to_string();
            if msg.contains("No such file or directory") || msg.contains("not found") {
                bail!("ripgrep (rg) is not installed. Install it with: cargo install ripgrep");
            }
            bail!("Failed to run ripgrep: {e}");
        }
        Err(_) => bail!("Grep timed out after 30s"),
    };

    match output.status.code() {
        Some(0) => {
            // Matches found — cap output lines
            let stdout = String::from_utf8_lossy(&output.stdout);
            let lines: Vec<&str> = stdout.lines().collect();
            if lines.len() > max_results {
                let truncated: String = lines[..max_results].join("\n");
                Ok(format!(
                    "{truncated}\n\n... (results capped at {max_results})"
                ))
            } else {
                Ok(util::clip_for_model(stdout.trim_end()))
            }
        }
        Some(1) => {
            // No matches
            Ok("No matches found.".to_string())
        }
        Some(2) => {
            // Error (invalid regex, etc.)
            let stderr = String::from_utf8_lossy(&output.stderr);
            bail!("Grep error: {}", stderr.trim())
        }
        _ => {
            let stderr = String::from_utf8_lossy(&output.stderr);
            if stderr.contains("not found") || stderr.contains("No such file") {
                bail!("ripgrep (rg) is not installed. Install it with: cargo install ripgrep");
            }
            bail!("Grep failed: {}", stderr.trim())
        }
    }
}

fn execute_list_files(arguments: &str, project_root: &std::path::Path) -> Result<String> {
    let args: Value = serde_json::from_str(arguments)?;
    let path = args["path"]
        .as_str()
        .ok_or_else(|| anyhow::anyhow!("Missing required parameter: path"))?;

    let resolved = resolve_tool_path(path, project_root);

    let mut entries: Vec<String> = Vec::new();
    let read_dir = std::fs::read_dir(&resolved)
        .map_err(|e| anyhow::anyhow!("Failed to list directory '{}': {}", path, e))?;

    for entry in read_dir {
        let entry = entry?;
        let name = entry.file_name().to_string_lossy().to_string();
        if entry.file_type()?.is_dir() {
            entries.push(format!("{name}/"));
        } else {
            entries.push(name);
        }
    }

    entries.sort();
    Ok(entries.join("\n"))
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Write;
    use std::path::Path;

    // Dummy root for tools that don't need path validation
    fn dummy_root() -> &'static Path {
        Path::new("/tmp")
    }

    #[tokio::test]
    async fn file_read_success() {
        let mut tmp = tempfile::NamedTempFile::new().unwrap();
        write!(tmp, "hello world").unwrap();
        let path = tmp.path().to_str().unwrap();

        let args = json!({"path": path}).to_string();
        let result = Tool::FileRead.execute(&args, dummy_root()).await;
        assert_eq!(result.output, "hello world");
    }

    #[tokio::test]
    async fn file_read_not_found() {
        let args = json!({"path": "/nonexistent/file.txt"}).to_string();
        let result = Tool::FileRead.execute(&args, dummy_root()).await;
        assert!(!result.success);
        assert!(result.output.contains("Failed to read file"));
    }

    #[tokio::test]
    async fn file_read_missing_path_param() {
        let args = json!({}).to_string();
        let result = Tool::FileRead.execute(&args, dummy_root()).await;
        assert!(!result.success);
        assert!(result.output.contains("Missing required parameter"));
    }

    #[test]
    fn from_name_known() {
        assert!(Tool::from_name("file_read").is_some());
        assert!(Tool::from_name("file_write").is_some());
        assert!(Tool::from_name("file_edit").is_some());
        assert!(Tool::from_name("list_files").is_some());
        assert!(Tool::from_name("shell_exec").is_some());
        assert!(Tool::from_name("grep").is_some());
    }

    #[test]
    fn from_name_unknown() {
        assert!(Tool::from_name("unknown_tool").is_none());
    }

    #[test]
    fn definition_has_required_fields() {
        for tool in Tool::all() {
            let def = tool.definition();
            assert!(def["name"].is_string());
            assert_eq!(def["type"], "function");
            assert!(def["parameters"].is_object());
        }
    }

    #[tokio::test]
    async fn file_read_binary_rejected() {
        let mut tmp = tempfile::NamedTempFile::new().unwrap();
        Write::write_all(&mut tmp, &[0x89, 0x50, 0x4E, 0x47, 0x00, 0x00]).unwrap();
        let path = tmp.path().to_str().unwrap();

        let args = json!({"path": path}).to_string();
        let result = Tool::FileRead.execute(&args, dummy_root()).await;
        assert!(!result.success);
        assert!(result.output.contains("binary"));
    }

    #[tokio::test]
    async fn file_read_large_file_clipped() {
        let mut tmp = tempfile::NamedTempFile::new().unwrap();
        let content = "x".repeat(50_000);
        Write::write_all(&mut tmp, content.as_bytes()).unwrap();
        let path = tmp.path().to_str().unwrap();

        let args = json!({"path": path}).to_string();
        let result = Tool::FileRead.execute(&args, dummy_root()).await;
        assert!(result.output.len() < content.len());
        assert!(result.output.contains("truncated"));
    }

    // --- file_write tests ---

    #[tokio::test]
    async fn file_write_creates_file() {
        let dir = tempfile::tempdir().unwrap();
        let file_path = dir.path().join("new.txt");
        let args = json!({
            "path": file_path.to_str().unwrap(),
            "content": "hello write"
        })
        .to_string();

        let result = Tool::FileWrite.execute(&args, dir.path()).await;
        assert!(result.output.contains("11 bytes"));
        assert_eq!(std::fs::read_to_string(&file_path).unwrap(), "hello write");
    }

    #[tokio::test]
    async fn file_write_creates_parent_dirs() {
        let dir = tempfile::tempdir().unwrap();
        let file_path = dir.path().join("deep/nested/file.txt");
        let args = json!({
            "path": file_path.to_str().unwrap(),
            "content": "nested"
        })
        .to_string();

        let result = Tool::FileWrite.execute(&args, dir.path()).await;
        assert!(result.output.contains("bytes"));
        assert_eq!(std::fs::read_to_string(&file_path).unwrap(), "nested");
    }

    #[cfg(unix)]
    #[tokio::test]
    async fn file_write_rejects_symlink_escape() {
        let dir = tempfile::tempdir().unwrap();
        let outside = tempfile::tempdir().unwrap();

        let link = dir.path().join("escape_link");
        std::os::unix::fs::symlink(outside.path(), &link).unwrap();

        let file_path = link.join("evil.txt");
        let args = json!({
            "path": file_path.to_str().unwrap(),
            "content": "evil"
        })
        .to_string();

        let result = Tool::FileWrite.execute(&args, dir.path()).await;
        assert!(!result.success);
        assert!(result.output.contains("outside the project root"));
    }

    // --- file_edit tests ---

    #[tokio::test]
    async fn file_edit_single_match() {
        let dir = tempfile::tempdir().unwrap();
        let file_path = dir.path().join("test.txt");
        std::fs::write(&file_path, "hello world").unwrap();

        let args = json!({
            "path": file_path.to_str().unwrap(),
            "old_string": "hello",
            "new_string": "goodbye"
        })
        .to_string();

        let result = Tool::FileEdit.execute(&args, dir.path()).await;
        assert!(result.output.contains("Edited"));
        assert_eq!(
            std::fs::read_to_string(&file_path).unwrap(),
            "goodbye world"
        );
    }

    #[tokio::test]
    async fn file_edit_returns_context() {
        let dir = tempfile::tempdir().unwrap();
        let file_path = dir.path().join("test.txt");
        let content = "line 1\nline 2\nline 3\nline 4\nline 5\nline 6\nline 7\n";
        std::fs::write(&file_path, content).unwrap();

        let args = json!({
            "path": file_path.to_str().unwrap(),
            "old_string": "line 4",
            "new_string": "LINE FOUR"
        })
        .to_string();

        let result = Tool::FileEdit.execute(&args, dir.path()).await;
        assert!(result.output.contains("LINE FOUR"));
        // Should show surrounding lines
        assert!(result.output.contains("line 2") || result.output.contains("line 3"));
    }

    #[tokio::test]
    async fn file_edit_zero_matches() {
        let dir = tempfile::tempdir().unwrap();
        let file_path = dir.path().join("test.txt");
        std::fs::write(&file_path, "hello world").unwrap();

        let args = json!({
            "path": file_path.to_str().unwrap(),
            "old_string": "nonexistent",
            "new_string": "replacement"
        })
        .to_string();

        let result = Tool::FileEdit.execute(&args, dir.path()).await;
        assert!(!result.success);
        assert!(result.output.contains("not found"));
    }

    #[tokio::test]
    async fn file_edit_multiple_matches() {
        let dir = tempfile::tempdir().unwrap();
        let file_path = dir.path().join("test.txt");
        std::fs::write(&file_path, "hello hello hello").unwrap();

        let args = json!({
            "path": file_path.to_str().unwrap(),
            "old_string": "hello",
            "new_string": "goodbye"
        })
        .to_string();

        let result = Tool::FileEdit.execute(&args, dir.path()).await;
        assert!(!result.success);
        let err = &result.output;
        assert!(err.contains("3 locations"));
        // File should be unchanged
        assert_eq!(
            std::fs::read_to_string(&file_path).unwrap(),
            "hello hello hello"
        );
    }

    #[tokio::test]
    async fn file_edit_empty_old_string() {
        let dir = tempfile::tempdir().unwrap();
        let file_path = dir.path().join("test.txt");
        std::fs::write(&file_path, "hello").unwrap();

        let args = json!({
            "path": file_path.to_str().unwrap(),
            "old_string": "",
            "new_string": "x"
        })
        .to_string();

        let result = Tool::FileEdit.execute(&args, dir.path()).await;
        assert!(!result.success);
        assert!(result.output.contains("must not be empty"));
    }

    #[tokio::test]
    async fn file_edit_identical_strings() {
        let dir = tempfile::tempdir().unwrap();
        let file_path = dir.path().join("test.txt");
        std::fs::write(&file_path, "hello").unwrap();

        let args = json!({
            "path": file_path.to_str().unwrap(),
            "old_string": "hello",
            "new_string": "hello"
        })
        .to_string();

        let result = Tool::FileEdit.execute(&args, dir.path()).await;
        assert!(!result.success);
        assert!(result.output.contains("identical"));
    }

    #[tokio::test]
    async fn file_edit_file_not_found() {
        let dir = tempfile::tempdir().unwrap();
        let file_path = dir.path().join("nonexistent.txt");

        let args = json!({
            "path": file_path.to_str().unwrap(),
            "old_string": "hello",
            "new_string": "goodbye"
        })
        .to_string();

        let result = Tool::FileEdit.execute(&args, dir.path()).await;
        assert!(!result.success);
    }

    #[tokio::test]
    async fn file_edit_binary_file() {
        let dir = tempfile::tempdir().unwrap();
        let file_path = dir.path().join("binary.bin");
        std::fs::write(&file_path, [0x89, 0x50, 0x4E, 0x47, 0x00, 0x00]).unwrap();

        let args = json!({
            "path": file_path.to_str().unwrap(),
            "old_string": "PNG",
            "new_string": "JPG"
        })
        .to_string();

        let result = Tool::FileEdit.execute(&args, dir.path()).await;
        assert!(!result.success);
        assert!(result.output.contains("binary"));
    }

    #[tokio::test]
    async fn file_edit_outside_project_root() {
        let dir = tempfile::tempdir().unwrap();
        let outside = tempfile::tempdir().unwrap();
        let file_path = outside.path().join("escape.txt");
        std::fs::write(&file_path, "secret").unwrap();

        let args = json!({
            "path": file_path.to_str().unwrap(),
            "old_string": "secret",
            "new_string": "safe"
        })
        .to_string();

        let result = Tool::FileEdit.execute(&args, dir.path()).await;
        assert!(!result.success);
        assert!(result.output.contains("outside the project root"));
    }

    #[cfg(unix)]
    #[tokio::test]
    async fn file_edit_rejects_symlink_escape() {
        let dir = tempfile::tempdir().unwrap();
        let outside = tempfile::tempdir().unwrap();
        let outside_file = outside.path().join("target.txt");
        std::fs::write(&outside_file, "secret data").unwrap();

        let link = dir.path().join("link.txt");
        std::os::unix::fs::symlink(&outside_file, &link).unwrap();

        let args = json!({
            "path": link.to_str().unwrap(),
            "old_string": "secret",
            "new_string": "safe"
        })
        .to_string();

        let result = Tool::FileEdit.execute(&args, dir.path()).await;
        assert!(!result.success);
        assert!(result.output.contains("outside the project root"));
    }

    #[tokio::test]
    async fn file_edit_overlapping_matches_detected() {
        let dir = tempfile::tempdir().unwrap();
        let file_path = dir.path().join("overlap.txt");
        std::fs::write(&file_path, "aaa").unwrap();

        let args = json!({
            "path": file_path.to_str().unwrap(),
            "old_string": "aa",
            "new_string": "bb"
        })
        .to_string();

        let result = Tool::FileEdit.execute(&args, dir.path()).await;
        assert!(!result.success);
        let err = &result.output;
        assert!(
            err.contains("2 locations"),
            "expected 2 overlapping matches, got: {err}"
        );
        // File should be unchanged
        assert_eq!(std::fs::read_to_string(&file_path).unwrap(), "aaa");
    }

    #[tokio::test]
    async fn file_read_relative_path_resolves_to_project_root() {
        let dir = tempfile::tempdir().unwrap();
        std::fs::create_dir(dir.path().join("src")).unwrap();
        std::fs::write(dir.path().join("src/lib.rs"), "pub fn hello() {}").unwrap();

        let args = json!({"path": "src/lib.rs"}).to_string();
        let result = Tool::FileRead.execute(&args, dir.path()).await;
        assert!(result.output.contains("pub fn hello()"));
    }

    #[tokio::test]
    async fn file_edit_relative_path_resolves_to_project_root() {
        let dir = tempfile::tempdir().unwrap();
        std::fs::create_dir(dir.path().join("src")).unwrap();
        std::fs::write(dir.path().join("src/lib.rs"), "old content").unwrap();

        let args = json!({
            "path": "src/lib.rs",
            "old_string": "old content",
            "new_string": "new content"
        })
        .to_string();

        let result = Tool::FileEdit.execute(&args, dir.path()).await;
        assert!(result.output.contains("Edited"));
        assert_eq!(
            std::fs::read_to_string(dir.path().join("src/lib.rs")).unwrap(),
            "new content"
        );
    }

    #[tokio::test]
    async fn list_files_relative_path_resolves_to_project_root() {
        let dir = tempfile::tempdir().unwrap();
        std::fs::create_dir(dir.path().join("src")).unwrap();
        std::fs::write(dir.path().join("src/main.rs"), "").unwrap();

        let args = json!({"path": "src"}).to_string();
        let result = Tool::ListFiles.execute(&args, dir.path()).await;
        assert!(result.output.contains("main.rs"));
    }

    #[test]
    fn count_occurrences_non_overlapping() {
        assert_eq!(count_occurrences("hello hello hello", "hello"), 3);
    }

    #[test]
    fn count_occurrences_overlapping() {
        assert_eq!(count_occurrences("aaa", "aa"), 2);
        assert_eq!(count_occurrences("aaaa", "aa"), 3);
    }

    #[test]
    fn count_occurrences_none() {
        assert_eq!(count_occurrences("hello", "xyz"), 0);
    }

    #[test]
    fn count_occurrences_empty_needle() {
        assert_eq!(count_occurrences("hello", ""), 0);
    }

    #[test]
    fn count_occurrences_multibyte_no_panic() {
        // "é" is 2 bytes in UTF-8. "éé" contains overlapping "é" twice.
        assert_eq!(count_occurrences("éé", "é"), 2);
        // "ée" should have 1 "é"
        assert_eq!(count_occurrences("ée", "é"), 1);
    }

    #[test]
    fn count_occurrences_multibyte_overlapping() {
        // "ééé" with needle "éé" — overlapping matches at positions 0 and 1 (char-wise)
        assert_eq!(count_occurrences("ééé", "éé"), 2);
    }

    // --- file_edit replace_all tests ---

    #[tokio::test]
    async fn file_edit_replace_all_replaces_all_occurrences() {
        let dir = tempfile::tempdir().unwrap();
        let file_path = dir.path().join("test.txt");
        std::fs::write(&file_path, "foo bar foo baz foo").unwrap();

        let args = json!({
            "path": file_path.to_str().unwrap(),
            "old_string": "foo",
            "new_string": "qux",
            "replace_all": true
        })
        .to_string();

        let result = Tool::FileEdit.execute(&args, dir.path()).await;
        assert!(result.success);
        assert!(result.output.contains("Replaced 3 occurrences"));
        assert_eq!(
            std::fs::read_to_string(&file_path).unwrap(),
            "qux bar qux baz qux"
        );
    }

    #[tokio::test]
    async fn file_edit_replace_all_zero_matches_errors() {
        let dir = tempfile::tempdir().unwrap();
        let file_path = dir.path().join("test.txt");
        std::fs::write(&file_path, "hello world").unwrap();

        let args = json!({
            "path": file_path.to_str().unwrap(),
            "old_string": "nonexistent",
            "new_string": "replacement",
            "replace_all": true
        })
        .to_string();

        let result = Tool::FileEdit.execute(&args, dir.path()).await;
        assert!(!result.success);
        assert!(result.output.contains("not found"));
    }

    #[tokio::test]
    async fn file_edit_replace_all_non_overlapping_semantics() {
        // "aa" in "aaa" should yield one replacement (non-overlapping left-to-right)
        let dir = tempfile::tempdir().unwrap();
        let file_path = dir.path().join("test.txt");
        std::fs::write(&file_path, "aaa").unwrap();

        let args = json!({
            "path": file_path.to_str().unwrap(),
            "old_string": "aa",
            "new_string": "X",
            "replace_all": true
        })
        .to_string();

        let result = Tool::FileEdit.execute(&args, dir.path()).await;
        assert!(result.success);
        assert!(
            result.output.contains("Replaced 1 occurrences"),
            "expected 1 replacement, got: {}",
            result.output
        );
        assert_eq!(std::fs::read_to_string(&file_path).unwrap(), "Xa");
    }

    #[tokio::test]
    async fn file_edit_replace_all_false_preserves_single_match_behavior() {
        let dir = tempfile::tempdir().unwrap();
        let file_path = dir.path().join("test.txt");
        std::fs::write(&file_path, "hello hello hello").unwrap();

        // replace_all explicitly false — should fail like the default (multiple matches)
        let args = json!({
            "path": file_path.to_str().unwrap(),
            "old_string": "hello",
            "new_string": "goodbye",
            "replace_all": false
        })
        .to_string();

        let result = Tool::FileEdit.execute(&args, dir.path()).await;
        assert!(!result.success);
        assert!(result.output.contains("3 locations"));
        assert_eq!(
            std::fs::read_to_string(&file_path).unwrap(),
            "hello hello hello"
        );
    }

    #[tokio::test]
    async fn file_edit_replace_all_single_occurrence() {
        let dir = tempfile::tempdir().unwrap();
        let file_path = dir.path().join("test.txt");
        std::fs::write(&file_path, "hello world").unwrap();

        let args = json!({
            "path": file_path.to_str().unwrap(),
            "old_string": "hello",
            "new_string": "goodbye",
            "replace_all": true
        })
        .to_string();

        let result = Tool::FileEdit.execute(&args, dir.path()).await;
        assert!(result.success);
        assert!(result.output.contains("Replaced 1 occurrences"));
        assert_eq!(
            std::fs::read_to_string(&file_path).unwrap(),
            "goodbye world"
        );
    }

    #[tokio::test]
    async fn file_edit_multibyte_content_no_panic() {
        let dir = tempfile::tempdir().unwrap();
        let file_path = dir.path().join("utf8.txt");
        std::fs::write(&file_path, "café résumé").unwrap();

        let args = json!({
            "path": file_path.to_str().unwrap(),
            "old_string": "café",
            "new_string": "coffee"
        })
        .to_string();

        let result = Tool::FileEdit.execute(&args, dir.path()).await;
        assert!(result.output.contains("Edited"));
        assert_eq!(
            std::fs::read_to_string(&file_path).unwrap(),
            "coffee résumé"
        );
    }

    // --- list_files tests ---

    #[tokio::test]
    async fn list_files_populated_dir() {
        let dir = tempfile::tempdir().unwrap();
        std::fs::write(dir.path().join("a.txt"), "").unwrap();
        std::fs::write(dir.path().join("b.rs"), "").unwrap();
        std::fs::create_dir(dir.path().join("subdir")).unwrap();

        let args = json!({"path": dir.path().to_str().unwrap()}).to_string();
        let result = Tool::ListFiles.execute(&args, dummy_root()).await;

        assert!(result.output.contains("a.txt"));
        assert!(result.output.contains("b.rs"));
        assert!(result.output.contains("subdir/"));
    }

    #[tokio::test]
    async fn list_files_empty_dir() {
        let dir = tempfile::tempdir().unwrap();
        let args = json!({"path": dir.path().to_str().unwrap()}).to_string();
        let result = Tool::ListFiles.execute(&args, dummy_root()).await;
        assert_eq!(result.output, "");
    }

    #[tokio::test]
    async fn list_files_nonexistent() {
        let args = json!({"path": "/nonexistent/dir"}).to_string();
        let result = Tool::ListFiles.execute(&args, dummy_root()).await;
        assert!(!result.success);
        assert!(result.output.contains("Failed to list directory"));
    }

    // --- shell_exec tests ---

    #[tokio::test]
    async fn shell_exec_happy_path() {
        let dir = tempfile::tempdir().unwrap();
        let args = json!({"command": "echo hello"}).to_string();
        let result = Tool::ShellExec.execute(&args, dir.path()).await;
        assert_eq!(result.output.trim(), "hello");
    }

    #[tokio::test]
    async fn shell_exec_nonzero_exit() {
        let dir = tempfile::tempdir().unwrap();
        let args = json!({"command": "exit 42"}).to_string();
        let result = Tool::ShellExec.execute(&args, dir.path()).await;
        assert!(result.output.contains("[exit code: 42]"));
    }

    #[tokio::test]
    async fn shell_exec_captures_stderr() {
        let dir = tempfile::tempdir().unwrap();
        let args = json!({"command": "echo err >&2"}).to_string();
        let result = Tool::ShellExec.execute(&args, dir.path()).await;
        assert!(result.output.contains("[stderr]"));
        assert!(result.output.contains("err"));
    }

    #[tokio::test]
    async fn shell_exec_timeout_kills_process() {
        let dir = tempfile::tempdir().unwrap();
        let args = json!({"command": "sleep 60", "timeout_secs": 1}).to_string();
        let result = Tool::ShellExec.execute(&args, dir.path()).await;
        assert!(!result.success);
        assert!(result.output.contains("timed out"));
    }

    #[tokio::test]
    async fn shell_exec_custom_cwd() {
        let dir = tempfile::tempdir().unwrap();
        let sub = dir.path().join("subdir");
        std::fs::create_dir(&sub).unwrap();

        let args = json!({
            "command": "pwd",
            "cwd": sub.to_str().unwrap()
        })
        .to_string();
        let result = Tool::ShellExec.execute(&args, dir.path()).await;
        // pwd output should contain the subdir path
        assert!(result.output.contains("subdir"));
    }

    #[tokio::test]
    async fn shell_exec_default_cwd_is_project_root() {
        let dir = tempfile::tempdir().unwrap();
        let args = json!({"command": "pwd"}).to_string();
        let result = Tool::ShellExec.execute(&args, dir.path()).await;
        // The canonical path of the tempdir should appear in pwd output
        let canonical = dir.path().canonicalize().unwrap();
        assert!(result.output.contains(canonical.to_str().unwrap()));
    }

    #[tokio::test]
    async fn shell_exec_nonexistent_cwd() {
        let dir = tempfile::tempdir().unwrap();
        let args = json!({
            "command": "echo hi",
            "cwd": "/nonexistent/dir"
        })
        .to_string();
        let result = Tool::ShellExec.execute(&args, dir.path()).await;
        assert!(!result.success);
        assert!(result.output.contains("does not exist"));
    }

    #[tokio::test]
    async fn shell_exec_timeout_capped_at_300() {
        let dir = tempfile::tempdir().unwrap();
        // Request 999 seconds — should be capped to 300
        let args = json!({"command": "echo ok", "timeout_secs": 999}).to_string();
        let result = Tool::ShellExec.execute(&args, dir.path()).await;
        assert_eq!(result.output.trim(), "ok");
    }

    #[tokio::test]
    async fn shell_exec_relative_cwd_resolves_to_project_root() {
        let dir = tempfile::tempdir().unwrap();
        let sub = dir.path().join("src");
        std::fs::create_dir(&sub).unwrap();

        let args = json!({
            "command": "pwd",
            "cwd": "src"
        })
        .to_string();
        let result = Tool::ShellExec.execute(&args, dir.path()).await;
        assert!(result.output.contains("src"));
    }

    #[tokio::test]
    async fn shell_exec_absolute_cwd_inside_project_works() {
        let dir = tempfile::tempdir().unwrap();
        let sub = dir.path().join("inner");
        std::fs::create_dir(&sub).unwrap();

        let args = json!({
            "command": "pwd",
            "cwd": sub.to_str().unwrap()
        })
        .to_string();
        let result = Tool::ShellExec.execute(&args, dir.path()).await;
        assert!(result.output.contains("inner"));
    }

    #[tokio::test]
    async fn shell_exec_cwd_escaping_project_root_rejected() {
        let dir = tempfile::tempdir().unwrap();
        let outside = tempfile::tempdir().unwrap();

        let args = json!({
            "command": "echo hi",
            "cwd": outside.path().to_str().unwrap()
        })
        .to_string();
        let result = Tool::ShellExec.execute(&args, dir.path()).await;
        assert!(!result.success);
        assert!(result.output.contains("outside the project root"));
    }

    #[tokio::test]
    async fn shell_exec_relative_cwd_escape_rejected() {
        let dir = tempfile::tempdir().unwrap();
        // Create a subdirectory so the project root has content
        std::fs::create_dir(dir.path().join("sub")).unwrap();

        // Use .. to escape to the parent of the tempdir
        let args = json!({
            "command": "echo hi",
            "cwd": ".."
        })
        .to_string();
        let result = Tool::ShellExec.execute(&args, dir.path()).await;
        assert!(!result.success);
        assert!(result.output.contains("outside the project root"));
    }

    #[cfg(unix)]
    #[tokio::test]
    async fn shell_exec_symlinked_cwd_rejected() {
        let dir = tempfile::tempdir().unwrap();
        let outside = tempfile::tempdir().unwrap();

        // Create a symlink inside the project root that points outside
        let link = dir.path().join("escape_link");
        std::os::unix::fs::symlink(outside.path(), &link).unwrap();

        let args = json!({
            "command": "pwd",
            "cwd": link.to_str().unwrap()
        })
        .to_string();
        let result = Tool::ShellExec.execute(&args, dir.path()).await;
        assert!(!result.success);
        assert!(result.output.contains("outside the project root"));
    }

    // --- grep tests ---

    #[tokio::test]
    async fn grep_matches_found() {
        let dir = tempfile::tempdir().unwrap();
        std::fs::write(
            dir.path().join("hello.txt"),
            "hello world\ngoodbye world\nhello again",
        )
        .unwrap();

        let args = json!({"pattern": "hello", "path": dir.path().to_str().unwrap()}).to_string();
        let result = Tool::Grep.execute(&args, dir.path()).await;
        assert!(result.output.contains("hello"));
        assert!(result.output.contains("hello.txt"));
    }

    #[tokio::test]
    async fn grep_no_matches() {
        let dir = tempfile::tempdir().unwrap();
        std::fs::write(dir.path().join("test.txt"), "nothing here").unwrap();

        let args = json!({"pattern": "xyz123", "path": dir.path().to_str().unwrap()}).to_string();
        let result = Tool::Grep.execute(&args, dir.path()).await;
        assert_eq!(result.output, "No matches found.");
    }

    #[tokio::test]
    async fn grep_invalid_regex() {
        let dir = tempfile::tempdir().unwrap();
        std::fs::write(dir.path().join("test.txt"), "content").unwrap();

        let args = json!({"pattern": "[invalid", "path": dir.path().to_str().unwrap()}).to_string();
        let result = Tool::Grep.execute(&args, dir.path()).await;
        assert!(!result.success);
        assert!(result.output.contains("Grep error"));
    }

    #[tokio::test]
    async fn grep_glob_filter() {
        let dir = tempfile::tempdir().unwrap();
        std::fs::write(dir.path().join("code.rs"), "fn main() {}").unwrap();
        std::fs::write(dir.path().join("notes.txt"), "fn notes").unwrap();

        let args = json!({
            "pattern": "fn",
            "path": dir.path().to_str().unwrap(),
            "glob": "*.rs"
        })
        .to_string();
        let result = Tool::Grep.execute(&args, dir.path()).await;
        assert!(result.output.contains("code.rs"));
        assert!(!result.output.contains("notes.txt"));
    }

    #[tokio::test]
    async fn grep_case_insensitive() {
        let dir = tempfile::tempdir().unwrap();
        std::fs::write(dir.path().join("test.txt"), "Hello World\nhello world").unwrap();

        let args = json!({
            "pattern": "HELLO",
            "path": dir.path().to_str().unwrap(),
            "case_insensitive": true
        })
        .to_string();
        let result = Tool::Grep.execute(&args, dir.path()).await;
        // Should match both lines
        assert!(result.output.contains("Hello"));
        assert!(result.output.contains("hello"));
    }

    #[tokio::test]
    async fn grep_max_results_cap() {
        let dir = tempfile::tempdir().unwrap();
        let content: String = (0..50).map(|i| format!("match line {i}\n")).collect();
        std::fs::write(dir.path().join("many.txt"), &content).unwrap();

        let args = json!({
            "pattern": "match",
            "path": dir.path().to_str().unwrap(),
            "max_results": 5
        })
        .to_string();
        let result = Tool::Grep.execute(&args, dir.path()).await;
        let match_lines: Vec<&str> = result
            .output
            .lines()
            .filter(|l| l.contains("match"))
            .collect();
        assert!(match_lines.len() <= 5);
    }

    #[tokio::test]
    async fn grep_defaults_to_project_root() {
        let dir = tempfile::tempdir().unwrap();
        std::fs::write(dir.path().join("root_file.txt"), "findme here").unwrap();

        // No path param — should search project root
        let args = json!({"pattern": "findme"}).to_string();
        let result = Tool::Grep.execute(&args, dir.path()).await;
        assert!(result.output.contains("findme"));
    }
}
