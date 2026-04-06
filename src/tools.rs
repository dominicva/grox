use std::path::{Path, PathBuf};

use anyhow::{Result, bail};
use serde_json::{Value, json};
use tokio::process::Command;
use tokio::time::{Duration, timeout};

use crate::util;

#[derive(Debug, Clone)]
pub enum Tool {
    FileRead,
    FileWrite,
    FileEdit,
    ListFiles,
    ShellExec,
    Grep,
}

#[derive(Debug, Clone)]
pub struct ToolCall {
    pub call_id: String,
    pub name: String,
    pub arguments: String,
}

impl Tool {
    pub fn all() -> Vec<Tool> {
        vec![Tool::FileRead, Tool::FileWrite, Tool::FileEdit, Tool::ListFiles, Tool::ShellExec, Tool::Grep]
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
                "description": "Edit a file by replacing a single occurrence of a string. The old_string must match exactly one location in the file. If it matches zero or multiple locations, the edit fails with an error. Use this for surgical edits — include enough surrounding context in old_string to ensure a unique match.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "path": {
                            "type": "string",
                            "description": "The path to the file to edit"
                        },
                        "old_string": {
                            "type": "string",
                            "description": "The exact string to find and replace (must match exactly once)"
                        },
                        "new_string": {
                            "type": "string",
                            "description": "The replacement string"
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

    pub async fn execute(&self, arguments: &str, project_root: &std::path::Path) -> Result<String> {
        match self {
            Tool::FileRead => execute_file_read(arguments),
            Tool::FileWrite => execute_file_write(arguments, project_root),
            Tool::FileEdit => execute_file_edit(arguments, project_root),
            Tool::ListFiles => execute_list_files(arguments),
            Tool::ShellExec => execute_shell_exec(arguments, project_root).await,
            Tool::Grep => execute_grep(arguments, project_root).await,
        }
    }
}

fn execute_file_read(arguments: &str) -> Result<String> {
    let args: Value = serde_json::from_str(arguments)?;
    let path = args["path"]
        .as_str()
        .ok_or_else(|| anyhow::anyhow!("Missing required parameter: path"))?;

    // Read as bytes first for binary detection
    let bytes = std::fs::read(path)
        .map_err(|e| anyhow::anyhow!("Failed to read file '{}': {}", path, e))?;

    if util::is_binary(&bytes) {
        anyhow::bail!("File '{}' appears to be binary — cannot display contents", path);
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

    let target = std::path::Path::new(path);

    // For nested paths, create parent directories first so validate_path can resolve them.
    // We create within the project root to ensure safety before canonicalization.
    if let Some(parent) = target.parent() {
        if !parent.exists() {
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
            std::fs::create_dir_all(parent)
                .map_err(|e| anyhow::anyhow!("Failed to create directory '{}': {}", parent.display(), e))?;
        }
    }

    // Validate the final path is within project root (symlink-safe)
    let resolved = util::validate_path(target, project_root)?;

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

    if old_string.is_empty() {
        bail!("old_string must not be empty");
    }

    if old_string == new_string {
        bail!("old_string and new_string are identical — nothing to change");
    }

    let target = std::path::Path::new(path);
    let resolved = util::validate_path(target, project_root)?;

    // Read as bytes first for binary detection
    let bytes = std::fs::read(&resolved)
        .map_err(|e| anyhow::anyhow!("Failed to read file '{}': {}", path, e))?;

    if util::is_binary(&bytes) {
        bail!("File '{}' appears to be binary — cannot edit", path);
    }

    let content = String::from_utf8(bytes)
        .map_err(|_| anyhow::anyhow!("File '{}' contains invalid UTF-8", path))?;

    let count = content.matches(old_string).count();

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
    let context_snippet = extract_edit_context(&new_content, byte_offset_in_new, new_string.len());

    Ok(format!("Edited {}\n\n{}", path, context_snippet))
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
    let cwd = args["cwd"]
        .as_str()
        .map(std::path::PathBuf::from)
        .unwrap_or_else(|| project_root.to_path_buf());
    let timeout_secs = args["timeout_secs"]
        .as_u64()
        .unwrap_or(60)
        .min(300);

    if !cwd.exists() {
        bail!("Working directory does not exist: {}", cwd.display());
    }

    let child = Command::new("sh")
        .arg("-c")
        .arg(command)
        .current_dir(&cwd)
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
        .map(PathBuf::from)
        .unwrap_or_else(|| project_root.to_path_buf());

    let max_results = args["max_results"]
        .as_u64()
        .unwrap_or(100) as usize;

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
                Ok(format!("{truncated}\n\n... (results capped at {max_results})"))
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

fn execute_list_files(arguments: &str) -> Result<String> {
    let args: Value = serde_json::from_str(arguments)?;
    let path = args["path"]
        .as_str()
        .ok_or_else(|| anyhow::anyhow!("Missing required parameter: path"))?;

    let mut entries: Vec<String> = Vec::new();
    let read_dir = std::fs::read_dir(path)
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
        let result = Tool::FileRead.execute(&args, dummy_root()).await.unwrap();
        assert_eq!(result, "hello world");
    }

    #[tokio::test]
    async fn file_read_not_found() {
        let args = json!({"path": "/nonexistent/file.txt"}).to_string();
        let result = Tool::FileRead.execute(&args, dummy_root()).await;
        assert!(result.is_err());
        assert!(result.unwrap_err().to_string().contains("Failed to read file"));
    }

    #[tokio::test]
    async fn file_read_missing_path_param() {
        let args = json!({}).to_string();
        let result = Tool::FileRead.execute(&args, dummy_root()).await;
        assert!(result.is_err());
        assert!(result.unwrap_err().to_string().contains("Missing required parameter"));
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
        assert!(result.is_err());
        assert!(result.unwrap_err().to_string().contains("binary"));
    }

    #[tokio::test]
    async fn file_read_large_file_clipped() {
        let mut tmp = tempfile::NamedTempFile::new().unwrap();
        let content = "x".repeat(50_000);
        Write::write_all(&mut tmp, content.as_bytes()).unwrap();
        let path = tmp.path().to_str().unwrap();

        let args = json!({"path": path}).to_string();
        let result = Tool::FileRead.execute(&args, dummy_root()).await.unwrap();
        assert!(result.len() < content.len());
        assert!(result.contains("truncated"));
    }

    // --- file_write tests ---

    #[tokio::test]
    async fn file_write_creates_file() {
        let dir = tempfile::tempdir().unwrap();
        let file_path = dir.path().join("new.txt");
        let args = json!({
            "path": file_path.to_str().unwrap(),
            "content": "hello write"
        }).to_string();

        let result = Tool::FileWrite.execute(&args, dir.path()).await.unwrap();
        assert!(result.contains("11 bytes"));
        assert_eq!(std::fs::read_to_string(&file_path).unwrap(), "hello write");
    }

    #[tokio::test]
    async fn file_write_creates_parent_dirs() {
        let dir = tempfile::tempdir().unwrap();
        let file_path = dir.path().join("deep/nested/file.txt");
        let args = json!({
            "path": file_path.to_str().unwrap(),
            "content": "nested"
        }).to_string();

        let result = Tool::FileWrite.execute(&args, dir.path()).await.unwrap();
        assert!(result.contains("bytes"));
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
        }).to_string();

        let result = Tool::FileWrite.execute(&args, dir.path()).await;
        assert!(result.is_err());
        assert!(result.unwrap_err().to_string().contains("outside the project root"));
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
        }).to_string();

        let result = Tool::FileEdit.execute(&args, dir.path()).await.unwrap();
        assert!(result.contains("Edited"));
        assert_eq!(std::fs::read_to_string(&file_path).unwrap(), "goodbye world");
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
        }).to_string();

        let result = Tool::FileEdit.execute(&args, dir.path()).await.unwrap();
        assert!(result.contains("LINE FOUR"));
        // Should show surrounding lines
        assert!(result.contains("line 2") || result.contains("line 3"));
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
        }).to_string();

        let result = Tool::FileEdit.execute(&args, dir.path()).await;
        assert!(result.is_err());
        assert!(result.unwrap_err().to_string().contains("not found"));
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
        }).to_string();

        let result = Tool::FileEdit.execute(&args, dir.path()).await;
        assert!(result.is_err());
        let err = result.unwrap_err().to_string();
        assert!(err.contains("3 locations"));
        // File should be unchanged
        assert_eq!(std::fs::read_to_string(&file_path).unwrap(), "hello hello hello");
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
        }).to_string();

        let result = Tool::FileEdit.execute(&args, dir.path()).await;
        assert!(result.is_err());
        assert!(result.unwrap_err().to_string().contains("must not be empty"));
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
        }).to_string();

        let result = Tool::FileEdit.execute(&args, dir.path()).await;
        assert!(result.is_err());
        assert!(result.unwrap_err().to_string().contains("identical"));
    }

    #[tokio::test]
    async fn file_edit_file_not_found() {
        let dir = tempfile::tempdir().unwrap();
        let file_path = dir.path().join("nonexistent.txt");

        let args = json!({
            "path": file_path.to_str().unwrap(),
            "old_string": "hello",
            "new_string": "goodbye"
        }).to_string();

        let result = Tool::FileEdit.execute(&args, dir.path()).await;
        assert!(result.is_err());
    }

    #[tokio::test]
    async fn file_edit_binary_file() {
        let dir = tempfile::tempdir().unwrap();
        let file_path = dir.path().join("binary.bin");
        std::fs::write(&file_path, &[0x89, 0x50, 0x4E, 0x47, 0x00, 0x00]).unwrap();

        let args = json!({
            "path": file_path.to_str().unwrap(),
            "old_string": "PNG",
            "new_string": "JPG"
        }).to_string();

        let result = Tool::FileEdit.execute(&args, dir.path()).await;
        assert!(result.is_err());
        assert!(result.unwrap_err().to_string().contains("binary"));
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
        }).to_string();

        let result = Tool::FileEdit.execute(&args, dir.path()).await;
        assert!(result.is_err());
        assert!(result.unwrap_err().to_string().contains("outside the project root"));
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
        }).to_string();

        let result = Tool::FileEdit.execute(&args, dir.path()).await;
        assert!(result.is_err());
        assert!(result.unwrap_err().to_string().contains("outside the project root"));
    }

    // --- list_files tests ---

    #[tokio::test]
    async fn list_files_populated_dir() {
        let dir = tempfile::tempdir().unwrap();
        std::fs::write(dir.path().join("a.txt"), "").unwrap();
        std::fs::write(dir.path().join("b.rs"), "").unwrap();
        std::fs::create_dir(dir.path().join("subdir")).unwrap();

        let args = json!({"path": dir.path().to_str().unwrap()}).to_string();
        let result = Tool::ListFiles.execute(&args, dummy_root()).await.unwrap();

        assert!(result.contains("a.txt"));
        assert!(result.contains("b.rs"));
        assert!(result.contains("subdir/"));
    }

    #[tokio::test]
    async fn list_files_empty_dir() {
        let dir = tempfile::tempdir().unwrap();
        let args = json!({"path": dir.path().to_str().unwrap()}).to_string();
        let result = Tool::ListFiles.execute(&args, dummy_root()).await.unwrap();
        assert_eq!(result, "");
    }

    #[tokio::test]
    async fn list_files_nonexistent() {
        let args = json!({"path": "/nonexistent/dir"}).to_string();
        let result = Tool::ListFiles.execute(&args, dummy_root()).await;
        assert!(result.is_err());
        assert!(result.unwrap_err().to_string().contains("Failed to list directory"));
    }

    // --- shell_exec tests ---

    #[tokio::test]
    async fn shell_exec_happy_path() {
        let dir = tempfile::tempdir().unwrap();
        let args = json!({"command": "echo hello"}).to_string();
        let result = Tool::ShellExec.execute(&args, dir.path()).await.unwrap();
        assert_eq!(result.trim(), "hello");
    }

    #[tokio::test]
    async fn shell_exec_nonzero_exit() {
        let dir = tempfile::tempdir().unwrap();
        let args = json!({"command": "exit 42"}).to_string();
        let result = Tool::ShellExec.execute(&args, dir.path()).await.unwrap();
        assert!(result.contains("[exit code: 42]"));
    }

    #[tokio::test]
    async fn shell_exec_captures_stderr() {
        let dir = tempfile::tempdir().unwrap();
        let args = json!({"command": "echo err >&2"}).to_string();
        let result = Tool::ShellExec.execute(&args, dir.path()).await.unwrap();
        assert!(result.contains("[stderr]"));
        assert!(result.contains("err"));
    }

    #[tokio::test]
    async fn shell_exec_timeout_kills_process() {
        let dir = tempfile::tempdir().unwrap();
        let args = json!({"command": "sleep 60", "timeout_secs": 1}).to_string();
        let result = Tool::ShellExec.execute(&args, dir.path()).await;
        assert!(result.is_err());
        assert!(result.unwrap_err().to_string().contains("timed out"));
    }

    #[tokio::test]
    async fn shell_exec_custom_cwd() {
        let dir = tempfile::tempdir().unwrap();
        let sub = dir.path().join("subdir");
        std::fs::create_dir(&sub).unwrap();

        let args = json!({
            "command": "pwd",
            "cwd": sub.to_str().unwrap()
        }).to_string();
        let result = Tool::ShellExec.execute(&args, dir.path()).await.unwrap();
        // pwd output should contain the subdir path
        assert!(result.contains("subdir"));
    }

    #[tokio::test]
    async fn shell_exec_default_cwd_is_project_root() {
        let dir = tempfile::tempdir().unwrap();
        let args = json!({"command": "pwd"}).to_string();
        let result = Tool::ShellExec.execute(&args, dir.path()).await.unwrap();
        // The canonical path of the tempdir should appear in pwd output
        let canonical = dir.path().canonicalize().unwrap();
        assert!(result.contains(canonical.to_str().unwrap()));
    }

    #[tokio::test]
    async fn shell_exec_nonexistent_cwd() {
        let dir = tempfile::tempdir().unwrap();
        let args = json!({
            "command": "echo hi",
            "cwd": "/nonexistent/dir"
        }).to_string();
        let result = Tool::ShellExec.execute(&args, dir.path()).await;
        assert!(result.is_err());
        assert!(result.unwrap_err().to_string().contains("does not exist"));
    }

    #[tokio::test]
    async fn shell_exec_timeout_capped_at_300() {
        let dir = tempfile::tempdir().unwrap();
        // Request 999 seconds — should be capped to 300
        let args = json!({"command": "echo ok", "timeout_secs": 999}).to_string();
        let result = Tool::ShellExec.execute(&args, dir.path()).await.unwrap();
        assert_eq!(result.trim(), "ok");
    }

    // --- grep tests ---

    #[tokio::test]
    async fn grep_matches_found() {
        let dir = tempfile::tempdir().unwrap();
        std::fs::write(dir.path().join("hello.txt"), "hello world\ngoodbye world\nhello again").unwrap();

        let args = json!({"pattern": "hello", "path": dir.path().to_str().unwrap()}).to_string();
        let result = Tool::Grep.execute(&args, dir.path()).await.unwrap();
        assert!(result.contains("hello"));
        assert!(result.contains("hello.txt"));
    }

    #[tokio::test]
    async fn grep_no_matches() {
        let dir = tempfile::tempdir().unwrap();
        std::fs::write(dir.path().join("test.txt"), "nothing here").unwrap();

        let args = json!({"pattern": "xyz123", "path": dir.path().to_str().unwrap()}).to_string();
        let result = Tool::Grep.execute(&args, dir.path()).await.unwrap();
        assert_eq!(result, "No matches found.");
    }

    #[tokio::test]
    async fn grep_invalid_regex() {
        let dir = tempfile::tempdir().unwrap();
        std::fs::write(dir.path().join("test.txt"), "content").unwrap();

        let args = json!({"pattern": "[invalid", "path": dir.path().to_str().unwrap()}).to_string();
        let result = Tool::Grep.execute(&args, dir.path()).await;
        assert!(result.is_err());
        assert!(result.unwrap_err().to_string().contains("Grep error"));
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
        }).to_string();
        let result = Tool::Grep.execute(&args, dir.path()).await.unwrap();
        assert!(result.contains("code.rs"));
        assert!(!result.contains("notes.txt"));
    }

    #[tokio::test]
    async fn grep_case_insensitive() {
        let dir = tempfile::tempdir().unwrap();
        std::fs::write(dir.path().join("test.txt"), "Hello World\nhello world").unwrap();

        let args = json!({
            "pattern": "HELLO",
            "path": dir.path().to_str().unwrap(),
            "case_insensitive": true
        }).to_string();
        let result = Tool::Grep.execute(&args, dir.path()).await.unwrap();
        // Should match both lines
        assert!(result.contains("Hello"));
        assert!(result.contains("hello"));
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
        }).to_string();
        let result = Tool::Grep.execute(&args, dir.path()).await.unwrap();
        let match_lines: Vec<&str> = result.lines().filter(|l| l.contains("match")).collect();
        assert!(match_lines.len() <= 5);
    }

    #[tokio::test]
    async fn grep_defaults_to_project_root() {
        let dir = tempfile::tempdir().unwrap();
        std::fs::write(dir.path().join("root_file.txt"), "findme here").unwrap();

        // No path param — should search project root
        let args = json!({"pattern": "findme"}).to_string();
        let result = Tool::Grep.execute(&args, dir.path()).await.unwrap();
        assert!(result.contains("findme"));
    }
}
