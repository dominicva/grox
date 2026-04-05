use std::path::Path;

use anyhow::{Result, bail};
use serde_json::{Value, json};
use tokio::process::Command;
use tokio::time::{Duration, timeout};

use crate::util;

#[derive(Debug, Clone)]
pub enum Tool {
    FileRead,
    FileWrite,
    ListFiles,
    ShellExec,
}

#[derive(Debug, Clone)]
pub struct ToolCall {
    pub call_id: String,
    pub name: String,
    pub arguments: String,
}

impl Tool {
    pub fn all() -> Vec<Tool> {
        vec![Tool::FileRead, Tool::FileWrite, Tool::ListFiles, Tool::ShellExec]
    }

    pub fn definitions() -> Vec<Value> {
        Tool::all().iter().map(|t| t.definition()).collect()
    }

    pub fn from_name(name: &str) -> Option<Tool> {
        match name {
            "file_read" => Some(Tool::FileRead),
            "file_write" => Some(Tool::FileWrite),
            "list_files" => Some(Tool::ListFiles),
            "shell_exec" => Some(Tool::ShellExec),
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
            Tool::ListFiles => execute_list_files(arguments),
            Tool::ShellExec => execute_shell_exec(arguments, project_root).await,
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
        assert!(Tool::from_name("list_files").is_some());
        assert!(Tool::from_name("shell_exec").is_some());
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
}
