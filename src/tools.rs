use anyhow::Result;
use serde_json::{Value, json};

#[derive(Debug, Clone)]
pub enum Tool {
    FileRead,
    ListFiles,
}

#[derive(Debug, Clone)]
pub struct ToolCall {
    pub call_id: String,
    pub name: String,
    pub arguments: String,
}

impl Tool {
    pub fn all() -> Vec<Tool> {
        vec![Tool::FileRead, Tool::ListFiles]
    }

    pub fn definitions() -> Vec<Value> {
        Tool::all().iter().map(|t| t.definition()).collect()
    }

    pub fn from_name(name: &str) -> Option<Tool> {
        match name {
            "file_read" => Some(Tool::FileRead),
            "list_files" => Some(Tool::ListFiles),
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
        }
    }

    pub fn execute(&self, arguments: &str) -> Result<String> {
        match self {
            Tool::FileRead => execute_file_read(arguments),
            Tool::ListFiles => execute_list_files(arguments),
        }
    }
}

fn execute_file_read(arguments: &str) -> Result<String> {
    let args: Value = serde_json::from_str(arguments)?;
    let path = args["path"]
        .as_str()
        .ok_or_else(|| anyhow::anyhow!("Missing required parameter: path"))?;

    std::fs::read_to_string(path)
        .map_err(|e| anyhow::anyhow!("Failed to read file '{}': {}", path, e))
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

    #[test]
    fn file_read_success() {
        let mut tmp = tempfile::NamedTempFile::new().unwrap();
        write!(tmp, "hello world").unwrap();
        let path = tmp.path().to_str().unwrap();

        let args = json!({"path": path}).to_string();
        let result = Tool::FileRead.execute(&args).unwrap();
        assert_eq!(result, "hello world");
    }

    #[test]
    fn file_read_not_found() {
        let args = json!({"path": "/nonexistent/file.txt"}).to_string();
        let result = Tool::FileRead.execute(&args);
        assert!(result.is_err());
        assert!(result.unwrap_err().to_string().contains("Failed to read file"));
    }

    #[test]
    fn file_read_missing_path_param() {
        let args = json!({}).to_string();
        let result = Tool::FileRead.execute(&args);
        assert!(result.is_err());
        assert!(result.unwrap_err().to_string().contains("Missing required parameter"));
    }

    #[test]
    fn from_name_known() {
        assert!(Tool::from_name("file_read").is_some());
        assert!(Tool::from_name("list_files").is_some());
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

    #[test]
    fn list_files_populated_dir() {
        let dir = tempfile::tempdir().unwrap();
        std::fs::write(dir.path().join("a.txt"), "").unwrap();
        std::fs::write(dir.path().join("b.rs"), "").unwrap();
        std::fs::create_dir(dir.path().join("subdir")).unwrap();

        let args = json!({"path": dir.path().to_str().unwrap()}).to_string();
        let result = Tool::ListFiles.execute(&args).unwrap();

        assert!(result.contains("a.txt"));
        assert!(result.contains("b.rs"));
        assert!(result.contains("subdir/"));
    }

    #[test]
    fn list_files_empty_dir() {
        let dir = tempfile::tempdir().unwrap();
        let args = json!({"path": dir.path().to_str().unwrap()}).to_string();
        let result = Tool::ListFiles.execute(&args).unwrap();
        assert_eq!(result, "");
    }

    #[test]
    fn list_files_nonexistent() {
        let args = json!({"path": "/nonexistent/dir"}).to_string();
        let result = Tool::ListFiles.execute(&args);
        assert!(result.is_err());
        assert!(result.unwrap_err().to_string().contains("Failed to list directory"));
    }
}
