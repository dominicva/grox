use anyhow::Result;
use serde_json::{Value, json};

#[derive(Debug, Clone)]
pub enum Tool {
    FileRead,
}

#[derive(Debug, Clone)]
pub struct ToolCall {
    pub call_id: String,
    pub name: String,
    pub arguments: String,
}

impl Tool {
    pub fn all() -> Vec<Tool> {
        vec![Tool::FileRead]
    }

    pub fn definitions() -> Vec<Value> {
        Tool::all().iter().map(|t| t.definition()).collect()
    }

    pub fn from_name(name: &str) -> Option<Tool> {
        match name {
            "file_read" => Some(Tool::FileRead),
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
        }
    }

    pub fn execute(&self, arguments: &str) -> Result<String> {
        match self {
            Tool::FileRead => execute_file_read(arguments),
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
    }

    #[test]
    fn from_name_unknown() {
        assert!(Tool::from_name("unknown_tool").is_none());
    }

    #[test]
    fn definition_has_required_fields() {
        let def = Tool::FileRead.definition();
        assert_eq!(def["name"], "file_read");
        assert_eq!(def["type"], "function");
        assert!(def["parameters"]["properties"]["path"].is_object());
    }
}
