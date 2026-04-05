use std::path::{Path, PathBuf};

use anyhow::{Result, bail};

// Clipping thresholds
const DISPLAY_MAX_CHARS: usize = 1_000;
const MODEL_MAX_CHARS: usize = 30_000;

/// Detect the project root by walking up from `start` looking for marker files.
/// Falls back to `start` if no markers are found.
pub fn detect_project_root(start: &Path) -> PathBuf {
    const MARKERS: &[&str] = &[
        ".git",
        "Cargo.toml",
        "package.json",
        "go.mod",
        "pyproject.toml",
    ];

    let mut dir = start.to_path_buf();
    loop {
        for marker in MARKERS {
            if dir.join(marker).exists() {
                return dir;
            }
        }
        if !dir.pop() {
            return start.to_path_buf();
        }
    }
}

/// Validate that `target` resolves to a path within `root` after symlink resolution.
/// Used for write operations to prevent escaping the project root.
pub fn validate_path(target: &Path, root: &Path) -> Result<PathBuf> {
    // Resolve the root to its canonical form
    let canonical_root = root.canonicalize()
        .map_err(|e| anyhow::anyhow!("Failed to resolve project root '{}': {}", root.display(), e))?;

    // If the target doesn't exist yet, resolve as much of the path as possible
    // by canonicalizing the parent directory
    let resolved = if target.exists() {
        target.canonicalize()
            .map_err(|e| anyhow::anyhow!("Failed to resolve path '{}': {}", target.display(), e))?
    } else {
        // Resolve the parent, then append the filename
        let parent = target.parent()
            .ok_or_else(|| anyhow::anyhow!("Invalid path: {}", target.display()))?;

        let canonical_parent = if parent.as_os_str().is_empty() {
            // Relative path with no parent component — use current dir
            std::env::current_dir()?
        } else if parent.exists() {
            parent.canonicalize()
                .map_err(|e| anyhow::anyhow!("Failed to resolve parent '{}': {}", parent.display(), e))?
        } else {
            bail!("Parent directory does not exist: {}", parent.display());
        };

        let filename = target.file_name()
            .ok_or_else(|| anyhow::anyhow!("Invalid path: {}", target.display()))?;

        canonical_parent.join(filename)
    };

    if !resolved.starts_with(&canonical_root) {
        bail!(
            "Path '{}' is outside the project root '{}'",
            resolved.display(),
            canonical_root.display()
        );
    }

    Ok(resolved)
}

/// Clip output for terminal display (shorter threshold).
pub fn clip_for_display(output: &str) -> String {
    clip(output, DISPLAY_MAX_CHARS)
}

/// Clip output for model context insertion (longer threshold).
pub fn clip_for_model(output: &str) -> String {
    clip(output, MODEL_MAX_CHARS)
}

fn clip(output: &str, max: usize) -> String {
    if output.len() <= max {
        output.to_string()
    } else {
        let truncated = &output[..max];
        let remaining = output.len() - max;
        format!("{truncated}\n\n... ({remaining} characters truncated)")
    }
}

/// Check if file contents appear to be binary (contain null bytes in the first 8KB).
pub fn is_binary(data: &[u8]) -> bool {
    let check_len = data.len().min(8192);
    data[..check_len].contains(&0)
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::fs;
    use tempfile::tempdir;

    // --- detect_project_root ---

    #[test]
    fn detect_root_git_repo() {
        let dir = tempdir().unwrap();
        fs::create_dir(dir.path().join(".git")).unwrap();
        let sub = dir.path().join("src");
        fs::create_dir(&sub).unwrap();

        let root = detect_project_root(&sub);
        assert_eq!(root, dir.path());
    }

    #[test]
    fn detect_root_cargo_project() {
        let dir = tempdir().unwrap();
        fs::write(dir.path().join("Cargo.toml"), "[package]").unwrap();

        let root = detect_project_root(dir.path());
        assert_eq!(root, dir.path());
    }

    #[test]
    fn detect_root_no_markers_falls_back() {
        let dir = tempdir().unwrap();
        let root = detect_project_root(dir.path());
        assert_eq!(root, dir.path());
    }

    // --- validate_path ---

    #[test]
    fn validate_path_within_root() {
        let dir = tempdir().unwrap();
        let file = dir.path().join("test.txt");
        fs::write(&file, "").unwrap();

        let result = validate_path(&file, dir.path());
        assert!(result.is_ok());
    }

    #[test]
    fn validate_path_outside_root() {
        let dir = tempdir().unwrap();
        let other = tempdir().unwrap();
        let file = other.path().join("escape.txt");
        fs::write(&file, "").unwrap();

        let result = validate_path(&file, dir.path());
        assert!(result.is_err());
        assert!(result.unwrap_err().to_string().contains("outside the project root"));
    }

    #[test]
    fn validate_path_relative_escape() {
        let dir = tempdir().unwrap();
        let escape = dir.path().join("../escape.txt");

        // This should either fail or resolve outside the root
        let result = validate_path(&escape, dir.path());
        assert!(result.is_err());
    }

    #[cfg(unix)]
    #[test]
    fn validate_path_symlink_escape() {
        let dir = tempdir().unwrap();
        let outside = tempdir().unwrap();
        let target_file = outside.path().join("secret.txt");
        fs::write(&target_file, "secret").unwrap();

        // Create a symlink inside the project pointing outside
        let link = dir.path().join("sneaky_link");
        std::os::unix::fs::symlink(&target_file, &link).unwrap();

        let result = validate_path(&link, dir.path());
        assert!(result.is_err());
        assert!(result.unwrap_err().to_string().contains("outside the project root"));
    }

    #[test]
    fn validate_path_new_file_in_root() {
        let dir = tempdir().unwrap();
        let new_file = dir.path().join("new.txt");
        // File doesn't exist yet — should still validate if parent is in root
        let result = validate_path(&new_file, dir.path());
        assert!(result.is_ok());
    }

    // --- clip_output ---

    #[test]
    fn clip_under_threshold() {
        let short = "hello world";
        assert_eq!(clip_for_display(short), short);
        assert_eq!(clip_for_model(short), short);
    }

    #[test]
    fn clip_over_display_threshold() {
        let long = "x".repeat(DISPLAY_MAX_CHARS + 500);
        let clipped = clip_for_display(&long);
        assert!(clipped.len() < long.len());
        assert!(clipped.contains("truncated"));
        assert!(clipped.contains("500"));
    }

    #[test]
    fn clip_over_model_threshold() {
        let long = "x".repeat(MODEL_MAX_CHARS + 1000);
        let clipped = clip_for_model(&long);
        assert!(clipped.len() < long.len());
        assert!(clipped.contains("truncated"));
    }

    #[test]
    fn clip_at_exact_boundary() {
        let exact = "x".repeat(DISPLAY_MAX_CHARS);
        assert_eq!(clip_for_display(&exact), exact);
    }

    // --- is_binary ---

    #[test]
    fn binary_detection_text() {
        assert!(!is_binary(b"Hello, world!\nThis is text."));
    }

    #[test]
    fn binary_detection_binary() {
        let mut data = vec![0u8; 100];
        data[50] = 0; // null byte
        assert!(is_binary(&data));
    }

    #[test]
    fn binary_detection_empty() {
        assert!(!is_binary(b""));
    }
}
