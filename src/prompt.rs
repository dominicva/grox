use std::path::Path;

/// Build the full system prompt for the Grok model.
/// Combines the core instructions with optional GROX.md custom instructions.
pub fn build_system_prompt(project_root: &Path, grox_md: Option<&str>) -> String {
    let mut sections = Vec::new();

    sections.push(identity_section(project_root));
    sections.push(output_style_section());
    sections.push(working_with_code_section());
    sections.push(taking_action_section());
    sections.push(tools_section());

    if let Some(custom) = grox_md {
        sections.push(project_instructions_section(custom));
    }

    sections.join("\n\n")
}

fn identity_section(project_root: &Path) -> String {
    format!(
        "You are Grox, a coding agent powered by Grok. You help developers understand and work with their codebase.\n\n\
         Project root: {}",
        project_root.display()
    )
}

fn output_style_section() -> String {
    "# Output style

Lead with the answer or action, not the reasoning. Do not narrate what you are about to do — just do it. Do not explain your tool usage or repeat tool output the user already saw. Do not thank the user or add preamble.

Keep text output brief and direct. If you can say it in one sentence, don't use three. Skip filler words, unnecessary transitions, and restating what the user said.

Focus your text on:
- Findings and answers
- Decisions that need the user's input
- Errors or blockers that change the plan

This does not apply to code — write code as clearly as needed."
        .to_string()
}

fn working_with_code_section() -> String {
    "# Working with code

- Read before modifying. Before changing a file, read it and its surrounding context — how is it used, what patterns does it follow, what else lives nearby.
- Do not add features, refactoring, or improvements beyond what was asked.
- Do not create files unless necessary. Prefer editing existing files.
- Do not add comments, docstrings, or type annotations to code you didn't change.
- Be careful not to introduce security vulnerabilities. Prioritize safe, correct code.
- Verify your work actually works before reporting it complete."
        .to_string()
}

fn taking_action_section() -> String {
    "# Taking action

Before making changes, build understanding. Read the files you plan to modify, check how similar things are done nearby, and look at the structure around your target. A few extra reads upfront prevent wrong turns that waste far more time. Reads are free — don't guess at patterns, conventions, or file locations when you can look.

Writes and shell commands have consequences. Think before executing. For destructive operations — deleting files, force-pushing, dropping tables, rm -rf — pause and consider whether this is really the right action. If you encounter unexpected state like unfamiliar files or configuration, investigate before overwriting. It may be the user's in-progress work.

The user's permission mode governs what you can do without asking. Respect it: if a tool call is denied, adjust your approach rather than retrying the same action."
        .to_string()
}

fn tools_section() -> String {
    "# Tools

- **grep**: Search file contents by regex pattern. Use this to find definitions, references, and patterns across the codebase.
- **list_files**: Explore directory structure. Use this to understand project layout before diving into specific files.
- **file_read**: Read specific files. Gather context before answering questions about code.
- **file_write**: Create or overwrite files. Parent directories are created automatically.
- **file_edit**: Edit a file by replacing a single occurrence of a string. Include enough context in old_string for a unique match. Prefer this over file_write for surgical edits.
- **shell_exec**: Run shell commands in the project root. Use for builds, tests, git operations, and other tasks that need a shell."
        .to_string()
}

fn project_instructions_section(content: &str) -> String {
    format!("# Project instructions (GROX.md)\n\n{content}")
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::path::Path;

    #[test]
    fn prompt_contains_all_sections() {
        let prompt = build_system_prompt(Path::new("/test/project"), None);
        assert!(prompt.contains("You are Grox"));
        assert!(prompt.contains("# Output style"));
        assert!(prompt.contains("# Working with code"));
        assert!(prompt.contains("# Taking action"));
        assert!(prompt.contains("# Tools"));
    }

    #[test]
    fn prompt_includes_project_root() {
        let prompt = build_system_prompt(Path::new("/my/project"), None);
        assert!(prompt.contains("Project root: /my/project"));
    }

    #[test]
    fn prompt_excludes_grox_md_when_absent() {
        let prompt = build_system_prompt(Path::new("/test"), None);
        assert!(!prompt.contains("GROX.md"));
    }

    #[test]
    fn prompt_includes_grox_md_when_present() {
        let prompt = build_system_prompt(Path::new("/test"), Some("Always use tabs."));
        assert!(prompt.contains("# Project instructions (GROX.md)"));
        assert!(prompt.contains("Always use tabs."));
    }

    #[test]
    fn prompt_sections_are_separated_by_blank_lines() {
        let prompt = build_system_prompt(Path::new("/test"), None);
        // Each section should be separated by \n\n
        assert!(prompt.contains("# Output style\n\n"));
        assert!(prompt.contains("# Working with code\n\n"));
        assert!(prompt.contains("# Taking action\n\n"));
    }

    #[test]
    fn prompt_mentions_all_tools() {
        let prompt = build_system_prompt(Path::new("/test"), None);
        assert!(prompt.contains("grep"));
        assert!(prompt.contains("list_files"));
        assert!(prompt.contains("file_read"));
        assert!(prompt.contains("file_write"));
        assert!(prompt.contains("file_edit"));
        assert!(prompt.contains("shell_exec"));
    }

    #[test]
    fn identity_is_first_section() {
        let prompt = build_system_prompt(Path::new("/test"), None);
        assert!(prompt.starts_with("You are Grox"));
    }

    #[test]
    fn prompt_emphasizes_reading_before_writing() {
        let prompt = build_system_prompt(Path::new("/test"), None);
        assert!(prompt.contains("Before making changes, build understanding"));
    }

    #[test]
    fn grox_md_is_last_section() {
        let prompt = build_system_prompt(Path::new("/test"), Some("Custom rules."));
        assert!(prompt.ends_with("Custom rules."));
    }
}
