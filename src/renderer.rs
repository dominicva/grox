use std::io::{Write, stdout};
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};

use colored::Colorize;
use syntect::easy::HighlightLines;
use syntect::highlighting::ThemeSet;
use syntect::parsing::SyntaxSet;
use syntect::util::as_24_bit_terminal_escaped;

/// Display callbacks for the agent loop.
///
/// Encapsulates all rendering concerns: token streaming, tool call summaries,
/// tool result formatting, and reasoning display. Owns rendering state like
/// the first-token flag and syntax highlighting context.
pub trait Renderer: Send {
    /// Called for each streamed token from the model.
    fn on_token(&mut self, token: String);

    /// Called when the model invokes a tool (before execution).
    fn on_tool_call(&mut self, name: &str, args: &str);

    /// Called after a tool finishes executing, with its output.
    fn on_tool_result(&mut self, name: &str, output: &str);

    /// Called when authorization produces a warning (e.g. destructive command).
    /// Phase 5 will populate these; until then the warning is always `None`
    /// and this method is a no-op pass-through.
    fn on_auth_warning(&mut self, _warning: &str) {}

    /// Called when the model emits reasoning content.
    ///
    /// `plaintext` is set for models that return readable reasoning (grok-3-mini).
    /// `encrypted` is set for models that return opaque reasoning blobs (grok-4).
    /// `reasoning_tokens` is the token count when available.
    fn on_reasoning(
        &mut self,
        plaintext: Option<&str>,
        encrypted: Option<&str>,
        reasoning_tokens: Option<u64>,
    );
}

/// Terminal renderer that writes to stdout with ANSI colors and syntax highlighting.
pub struct TerminalRenderer {
    first_token: bool,
    thinking_expanded: Arc<AtomicBool>,
    line_buffer: String,
    ss: SyntaxSet,
    ts: ThemeSet,
}

impl TerminalRenderer {
    pub fn new() -> Self {
        Self {
            first_token: true,
            thinking_expanded: Arc::new(AtomicBool::new(true)),
            line_buffer: String::new(),
            ss: SyntaxSet::load_defaults_newlines(),
            ts: ThemeSet::load_defaults(),
        }
    }

    /// Reset per-turn state. Called before each agent turn so ephemeral flags
    /// like first_token reset while session-level state (thinking toggle, etc.)
    /// persists across turns.
    pub fn begin_turn(&mut self) {
        self.first_token = true;
        self.line_buffer.clear();
    }

    /// Flush any remaining content in the line buffer to stdout.
    /// Called after a turn completes to emit the final partial line.
    pub fn flush_line_buffer(&mut self) {
        if !self.line_buffer.is_empty() {
            let formatted = format_markdown_line(&self.line_buffer);
            print!("{formatted}");
            let _ = stdout().flush();
            self.line_buffer.clear();
        }
    }

    /// Get a shared handle to the thinking-expanded flag.
    /// Used by the Ctrl+T keybinding handler.
    pub fn thinking_expanded_handle(&self) -> Arc<AtomicBool> {
        Arc::clone(&self.thinking_expanded)
    }

    /// Toggle thinking display between expanded and collapsed.
    /// Returns the new state (true = expanded).
    pub fn toggle_thinking_display(&self) -> bool {
        let prev = self.thinking_expanded.load(Ordering::Relaxed);
        self.thinking_expanded.store(!prev, Ordering::Relaxed);
        !prev
    }
}

impl Renderer for TerminalRenderer {
    fn on_token(&mut self, token: String) {
        if self.first_token && !token.trim().is_empty() {
            self.first_token = false;
        }
        self.line_buffer.push_str(&token);
        // Emit all complete lines, keep the trailing partial line buffered.
        while let Some(nl_pos) = self.line_buffer.find('\n') {
            let line = self.line_buffer[..nl_pos].to_string();
            self.line_buffer = self.line_buffer[nl_pos + 1..].to_string();
            let formatted = format_markdown_line(&line);
            println!("{formatted}");
            let _ = stdout().flush();
        }
    }

    fn on_tool_call(&mut self, name: &str, args: &str) {
        self.flush_line_buffer();
        let summary = summarize_tool_call(name, args);
        println!("  {} {}", format!("▸ {name}").cyan(), summary.dimmed());
        let _ = stdout().flush();
    }

    fn on_tool_result(&mut self, name: &str, output: &str) {
        self.flush_line_buffer();
        const MAX_DISPLAY_LINES: usize = 20;

        let is_error = output.starts_with("Error:")
            || output.starts_with("File '")
            || output.starts_with("Permission denied");
        if is_error {
            let msg = output.lines().next().unwrap_or(output);
            println!("{}", format!("  {} {}", "✗".red(), msg).dimmed());
        } else if name == "shell_exec" {
            if output.is_empty() || output == "(no output)" {
                println!("{}", format!("  {} (no output)", "✓".green()).dimmed());
            } else {
                let lines: Vec<&str> = output.lines().collect();
                let total = lines.len();
                let show = total.min(MAX_DISPLAY_LINES);
                println!("{}", format!("  {}", "✓".green()).dimmed());
                for line in &lines[..show] {
                    println!("  {}", line.dimmed());
                }
                if total > MAX_DISPLAY_LINES {
                    println!(
                        "  {}",
                        format!("... ({} more lines)", total - MAX_DISPLAY_LINES).dimmed()
                    );
                }
            }
        } else if name == "file_edit" {
            println!("{}", format!("  {}", "✓".green()).dimmed());
            println!();
            self.format_edit_context(output);
            println!();
        } else if output.is_empty() {
            println!("{}", format!("  {} (empty)", "✓".green()).dimmed());
        } else {
            let lines: Vec<&str> = output.lines().collect();
            let summary = if lines.len() > 5 {
                format!("  {} ({} lines)", "✓".green(), lines.len())
            } else {
                format!("  {} ({} bytes)", "✓".green(), output.len())
            };
            println!("{}", summary.dimmed());
        }
    }

    fn on_reasoning(
        &mut self,
        plaintext: Option<&str>,
        encrypted: Option<&str>,
        reasoning_tokens: Option<u64>,
    ) {
        self.flush_line_buffer();
        let expanded = self.thinking_expanded.load(Ordering::Relaxed);

        if let Some(rc) = plaintext {
            let token_count = reasoning_tokens.unwrap_or(0);
            if expanded {
                println!("\n{}", "Thinking".bold().dimmed());
                for line in rc.lines() {
                    println!("{}", format!("  {line}").dimmed());
                }
                println!();
            } else {
                println!(
                    "{}",
                    format!("Thinking [{token_count} tokens]").dimmed()
                );
            }
        }
        if encrypted.is_some() {
            let token_count = reasoning_tokens.unwrap_or(0);
            println!(
                "{}",
                format!("[thinking... {token_count} tokens]").dimmed()
            );
        }
    }
}

impl TerminalRenderer {
    /// Format file_edit output with syntax highlighting and spacing.
    ///
    /// Input format: "Edited path/to/file.ext\n\n   1 | code\n   2 | code\n..."
    fn format_edit_context(&self, output: &str) {
        let mut lines = output.lines();

        let header = lines.next().unwrap_or("");
        let filename = header.strip_prefix("Edited ").unwrap_or("");
        println!("  {}", header.dimmed());

        // Skip the blank line separator
        let _ = lines.next();

        let context_lines: Vec<&str> = lines.collect();
        if context_lines.is_empty() {
            return;
        }

        let theme = &self.ts.themes["base16-ocean.dark"];
        let syntax = self
            .ss
            .find_syntax_for_file(filename)
            .ok()
            .flatten()
            .unwrap_or_else(|| self.ss.find_syntax_plain_text());

        let mut highlighter = HighlightLines::new(syntax, theme);

        for line in &context_lines {
            if let Some(pipe_pos) = line.find(" | ") {
                let gutter = &line[..pipe_pos + 3];
                let code = &line[pipe_pos + 3..];

                let code_with_nl = format!("{code}\n");
                let regions = highlighter
                    .highlight_line(&code_with_nl, &self.ss)
                    .unwrap_or_default();
                let highlighted = as_24_bit_terminal_escaped(&regions, false);

                print!("  {}", gutter.dimmed());
                print!("{highlighted}\x1b[0m");
            } else {
                println!("  {}", line.dimmed());
            }
        }
    }
}

/// A renderer that records all calls for testing.
#[cfg(test)]
pub struct RecordingRenderer {
    pub tokens: Vec<String>,
    pub tool_calls: Vec<(String, String)>,
    pub tool_results: Vec<(String, String)>,
    pub reasoning_calls: Vec<(Option<String>, Option<String>, Option<u64>)>,
    pub auth_warnings: Vec<String>,
}

#[cfg(test)]
impl RecordingRenderer {
    pub fn new() -> Self {
        Self {
            tokens: Vec::new(),
            tool_calls: Vec::new(),
            tool_results: Vec::new(),
            reasoning_calls: Vec::new(),
            auth_warnings: Vec::new(),
        }
    }
}

#[cfg(test)]
impl Renderer for RecordingRenderer {
    fn on_token(&mut self, token: String) {
        self.tokens.push(token);
    }
    fn on_tool_call(&mut self, name: &str, args: &str) {
        self.tool_calls.push((name.to_string(), args.to_string()));
    }
    fn on_tool_result(&mut self, name: &str, output: &str) {
        self.tool_results.push((name.to_string(), output.to_string()));
    }
    fn on_auth_warning(&mut self, warning: &str) {
        self.auth_warnings.push(warning.to_string());
    }
    fn on_reasoning(
        &mut self,
        plaintext: Option<&str>,
        encrypted: Option<&str>,
        reasoning_tokens: Option<u64>,
    ) {
        self.reasoning_calls.push((
            plaintext.map(|s| s.to_string()),
            encrypted.map(|s| s.to_string()),
            reasoning_tokens,
        ));
    }
}

/// Format a single line of markdown for terminal display.
///
/// Applies limited markdown formatting:
/// - Headings: `#`, `##`, `###` at line start → bold text (without the `#` markers)
/// - Bold: `**text**` → ANSI bold
/// - Everything else passes through unstyled.
pub fn format_markdown_line(line: &str) -> String {
    let trimmed = line.trim_start();

    // Heading: # at start of line → bold (strip the # markers)
    if let Some(rest) = trimmed.strip_prefix("### ") {
        return format!("\x1b[1m{rest}\x1b[0m");
    }
    if let Some(rest) = trimmed.strip_prefix("## ") {
        return format!("\x1b[1m{rest}\x1b[0m");
    }
    if let Some(rest) = trimmed.strip_prefix("# ") {
        return format!("\x1b[1m{rest}\x1b[0m");
    }

    // Bold: **text** → ANSI bold
    if !line.contains("**") {
        return line.to_string();
    }

    let mut result = String::with_capacity(line.len() + 20);
    let mut chars = line.chars().peekable();
    while let Some(ch) = chars.next() {
        if ch == '*' && chars.peek() == Some(&'*') {
            chars.next(); // consume second *
            // Collect bold content until closing **
            let mut bold = String::new();
            let mut closed = false;
            while let Some(inner) = chars.next() {
                if inner == '*' && chars.peek() == Some(&'*') {
                    chars.next(); // consume closing *
                    closed = true;
                    break;
                }
                bold.push(inner);
            }
            if closed && !bold.is_empty() {
                result.push_str("\x1b[1m");
                result.push_str(&bold);
                result.push_str("\x1b[0m");
            } else {
                // Unclosed ** — pass through raw
                result.push_str("**");
                result.push_str(&bold);
            }
        } else {
            result.push(ch);
        }
    }
    result
}

/// Summarize a tool call into a compact display string.
pub fn summarize_tool_call(name: &str, args: &str) -> String {
    let parsed: serde_json::Value = serde_json::from_str(args).unwrap_or_default();
    match name {
        "file_read" | "list_files" => parsed["path"].as_str().unwrap_or("?").to_string(),
        "grep" => {
            let pattern = parsed["pattern"].as_str().unwrap_or("?");
            let path = parsed["path"].as_str().unwrap_or(".");
            format!("{pattern} in {path}")
        }
        "file_write" => {
            let path = parsed["path"].as_str().unwrap_or("?");
            let len = parsed["content"].as_str().map(|c| c.len()).unwrap_or(0);
            format!("{path} ({len} bytes)")
        }
        "file_edit" => {
            let path = parsed["path"].as_str().unwrap_or("?");
            path.to_string()
        }
        "shell_exec" => {
            let cmd = parsed["command"].as_str().unwrap_or("?");
            let truncated: String = cmd.chars().take(80).collect();
            if truncated.len() < cmd.len() {
                format!("{truncated}…")
            } else {
                truncated
            }
        }
        _ => args.to_string(),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // --- Renderer trait compliance ---

    #[test]
    fn recording_renderer_captures_tokens() {
        let mut r = RecordingRenderer::new();
        r.on_token("Hello".into());
        r.on_token(" world".into());
        assert_eq!(r.tokens, vec!["Hello", " world"]);
    }

    #[test]
    fn recording_renderer_captures_tool_calls() {
        let mut r = RecordingRenderer::new();
        r.on_tool_call("file_read", r#"{"path":"src/main.rs"}"#);
        assert_eq!(r.tool_calls.len(), 1);
        assert_eq!(r.tool_calls[0].0, "file_read");
    }

    #[test]
    fn recording_renderer_captures_tool_results() {
        let mut r = RecordingRenderer::new();
        r.on_tool_result("file_read", "fn main() {}");
        assert_eq!(r.tool_results.len(), 1);
        assert_eq!(r.tool_results[0].0, "file_read");
        assert_eq!(r.tool_results[0].1, "fn main() {}");
    }

    #[test]
    fn recording_renderer_captures_reasoning() {
        let mut r = RecordingRenderer::new();
        r.on_reasoning(Some("thinking"), None, Some(42));
        r.on_reasoning(None, Some("encrypted_blob"), Some(100));

        assert_eq!(r.reasoning_calls.len(), 2);
        assert_eq!(r.reasoning_calls[0].0.as_deref(), Some("thinking"));
        assert_eq!(r.reasoning_calls[0].1, None);
        assert_eq!(r.reasoning_calls[0].2, Some(42));
        assert_eq!(r.reasoning_calls[1].0, None);
        assert_eq!(r.reasoning_calls[1].1.as_deref(), Some("encrypted_blob"));
    }

    #[test]
    fn recording_renderer_captures_auth_warnings() {
        let mut r = RecordingRenderer::new();
        r.on_auth_warning("destructive command detected");
        assert_eq!(r.auth_warnings, vec!["destructive command detected"]);
    }

    #[test]
    fn recording_renderer_starts_empty() {
        let r = RecordingRenderer::new();
        assert!(r.tokens.is_empty());
        assert!(r.tool_calls.is_empty());
        assert!(r.tool_results.is_empty());
        assert!(r.reasoning_calls.is_empty());
        assert!(r.auth_warnings.is_empty());
    }

    #[test]
    fn default_on_auth_warning_is_noop() {
        // Verify the default trait method compiles and doesn't panic.
        // TerminalRenderer inherits the default no-op; Phase 5 will override it.
        struct MinimalRenderer;
        impl Renderer for MinimalRenderer {
            fn on_token(&mut self, _: String) {}
            fn on_tool_call(&mut self, _: &str, _: &str) {}
            fn on_tool_result(&mut self, _: &str, _: &str) {}
            fn on_reasoning(&mut self, _: Option<&str>, _: Option<&str>, _: Option<u64>) {}
            // on_auth_warning intentionally NOT overridden — uses default no-op
        }
        let mut r = MinimalRenderer;
        r.on_auth_warning("should not panic");
    }

    // --- summarize_tool_call ---

    // --- Thinking display state ---

    #[test]
    fn thinking_expanded_by_default() {
        let r = TerminalRenderer::new();
        assert!(r.thinking_expanded.load(Ordering::Relaxed));
    }

    #[test]
    fn toggle_thinking_display_cycles() {
        let r = TerminalRenderer::new();
        assert!(r.thinking_expanded.load(Ordering::Relaxed));
        let new_state = r.toggle_thinking_display();
        assert!(!new_state);
        assert!(!r.thinking_expanded.load(Ordering::Relaxed));
        let new_state = r.toggle_thinking_display();
        assert!(new_state);
        assert!(r.thinking_expanded.load(Ordering::Relaxed));
    }

    #[test]
    fn thinking_expanded_handle_shares_state() {
        let r = TerminalRenderer::new();
        let handle = r.thinking_expanded_handle();
        assert!(handle.load(Ordering::Relaxed));
        r.toggle_thinking_display();
        assert!(!handle.load(Ordering::Relaxed));
    }

    #[test]
    fn begin_turn_preserves_thinking_state() {
        let mut r = TerminalRenderer::new();
        r.toggle_thinking_display(); // collapsed
        r.begin_turn();
        assert!(!r.thinking_expanded.load(Ordering::Relaxed));
    }

    // --- Markdown formatting ---

    #[test]
    fn markdown_heading_h1() {
        assert_eq!(format_markdown_line("# Hello"), "\x1b[1mHello\x1b[0m");
    }

    #[test]
    fn markdown_heading_h2() {
        assert_eq!(format_markdown_line("## World"), "\x1b[1mWorld\x1b[0m");
    }

    #[test]
    fn markdown_heading_h3() {
        assert_eq!(format_markdown_line("### Details"), "\x1b[1mDetails\x1b[0m");
    }

    #[test]
    fn markdown_bold() {
        assert_eq!(
            format_markdown_line("this is **bold** text"),
            "this is \x1b[1mbold\x1b[0m text"
        );
    }

    #[test]
    fn markdown_multiple_bold() {
        assert_eq!(
            format_markdown_line("**a** and **b**"),
            "\x1b[1ma\x1b[0m and \x1b[1mb\x1b[0m"
        );
    }

    #[test]
    fn markdown_unclosed_bold_passthrough() {
        assert_eq!(format_markdown_line("this is **unclosed"), "this is **unclosed");
    }

    #[test]
    fn markdown_plain_text_passthrough() {
        assert_eq!(format_markdown_line("just plain text"), "just plain text");
    }

    #[test]
    fn markdown_empty_line() {
        assert_eq!(format_markdown_line(""), "");
    }

    #[test]
    fn markdown_heading_with_leading_whitespace() {
        assert_eq!(format_markdown_line("  # Indented"), "\x1b[1mIndented\x1b[0m");
    }

    #[test]
    fn markdown_no_bold_without_stars() {
        let line = "no formatting here at all";
        assert_eq!(format_markdown_line(line), line);
    }

    #[test]
    fn markdown_empty_bold_passthrough() {
        // **** is ** then empty bold content — the empty bold is dropped,
        // and the remaining ** passes through as unclosed
        assert_eq!(format_markdown_line("****"), "**");
    }

    // --- summarize_tool_call ---

    #[test]
    fn summarize_file_read() {
        assert_eq!(
            summarize_tool_call("file_read", r#"{"path":"src/main.rs"}"#),
            "src/main.rs"
        );
    }

    #[test]
    fn summarize_grep() {
        assert_eq!(
            summarize_tool_call("grep", r#"{"pattern":"TODO","path":"src"}"#),
            "TODO in src"
        );
    }

    #[test]
    fn summarize_file_write() {
        assert_eq!(
            summarize_tool_call("file_write", r#"{"path":"x.rs","content":"hello"}"#),
            "x.rs (5 bytes)"
        );
    }

    #[test]
    fn summarize_shell_exec_truncation() {
        let long_cmd = "a".repeat(100);
        let args = format!(r#"{{"command":"{long_cmd}"}}"#);
        let result = summarize_tool_call("shell_exec", &args);
        assert!(result.ends_with('…'));
        assert!(result.len() <= 84); // 80 chars + "…"
    }
}
