use std::io::{Write, stdout};

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
    ss: SyntaxSet,
    ts: ThemeSet,
}

impl TerminalRenderer {
    pub fn new() -> Self {
        Self {
            first_token: true,
            ss: SyntaxSet::load_defaults_newlines(),
            ts: ThemeSet::load_defaults(),
        }
    }
}

impl Renderer for TerminalRenderer {
    fn on_token(&mut self, token: String) {
        if self.first_token && !token.trim().is_empty() {
            self.first_token = false;
        }
        print!("{token}");
        let _ = stdout().flush();
    }

    fn on_tool_call(&mut self, name: &str, args: &str) {
        let summary = summarize_tool_call(name, args);
        println!("  {} {}", format!("▸ {name}").cyan(), summary.dimmed());
        let _ = stdout().flush();
    }

    fn on_tool_result(&mut self, name: &str, output: &str) {
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
        if let Some(rc) = plaintext {
            println!("{rc}");
        }
        if encrypted.is_some() {
            let token_count = reasoning_tokens.unwrap_or(0);
            println!("{}", format!("[thinking... {token_count} tokens]").dimmed());
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

fn summarize_tool_call(name: &str, args: &str) -> String {
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
