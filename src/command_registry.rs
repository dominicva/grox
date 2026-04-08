use std::borrow::Cow;
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};

use rustyline::Cmd;
use rustyline::completion::{Completer, Pair};
use rustyline::highlight::Highlighter;
use rustyline::hint::{Hint, Hinter};
use rustyline::validate::Validator;
use rustyline::{Context, Helper};

use crate::file_index::FileIndex;

/// Identifies which slash command was matched.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Command {
    Quit,
    Model,
    Think,
    Status,
    Undo,
    Sessions,
    Resume,
    Compact,
}

/// How a command accepts arguments.
#[derive(Debug, Clone, Copy)]
pub enum ArgSpec {
    /// No arguments.
    None,
    /// Optional argument (shown as `[<label>]` in completion).
    Optional(&'static str),
}

/// A single slash-command definition.
#[derive(Debug)]
pub struct CommandSpec {
    pub name: &'static str,
    pub aliases: &'static [&'static str],
    pub description: &'static str,
    pub arg_spec: ArgSpec,
    pub command: Command,
}

/// All commands, sorted alphabetically by name.
static COMMANDS: [CommandSpec; 8] = [
    CommandSpec {
        name: "compact",
        aliases: &[],
        description: "compact conversation history",
        arg_spec: ArgSpec::None,
        command: Command::Compact,
    },
    CommandSpec {
        name: "model",
        aliases: &[],
        description: "switch or browse models",
        arg_spec: ArgSpec::Optional("model-name"),
        command: Command::Model,
    },
    CommandSpec {
        name: "quit",
        aliases: &["exit"],
        description: "exit grox",
        arg_spec: ArgSpec::None,
        command: Command::Quit,
    },
    CommandSpec {
        name: "resume",
        aliases: &[],
        description: "resume a previous session",
        arg_spec: ArgSpec::Optional("session-id"),
        command: Command::Resume,
    },
    CommandSpec {
        name: "sessions",
        aliases: &[],
        description: "list recent sessions",
        arg_spec: ArgSpec::None,
        command: Command::Sessions,
    },
    CommandSpec {
        name: "status",
        aliases: &[],
        description: "show session status",
        arg_spec: ArgSpec::None,
        command: Command::Status,
    },
    CommandSpec {
        name: "think",
        aliases: &[],
        description: "cycle reasoning effort, or toggle display",
        arg_spec: ArgSpec::Optional("display"),
        command: Command::Think,
    },
    CommandSpec {
        name: "undo",
        aliases: &[],
        description: "undo last turn",
        arg_spec: ArgSpec::Optional("N --code|--conversation|--both"),
        command: Command::Undo,
    },
];

/// Compile-time command registry. Single source of truth for dispatch and completion.
pub struct CommandRegistry;

impl CommandRegistry {
    /// Look up a command by exact name or alias from slash-command input.
    /// Returns the matched spec and any trailing arguments.
    pub fn find(input: &str) -> Option<(&'static CommandSpec, &str)> {
        let input = input.trim();
        let without_slash = input.strip_prefix('/')?;
        let (cmd_name, args) = match without_slash.split_once(char::is_whitespace) {
            Some((name, rest)) => (name, rest.trim()),
            None => (without_slash, ""),
        };
        let spec = COMMANDS
            .iter()
            .find(|spec| spec.name == cmd_name || spec.aliases.contains(&cmd_name))?;
        // Reject trailing args for commands that accept none — preserves the
        // original exact-match semantics for /quit, /think, /status, etc.
        if matches!(spec.arg_spec, ArgSpec::None) && !args.is_empty() {
            return None;
        }
        Some((spec, args))
    }

    /// Return commands whose name or any alias starts with `prefix`.
    /// Exact matches sort first, then alphabetically by name.
    pub fn prefix_matches(prefix: &str) -> Vec<&'static CommandSpec> {
        let prefix = prefix.strip_prefix('/').unwrap_or(prefix);
        let mut matches: Vec<&CommandSpec> = COMMANDS
            .iter()
            .filter(|spec| {
                spec.name.starts_with(prefix)
                    || spec.aliases.iter().any(|a| a.starts_with(prefix))
            })
            .collect();
        matches.sort_by(|a, b| {
            let a_exact = a.name == prefix || a.aliases.contains(&prefix);
            let b_exact = b.name == prefix || b.aliases.contains(&prefix);
            match (a_exact, b_exact) {
                (true, false) => std::cmp::Ordering::Less,
                (false, true) => std::cmp::Ordering::Greater,
                _ => a.name.cmp(b.name),
            }
        });
        matches
    }

    /// Return all commands, sorted alphabetically.
    pub fn all() -> &'static [CommandSpec] {
        &COMMANDS
    }
}

// ---------------------------------------------------------------------------
// Rustyline helper: slash-command completion and ghost-text hints
// ---------------------------------------------------------------------------

/// Rustyline helper providing slash-command completion, file path completion,
/// and ghost-text hints.
pub struct GroxHelper {
    pub file_index: Option<FileIndex>,
}

impl GroxHelper {
    pub fn new() -> Self {
        Self { file_index: None }
    }

    pub fn with_file_index(file_index: FileIndex) -> Self {
        Self {
            file_index: Some(file_index),
        }
    }
}

impl Helper for GroxHelper {}

impl Completer for GroxHelper {
    type Candidate = Pair;

    fn complete(
        &self,
        line: &str,
        pos: usize,
        _ctx: &Context<'_>,
    ) -> rustyline::Result<(usize, Vec<Pair>)> {
        if pos != line.len() || pos < 1 {
            return Ok((pos, vec![]));
        }

        // --- Slash command completion ---
        if line.starts_with('/') && !line[1..].contains(char::is_whitespace) {
            let prefix = &line[1..];
            let matches: Vec<&CommandSpec> = if prefix.is_empty() {
                CommandRegistry::all().iter().collect()
            } else {
                CommandRegistry::prefix_matches(prefix)
            };

            let pairs = matches
                .into_iter()
                .flat_map(|cmd| {
                    let arg_hint = match cmd.arg_spec {
                        ArgSpec::Optional(label) => format!(" [{label}]"),
                        ArgSpec::None => String::new(),
                    };
                    let mut entries = vec![];
                    if cmd.name.starts_with(prefix) || prefix.is_empty() {
                        entries.push(Pair {
                            display: format!(
                                "/{}{} — {}",
                                cmd.name, arg_hint, cmd.description
                            ),
                            replacement: format!("/{}", cmd.name),
                        });
                    }
                    for alias in cmd.aliases {
                        if alias.starts_with(prefix) {
                            entries.push(Pair {
                                display: format!(
                                    "/{} (alias for /{}){} — {}",
                                    alias, cmd.name, arg_hint, cmd.description
                                ),
                                replacement: format!("/{alias}"),
                            });
                        }
                    }
                    entries
                })
                .collect();

            return Ok((0, pairs));
        }

        // --- File path completion ---
        if let Some(ref file_index) = self.file_index {
            if let Some((token_start, token)) = extract_path_token(line, pos) {
                let completions = file_index.completions(token);
                let pairs = completions
                    .into_iter()
                    .map(|(display, replacement)| Pair {
                        display,
                        replacement,
                    })
                    .collect();
                return Ok((token_start, pairs));
            }
        }

        Ok((pos, vec![]))
    }
}

impl Hinter for GroxHelper {
    type Hint = CommandHint;

    fn hint(&self, line: &str, pos: usize, _ctx: &Context<'_>) -> Option<CommandHint> {
        // Only hint while typing the command name (no space yet), cursor at end.
        if !line.starts_with('/') || pos != line.len() || line.len() < 2 {
            return None;
        }
        if line[1..].contains(char::is_whitespace) {
            return None;
        }

        let prefix = &line[1..];
        let matches = CommandRegistry::prefix_matches(prefix);
        let first = matches.first()?;

        // Find which name matched — canonical name or an alias.
        let matched_name = if first.name.starts_with(prefix) {
            first.name
        } else {
            first.aliases.iter().find(|a| a.starts_with(prefix)).copied()?
        };

        // Don't hint for exact match.
        if matched_name == prefix {
            return None;
        }

        let suffix = &matched_name[prefix.len()..];
        Some(CommandHint {
            completion: suffix.to_string(),
            display: format!("{suffix} — {}", first.description),
        })
    }
}

impl Highlighter for GroxHelper {
    fn highlight_hint<'h>(&self, hint: &'h str) -> Cow<'h, str> {
        // Dim the ghost text.
        Cow::Owned(format!("\x1b[2m{hint}\x1b[0m"))
    }
}

impl Validator for GroxHelper {}

/// Ghost-text hint for a slash command.
pub struct CommandHint {
    completion: String,
    display: String,
}

impl Hint for CommandHint {
    fn display(&self) -> &str {
        &self.display
    }

    fn completion(&self) -> Option<&str> {
        // Right-arrow inserts just the command suffix, not the description.
        Some(&self.completion)
    }
}

// ---------------------------------------------------------------------------
// File path token extraction
// ---------------------------------------------------------------------------

/// Path trigger prefixes that activate file completion.
const PATH_TRIGGERS: &[&str] = &["@", "./", "../", "~/"];

/// Extract the path-like token at the cursor position.
///
/// Returns `(token_start_position, token_text)` if the text at the cursor
/// looks like a file path (starts with `@`, `./`, `../`, `~`, or contains `/`).
fn extract_path_token(line: &str, pos: usize) -> Option<(usize, &str)> {
    let before_cursor = &line[..pos];

    // Find the start of the current token (scan backwards for whitespace)
    let token_start = before_cursor
        .rfind(char::is_whitespace)
        .map(|i| i + 1)
        .unwrap_or(0);
    let token = &before_cursor[token_start..];

    if token.is_empty() {
        return None;
    }

    // Check if the token starts with a path trigger
    for trigger in PATH_TRIGGERS {
        if token.starts_with(trigger) {
            // For @, strip the prefix and adjust start position
            if *trigger == "@" {
                return Some((token_start + 1, &token[1..]));
            }
            return Some((token_start, token));
        }
    }

    // Also trigger on tokens containing `/` (e.g. `src/main`)
    if token.contains('/') {
        return Some((token_start, token));
    }

    None
}

// ---------------------------------------------------------------------------
// Ctrl+T keybinding: toggle thinking display
// ---------------------------------------------------------------------------

/// Rustyline event handler for Ctrl+T that toggles thinking display mode.
pub struct ThinkToggleHandler {
    thinking_expanded: Arc<AtomicBool>,
}

impl ThinkToggleHandler {
    pub fn new(thinking_expanded: Arc<AtomicBool>) -> Self {
        Self { thinking_expanded }
    }
}

impl rustyline::ConditionalEventHandler for ThinkToggleHandler {
    fn handle(
        &self,
        _evt: &rustyline::Event,
        _n: rustyline::RepeatCount,
        _positive: bool,
        _ctx: &rustyline::EventContext,
    ) -> Option<Cmd> {
        let prev = self.thinking_expanded.load(Ordering::Relaxed);
        self.thinking_expanded.store(!prev, Ordering::Relaxed);
        let label = if !prev { "expanded" } else { "collapsed" };
        // Print on a fresh line, then repaint the prompt
        eprintln!("\r  thinking display: {label}");
        Some(Cmd::Repaint)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // --- Registry lookup tests ---

    #[test]
    fn find_exact_name_with_args() {
        let (spec, args) = CommandRegistry::find("/model grok-3").unwrap();
        assert_eq!(spec.command, Command::Model);
        assert_eq!(args, "grok-3");
    }

    #[test]
    fn find_exact_name_no_args() {
        let (spec, args) = CommandRegistry::find("/status").unwrap();
        assert_eq!(spec.command, Command::Status);
        assert_eq!(args, "");
    }

    #[test]
    fn find_alias() {
        let (spec, args) = CommandRegistry::find("/exit").unwrap();
        assert_eq!(spec.command, Command::Quit);
        assert_eq!(args, "");
    }

    #[test]
    fn find_with_extra_whitespace() {
        let (spec, args) = CommandRegistry::find("/model   grok-3  ").unwrap();
        assert_eq!(spec.command, Command::Model);
        assert_eq!(args, "grok-3");
    }

    #[test]
    fn find_unknown_returns_none() {
        assert!(CommandRegistry::find("/unknown").is_none());
    }

    #[test]
    fn find_not_slash_returns_none() {
        assert!(CommandRegistry::find("model grok-3").is_none());
    }

    #[test]
    fn find_empty_returns_none() {
        assert!(CommandRegistry::find("").is_none());
        assert!(CommandRegistry::find("/").is_none());
    }

    #[test]
    fn find_rejects_trailing_args_for_no_arg_commands() {
        // ArgSpec::None commands must not match with trailing args
        assert!(CommandRegistry::find("/quit foo").is_none());
        assert!(CommandRegistry::find("/exit bar").is_none());
        assert!(CommandRegistry::find("/status hello").is_none());
        assert!(CommandRegistry::find("/sessions junk").is_none());
        assert!(CommandRegistry::find("/compact stuff").is_none());
    }

    #[test]
    fn find_allows_args_for_optional_arg_commands() {
        assert!(CommandRegistry::find("/model grok-3").is_some());
        assert!(CommandRegistry::find("/model").is_some()); // optional = still matches bare
        assert!(CommandRegistry::find("/think display").is_some());
        assert!(CommandRegistry::find("/think").is_some());
        assert!(CommandRegistry::find("/undo 3 --code").is_some());
        assert!(CommandRegistry::find("/undo").is_some());
        assert!(CommandRegistry::find("/resume abc123").is_some());
        assert!(CommandRegistry::find("/resume").is_some());
    }

    // --- Prefix matching tests ---

    #[test]
    fn prefix_matches_empty_returns_all() {
        let matches = CommandRegistry::prefix_matches("");
        assert_eq!(matches.len(), COMMANDS.len());
    }

    #[test]
    fn prefix_matches_single() {
        let matches = CommandRegistry::prefix_matches("mo");
        assert_eq!(matches.len(), 1);
        assert_eq!(matches[0].name, "model");
    }

    #[test]
    fn prefix_matches_multiple() {
        let matches = CommandRegistry::prefix_matches("s");
        assert_eq!(matches.len(), 2); // sessions, status
        assert_eq!(matches[0].name, "sessions");
        assert_eq!(matches[1].name, "status");
    }

    #[test]
    fn prefix_matches_exact_match_priority() {
        let matches = CommandRegistry::prefix_matches("status");
        assert_eq!(matches[0].name, "status");
    }

    #[test]
    fn prefix_matches_strips_slash() {
        let matches = CommandRegistry::prefix_matches("/mo");
        assert_eq!(matches.len(), 1);
        assert_eq!(matches[0].name, "model");
    }

    #[test]
    fn prefix_matches_no_match() {
        let matches = CommandRegistry::prefix_matches("xyz");
        assert!(matches.is_empty());
    }

    #[test]
    fn prefix_matches_includes_aliases() {
        // /ex should match quit via its "exit" alias
        let matches = CommandRegistry::prefix_matches("ex");
        assert_eq!(matches.len(), 1);
        assert_eq!(matches[0].name, "quit");
    }

    #[test]
    fn prefix_matches_alias_exact_match_priority() {
        let matches = CommandRegistry::prefix_matches("exit");
        assert_eq!(matches[0].name, "quit");
    }

    // --- All commands tests ---

    #[test]
    fn all_commands_have_descriptions() {
        for cmd in CommandRegistry::all() {
            assert!(
                !cmd.description.is_empty(),
                "command '{}' has empty description",
                cmd.name
            );
        }
    }

    #[test]
    fn all_commands_sorted_alphabetically() {
        let all = CommandRegistry::all();
        let names: Vec<&str> = all.iter().map(|c| c.name).collect();
        let mut sorted = names.clone();
        sorted.sort();
        assert_eq!(names, sorted);
    }

    // --- File path token extraction ---

    #[test]
    fn extract_at_prefix() {
        let (start, token) = extract_path_token("look at @src/main", 17).unwrap();
        assert_eq!(start, 9); // after the @
        assert_eq!(token, "src/main");
    }

    #[test]
    fn extract_dot_slash() {
        let (start, token) = extract_path_token("edit ./src/lib.rs", 17).unwrap();
        assert_eq!(start, 5);
        assert_eq!(token, "./src/lib.rs");
    }

    #[test]
    fn extract_dot_dot_slash() {
        let (start, token) = extract_path_token("read ../file.txt", 16).unwrap();
        assert_eq!(start, 5);
        assert_eq!(token, "../file.txt");
    }

    #[test]
    fn extract_tilde_prefix() {
        let (start, token) = extract_path_token("check ~/config", 14).unwrap();
        assert_eq!(start, 6);
        assert_eq!(token, "~/config");
    }

    #[test]
    fn extract_slash_in_token() {
        let (start, token) = extract_path_token("open src/main.rs", 16).unwrap();
        assert_eq!(start, 5);
        assert_eq!(token, "src/main.rs");
    }

    #[test]
    fn extract_no_path_token() {
        assert!(extract_path_token("hello world", 11).is_none());
    }

    #[test]
    fn extract_empty_line() {
        assert!(extract_path_token("", 0).is_none());
    }

    #[test]
    fn extract_at_alone() {
        // @ with nothing after — empty token after stripping
        let result = extract_path_token("look at @", 9);
        assert!(result.is_none() || result.unwrap().1.is_empty());
    }
}
