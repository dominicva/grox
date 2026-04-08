use rustyline::completion::{Completer, Pair};
use rustyline::highlight::Highlighter;
use rustyline::hint::{Hint, Hinter};
use rustyline::validate::Validator;
use rustyline::{Context, Helper};
use std::borrow::Cow;

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
        description: "cycle reasoning effort (off → low → high)",
        arg_spec: ArgSpec::None,
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

/// Rustyline helper providing slash-command completion and hints.
pub struct GroxHelper;

impl Helper for GroxHelper {}

impl Completer for GroxHelper {
    type Candidate = Pair;

    fn complete(
        &self,
        line: &str,
        pos: usize,
        _ctx: &Context<'_>,
    ) -> rustyline::Result<(usize, Vec<Pair>)> {
        // Only complete when cursor is at end of line on a slash command,
        // and there's no whitespace (still typing command name, not args).
        if pos != line.len() || pos < 1 || !line.starts_with('/') {
            return Ok((pos, vec![]));
        }
        if line[1..].contains(char::is_whitespace) {
            return Ok((pos, vec![]));
        }

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
                // Yield the canonical name entry.
                let mut entries = vec![];
                if cmd.name.starts_with(prefix) || prefix.is_empty() {
                    entries.push(Pair {
                        display: format!("/{}{} — {}", cmd.name, arg_hint, cmd.description),
                        replacement: format!("/{}", cmd.name),
                    });
                }
                // Also yield matching aliases so /ex Tab completes to /exit.
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

        // Replace from position 0 (includes the `/`).
        Ok((0, pairs))
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
        assert!(CommandRegistry::find("/think xyz").is_none());
        assert!(CommandRegistry::find("/sessions junk").is_none());
        assert!(CommandRegistry::find("/compact stuff").is_none());
    }

    #[test]
    fn find_allows_args_for_optional_arg_commands() {
        assert!(CommandRegistry::find("/model grok-3").is_some());
        assert!(CommandRegistry::find("/model").is_some()); // optional = still matches bare
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
}
