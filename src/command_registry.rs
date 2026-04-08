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
        COMMANDS
            .iter()
            .find(|spec| spec.name == cmd_name || spec.aliases.contains(&cmd_name))
            .map(|spec| (spec, args))
    }

    /// Return commands whose name starts with `prefix`.
    /// Exact matches sort first, then alphabetically.
    pub fn prefix_matches(prefix: &str) -> Vec<&'static CommandSpec> {
        let prefix = prefix.strip_prefix('/').unwrap_or(prefix);
        let mut matches: Vec<&CommandSpec> = COMMANDS
            .iter()
            .filter(|spec| spec.name.starts_with(prefix))
            .collect();
        matches.sort_by(|a, b| {
            let a_exact = a.name == prefix;
            let b_exact = b.name == prefix;
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
        // Only complete slash commands, not arguments after the command name.
        if !line.starts_with('/') || line[1..pos].contains(char::is_whitespace) {
            return Ok((pos, vec![]));
        }

        let prefix = &line[1..pos];
        let matches: Vec<&CommandSpec> = if prefix.is_empty() {
            CommandRegistry::all().iter().collect()
        } else {
            CommandRegistry::prefix_matches(prefix)
        };

        let pairs = matches
            .into_iter()
            .map(|cmd| {
                let arg_hint = match cmd.arg_spec {
                    ArgSpec::Optional(label) => format!(" [{label}]"),
                    ArgSpec::None => String::new(),
                };
                Pair {
                    display: format!("/{}{} — {}", cmd.name, arg_hint, cmd.description),
                    replacement: format!("/{}", cmd.name),
                }
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

        // Don't hint for exact match.
        if first.name == prefix {
            return None;
        }

        let suffix = &first.name[prefix.len()..];
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
