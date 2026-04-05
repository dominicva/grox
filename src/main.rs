mod agent;
mod api;
mod permissions;
mod tools;
mod util;

use agent::Agent;
use anyhow::{Context, Result};
use api::GrokClient;
use clap::Parser;
use colored::Colorize;
use permissions::{PermissionMode, SessionPermissions};
use rustyline::DefaultEditor;
use serde_json::json;
use std::io::{Write, stdout};

#[derive(Parser)]
#[command(name = "grox", about = "Agentic coding with Grok")]
struct Cli {
    /// Model name (overrides GROX_MODEL env var)
    #[arg(long)]
    model: Option<String>,

    /// Print raw SSE events to stderr
    #[arg(long)]
    verbose: bool,

    /// Auto-approve writes inside the project root
    #[arg(long, conflicts_with_all = ["read_only", "yolo"])]
    auto_approve_writes: bool,

    /// Read-only mode: deny all writes and shell execution
    #[arg(long, conflicts_with_all = ["auto_approve_writes", "yolo"])]
    read_only: bool,

    /// YOLO mode: auto-approve everything, including destructive commands
    #[arg(long, conflicts_with_all = ["auto_approve_writes", "read_only"])]
    yolo: bool,
}

#[tokio::main]
async fn main() -> Result<()> {
    let _ = dotenvy::dotenv();
    let cli = Cli::parse();

    let api_key =
        std::env::var("XAI_API_KEY").context("XAI_API_KEY environment variable not set")?;

    let model = cli.model
        .unwrap_or_else(|| std::env::var("GROX_MODEL").unwrap_or_else(|_| "grok-3-fast".to_string()));

    if cli.verbose {
        // SAFETY: called once at startup before any threads are spawned
        unsafe { std::env::set_var("GROX_VERBOSE", "1") };
    }

    let permission_mode = if cli.yolo {
        PermissionMode::Yolo
    } else if cli.read_only {
        PermissionMode::ReadOnly
    } else if cli.auto_approve_writes {
        PermissionMode::Trust
    } else {
        PermissionMode::Default
    };

    let cwd = std::env::current_dir().context("Failed to get current directory")?;
    let project_root = util::detect_project_root(&cwd);

    println!("{}", "grox — agentic coding with Grok".bold());
    println!(
        "model: {}  |  project: {}  |  mode: {}  |  type {} to exit\n",
        model.cyan(),
        project_root.display().to_string().cyan(),
        format!("{permission_mode}").cyan(),
        "/quit".dimmed()
    );

    let mut session_perms = SessionPermissions::new(permission_mode, project_root.clone());

    let client = GrokClient::new(api_key, model);
    let agent = Agent::new(&client, &project_root);

    let system_prompt = json!({
        "role": "system",
        "content": format!(
            "You are Grox, a coding agent powered by Grok. You help developers understand and work with their codebase.

Project root: {}

Rules:
- Be concise and direct. Lead with the answer, not the process.
- Do NOT narrate what you are about to do or explain your tool usage. Just use tools silently and respond with findings.
- Do NOT thank the user for letting you read files — you have autonomous access to tools.
- When exploring a codebase, use list_files and file_read proactively to gather context before responding.
- Keep responses short. Use bullet points over paragraphs. Skip preamble.",
            project_root.display()
        )
    });

    let mut previous_response_id: Option<String> = None;
    let mut rl = DefaultEditor::new()?;

    loop {
        let input = match rl.readline(&format!("{} ", ">>".green().bold())) {
            Ok(line) => line,
            Err(
                rustyline::error::ReadlineError::Interrupted
                | rustyline::error::ReadlineError::Eof,
            ) => break,
            Err(e) => return Err(e.into()),
        };

        let input = input.trim().to_string();
        if input.is_empty() {
            continue;
        }
        if input == "/quit" || input == "/exit" {
            break;
        }

        rl.add_history_entry(&input)?;

        let api_input = vec![
            system_prompt.clone(),
            json!({
                "role": "user",
                "content": input,
            }),
        ];

        println!();

        match agent
            .run(
                api_input,
                previous_response_id.as_deref(),
                &mut {
                    let mut first_token = true;
                    move |token: String| {
                        if first_token && !token.trim().is_empty() {
                            first_token = false;
                        }
                        print!("{token}");
                        let _ = stdout().flush();
                    }
                },
                &mut |name: &str, args: &str| {
                    // Compact one-line display: extract the key param for summary
                    let summary = summarize_tool_call(name, args);
                    println!("  {} {}", format!("▸ {name}").cyan(), summary.dimmed());
                    let _ = stdout().flush();
                },
                &mut |name: &str, output: &str| {
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
                },
                &mut |name: &str, args: &str| -> bool {
                    session_perms.authorize(name, args)
                },
            )
            .await
        {
            Ok(result) => {
                if result.text.is_empty() {
                    println!(
                        "{}",
                        "(no response from model)".dimmed()
                    );
                }
                println!();
                if let Some(usage) = &result.usage {
                    println!(
                        "{}",
                        format!(
                            "  tokens: {} in / {} out",
                            usage.input_tokens, usage.output_tokens
                        )
                        .dimmed()
                    );
                }
                previous_response_id = result.response_id;
            }
            Err(e) => {
                eprintln!("\n{} {e}\n", "error:".red().bold());
            }
        }
    }

    println!("{}", "\ngoodbye.".dimmed());
    Ok(())
}

fn summarize_tool_call(name: &str, args: &str) -> String {
    let parsed: serde_json::Value = serde_json::from_str(args).unwrap_or_default();
    match name {
        "file_read" | "list_files" => {
            parsed["path"].as_str().unwrap_or("?").to_string()
        }
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
