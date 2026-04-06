mod agent;
mod api;
mod compaction;
mod context_assembler;
mod model_profile;
mod permissions;
mod prompt;
mod repo_context;
mod session;
mod tools;
mod util;

use agent::Agent;
use anyhow::{Context, Result};
use api::GrokClient;
use chrono::Utc;
use clap::Parser;
use colored::Colorize;
use context_assembler::ContextAssembler;
use permissions::{PermissionMode, SessionPermissions};
use rustyline::DefaultEditor;
use session::{SessionIndex, SessionMeta, Transcript, TranscriptEntry};
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

    let api_key = match std::env::var("XAI_API_KEY") {
        Ok(key) if !key.is_empty() => key,
        _ => {
            eprintln!("{}", "XAI_API_KEY is not set.".red().bold());
            eprintln!();
            eprintln!("Get your API key at: {}", "https://console.x.ai/".cyan());
            eprintln!("Then export it in your shell:");
            eprintln!();
            eprintln!("  {}", "export XAI_API_KEY=your-key-here".dimmed());
            std::process::exit(1);
        }
    };

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

    // --- Session setup ---
    let sessions_dir = SessionIndex::default_sessions_dir()?;
    let mut session_meta = SessionMeta::new(&model, project_root.display().to_string());
    let transcript = Transcript::new(
        SessionMeta::transcript_path(&sessions_dir, &session_meta.session_id),
    );
    transcript.create()?;
    session_meta.save(&sessions_dir)?;

    println!("{}", "grox — agentic coding with Grok".bold());
    println!(
        "model: {}  |  project: {}  |  mode: {}  |  type {} to exit",
        model.cyan(),
        project_root.display().to_string().cyan(),
        format!("{permission_mode}").cyan(),
        "/quit".dimmed()
    );
    println!(
        "{}",
        "note: grox can read any file on your system. File contents are sent to xAI and stored for 30 days."
            .dimmed()
    );
    println!(
        "{}",
        format!("session: {}", &session_meta.session_id[..8]).dimmed()
    );
    println!();

    let mut session_perms = SessionPermissions::new(permission_mode, project_root.clone());

    let mut client = GrokClient::new(api_key, model);

    // Gather repo context
    let repo_ctx = repo_context::RepoContext::gather(&project_root);
    if repo_ctx.truncated {
        eprintln!(
            "{}",
            "  warning: repo context exceeds 10K characters and was truncated"
                .yellow()
        );
    }

    // Load GROX.md custom instructions if present
    let grox_md = util::load_grox_md(&project_root);
    if let Some(ref content) = grox_md
        && content.contains("truncated")
    {
        eprintln!(
            "{}",
            "  warning: GROX.md exceeds 10K characters and was truncated"
                .yellow()
        );
    }

    let repo_ctx_text = if repo_ctx.text.is_empty() { None } else { Some(repo_ctx.text.as_str()) };
    let system_content = prompt::build_system_prompt(
        &project_root,
        repo_ctx_text,
        grox_md.as_deref(),
    );

    let system_prompt = json!({
        "role": "system",
        "content": system_content
    });

    let mut assembler = ContextAssembler::new(system_prompt);
    let mut history: Vec<TranscriptEntry> = Vec::new();

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

        if input.starts_with("/model ") {
            let new_model = input.strip_prefix("/model ").unwrap().trim().to_string();
            if new_model.is_empty() {
                println!("{}", "usage: /model <name>".dimmed());
            } else {
                client.set_model(new_model.clone());
                session_meta.model = new_model.clone();
                println!("  model switched to {}", new_model.cyan());
            }
            continue;
        }

        if input == "/status" {
            let profile = model_profile::ModelProfile::for_model(client.model());
            let estimated = assembler.estimate_tokens(&history);
            println!("  model:   {}", client.model().cyan());
            println!("  project: {}", project_root.display().to_string().cyan());
            println!("  mode:    {}", format!("{permission_mode}").cyan());
            println!("  session: {}", &session_meta.session_id[..8].dimmed());
            println!(
                "  context: ~{} tokens (threshold: {})",
                estimated.to_string().cyan(),
                profile.compaction_threshold().to_string().cyan()
            );
            println!(
                "  tools:   {}",
                tools::Tool::all()
                    .iter()
                    .map(|t| format!("{t:?}"))
                    .collect::<Vec<_>>()
                    .join(", ")
                    .dimmed()
            );
            continue;
        }

        if input == "/compact" {
            let result = compaction::heuristic_compact(&history, &project_root);
            if result.compacted {
                let old_estimate = assembler.estimate_tokens(&history);
                let new_estimate = assembler.estimate_tokens(&result.entries);
                let old_count = history.len();
                let new_count = result.entries.len();
                history = result.entries;
                transcript.atomic_rewrite(&history)?;
                println!(
                    "{}",
                    format!(
                        "  compacted: {} entries → {} entries, ~{} → ~{} tokens",
                        old_count, new_count, old_estimate, new_estimate
                    )
                    .yellow()
                );
            } else {
                println!("{}", "  nothing to compact".dimmed());
            }
            continue;
        }

        rl.add_history_entry(&input)?;

        // Append user message to history and transcript
        let user_entry = TranscriptEntry::user_message(&input);
        history.push(user_entry.clone());
        transcript.append(&user_entry)?;

        // Preflight budget check: compact if estimated tokens exceed threshold
        if let Some(result) = compaction::maybe_compact(&history, &assembler, client.model(), &project_root) {
            let old_estimate = assembler.estimate_tokens(&history);
            let new_estimate = assembler.estimate_tokens(&result.entries);
            let threshold = model_profile::ModelProfile::for_model(client.model()).compaction_threshold();
            history = result.entries;
            transcript.atomic_rewrite(&history)?;
            eprintln!(
                "{}",
                format!(
                    "  auto-compacted: ~{old_estimate} → ~{new_estimate} tokens (threshold: {threshold})",
                )
                .yellow()
            );
        }

        // Build full message array from history
        let api_input = assembler.build_messages(&history);

        println!();

        let agent = Agent::new(&client, &project_root);
        match agent
            .run(
                api_input,
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
                &mut || {
                    // Refresh repo context after mutating tools
                    let fresh_ctx = repo_context::RepoContext::gather(&project_root);
                    let fresh_ctx_text = if fresh_ctx.text.is_empty() {
                        None
                    } else {
                        Some(fresh_ctx.text.as_str())
                    };
                    let fresh_system_content = prompt::build_system_prompt(
                        &project_root,
                        fresh_ctx_text,
                        grox_md.as_deref(),
                    );
                    let fresh_prompt = json!({
                        "role": "system",
                        "content": fresh_system_content
                    });
                    assembler.set_system_prompt(fresh_prompt.clone());
                    fresh_prompt
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

                // Persist transcript entries and update history
                for entry in &result.entries {
                    transcript.append(entry)?;
                }
                history.extend(result.entries);

                // Update session metadata
                if let Some(usage) = &result.usage {
                    let cost_str = estimate_cost(client.model(), usage);
                    println!(
                        "{}",
                        format!(
                            "  tokens: {} in / {} out{}",
                            usage.input_tokens, usage.output_tokens, cost_str
                        )
                        .dimmed()
                    );
                    session_meta.cumulative_input_tokens += usage.input_tokens;
                    session_meta.cumulative_output_tokens += usage.output_tokens;
                }
                session_meta.last_active = Utc::now();
                // Best-effort metadata save — don't fail the session on metadata errors
                let _ = session_meta.save(&sessions_dir);
            }
            Err(e) => {
                eprintln!("\n{} {e}\n", "error:".red().bold());
            }
        }
    }

    println!("{}", "\ngoodbye.".dimmed());
    Ok(())
}

/// Best-effort cost estimate. Returns empty string if pricing is unknown.
fn estimate_cost(model: &str, usage: &api::Usage) -> String {
    let profile = model_profile::ModelProfile::for_model(model);
    match profile.estimate_cost(usage.input_tokens, usage.output_tokens) {
        Some(cost) => format!("  (~${cost:.4})"),
        None => String::new(),
    }
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
