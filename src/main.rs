mod agent;
mod api;
mod checkpoint;
mod compaction;
mod context_assembler;
mod model_profile;
mod permissions;
mod prompt;
mod repo_context;
mod rewind;
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

    /// Resume a previous session (optionally specify session ID; defaults to most recent for this project)
    #[arg(long, num_args = 0..=1, default_missing_value = "")]
    resume: Option<String>,
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
    let (mut session_meta, mut transcript, mut history, resumed) = if let Some(ref resume_arg) = cli.resume {
        // Resume an existing session
        let resume_id = resume_arg.trim();
        let meta = if resume_id.is_empty() {
            // No ID specified — resume most recent session for this project
            let project_sessions = SessionIndex::list_for_project(
                &sessions_dir,
                &project_root.display().to_string(),
            )?;
            match project_sessions.into_iter().next() {
                Some(m) => m,
                None => {
                    eprintln!("{}", "no previous sessions found for this project".red().bold());
                    std::process::exit(1);
                }
            }
        } else {
            // ID specified — find by prefix match
            let all = SessionIndex::list(&sessions_dir)?;
            let matches: Vec<_> = all.into_iter()
                .filter(|s| s.session_id.starts_with(resume_id))
                .collect();
            match matches.len() {
                0 => {
                    eprintln!("{}", format!("no session found matching '{resume_id}'").red().bold());
                    std::process::exit(1);
                }
                1 => matches.into_iter().next().unwrap(),
                n => {
                    eprintln!("{}", format!("'{resume_id}' matches {n} sessions — be more specific:").red().bold());
                    for m in &matches {
                        eprintln!("  {} ({})", &m.session_id[..8], m.last_active.format("%Y-%m-%d %H:%M"));
                    }
                    std::process::exit(1);
                }
            }
        };

        let t = Transcript::new(SessionMeta::transcript_path(&sessions_dir, &meta.session_id));
        let entries = t.read_all()
            .with_context(|| format!("failed to read transcript for session {}", &meta.session_id[..8]))?;
        (meta, t, entries, true)
    } else {
        // New session
        let meta = SessionMeta::new(&model, project_root.display().to_string());
        let t = Transcript::new(SessionMeta::transcript_path(&sessions_dir, &meta.session_id));
        t.create()?;
        meta.save(&sessions_dir)?;
        (meta, t, Vec::new(), false)
    };

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
    if resumed {
        println!(
            "{}",
            format!("session: {} (resumed)", &session_meta.session_id[..8]).dimmed()
        );
        // Show brief summary of previous state
        let user_turns = history.iter().filter(|e| matches!(e, TranscriptEntry::UserMessage { .. })).count();
        let assistant_turns = history.iter().filter(|e| matches!(e, TranscriptEntry::AssistantMessage { .. })).count();
        println!(
            "{}",
            format!(
                "  {} user turns, {} assistant responses — last active {}",
                user_turns,
                assistant_turns,
                session_meta.last_active.format("%Y-%m-%d %H:%M UTC")
            ).dimmed()
        );
    } else {
        println!(
            "{}",
            format!("session: {}", &session_meta.session_id[..8]).dimmed()
        );
    }
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

        if input == "/undo" || input.starts_with("/undo ") {
            let args = input.strip_prefix("/undo").unwrap().trim();

            // Parse optional mode flag and turn number
            let mut mode = rewind::RewindMode::Both;
            let mut turn_number: Option<usize> = None;
            let mut parse_error = false;

            for arg in args.split_whitespace() {
                match arg {
                    "--code" => mode = rewind::RewindMode::CodeOnly,
                    "--conversation" => mode = rewind::RewindMode::ConversationOnly,
                    "--both" => mode = rewind::RewindMode::Both,
                    _ => {
                        if let Ok(n) = arg.parse::<usize>() {
                            turn_number = Some(n);
                        } else {
                            println!(
                                "{}",
                                format!("  unknown argument: {arg}. usage: /undo [N] [--code|--conversation|--both]")
                                    .dimmed()
                            );
                            parse_error = true;
                            break;
                        }
                    }
                }
            }

            if parse_error {
                continue;
            }

            let result = if let Some(n) = turn_number {
                rewind::rewind_to_turn(&history, n, &project_root, mode)
            } else {
                rewind::undo_last_turn(&history, &project_root, mode)
            };

            match result {
                Ok(rr) => {
                    println!("{}", rewind::format_rewind_result(&rr).yellow());
                    history = rr.entries;
                    transcript.atomic_rewrite(&history)?;
                }
                Err(e) => {
                    println!("{}", format!("  undo failed: {e}").red());
                }
            }
            continue;
        }

        if input == "/sessions" {
            let project_str = project_root.display().to_string();
            match SessionIndex::list_for_project(&sessions_dir, &project_str) {
                Ok(sessions) if sessions.is_empty() => {
                    println!("{}", "  no sessions found for this project".dimmed());
                }
                Ok(sessions) => {
                    println!("  {}", "recent sessions:".bold());
                    for (i, s) in sessions.iter().take(10).enumerate() {
                        let active = if s.session_id == session_meta.session_id { " (current)" } else { "" };
                        println!(
                            "  {}. {} — {} — {} in / {} out{}",
                            i + 1,
                            &s.session_id[..8].cyan(),
                            s.last_active.format("%Y-%m-%d %H:%M").to_string().dimmed(),
                            s.cumulative_input_tokens,
                            s.cumulative_output_tokens,
                            active.green(),
                        );
                    }
                    println!("{}", "  use /resume <id> to resume a session".dimmed());
                }
                Err(e) => {
                    println!("{}", format!("  failed to list sessions: {e}").red());
                }
            }
            continue;
        }

        if input == "/resume" || input.starts_with("/resume ") {
            let id_arg = input.strip_prefix("/resume").unwrap().trim();
            if id_arg.is_empty() {
                println!("{}", "  usage: /resume <session-id-prefix>".dimmed());
                println!("{}", "  use /sessions to see available sessions".dimmed());
                continue;
            }

            // Find matching session
            let all = match SessionIndex::list(&sessions_dir) {
                Ok(s) => s,
                Err(e) => {
                    println!("{}", format!("  failed to list sessions: {e}").red());
                    continue;
                }
            };
            let matches: Vec<_> = all.into_iter()
                .filter(|s| s.session_id.starts_with(id_arg))
                .collect();

            match matches.len() {
                0 => {
                    println!("{}", format!("  no session found matching '{id_arg}'").dimmed());
                }
                1 => {
                    let target = &matches[0];
                    if target.session_id == session_meta.session_id {
                        println!("{}", "  already in this session".dimmed());
                        continue;
                    }
                    let t = Transcript::new(SessionMeta::transcript_path(&sessions_dir, &target.session_id));
                    match t.read_all() {
                        Ok(entries) => {
                            let user_turns = entries.iter().filter(|e| matches!(e, TranscriptEntry::UserMessage { .. })).count();
                            let assistant_turns = entries.iter().filter(|e| matches!(e, TranscriptEntry::AssistantMessage { .. })).count();
                            session_meta = target.clone();
                            history = entries;
                            transcript = Transcript::new(
                                SessionMeta::transcript_path(&sessions_dir, &session_meta.session_id),
                            );
                            println!(
                                "  {} session {} — {} user turns, {} assistant responses",
                                "resumed".green(),
                                &session_meta.session_id[..8].cyan(),
                                user_turns,
                                assistant_turns,
                            );
                        }
                        Err(e) => {
                            println!("{}", format!("  failed to read transcript: {e}").red());
                        }
                    }
                }
                n => {
                    println!("{}", format!("  '{id_arg}' matches {n} sessions — be more specific:").dimmed());
                    for m in &matches {
                        println!("    {} ({})", &m.session_id[..8], m.last_active.format("%Y-%m-%d %H:%M"));
                    }
                }
            }
            continue;
        }

        if input == "/compact" {
            let old_estimate = assembler.estimate_tokens(&history);
            let old_count = history.len();

            // Try heuristic first
            let heuristic = compaction::heuristic_compact(&history, &project_root);
            let after_heuristic = assembler.estimate_tokens(&heuristic.entries);
            let threshold = model_profile::ModelProfile::for_model(client.model()).compaction_threshold();

            // If still over threshold after heuristic, escalate to LLM compaction
            let result = if after_heuristic > threshold {
                match compaction::llm_compact(&heuristic.entries, client.model(), &client).await {
                    Ok(llm_result) if llm_result.compacted => llm_result,
                    Ok(llm_result) => {
                        // LLM didn't help — carry usage into fallback
                        compaction::CompactionResult {
                            llm_usage: llm_result.llm_usage,
                            ..heuristic
                        }
                    }
                    Err(e) => {
                        eprintln!("{}", format!("  warning: LLM compaction failed: {e}").yellow());
                        heuristic
                    }
                }
            } else {
                heuristic
            };

            // Always account for LLM usage even if compaction didn't reduce size
            if let Some(u) = &result.llm_usage {
                session_meta.cumulative_input_tokens += u.input_tokens;
                session_meta.cumulative_output_tokens += u.output_tokens;
                let _ = session_meta.save(&sessions_dir);
            }

            if result.compacted {
                let new_estimate = assembler.estimate_tokens(&result.entries);
                let new_count = result.entries.len();
                let llm_cost = result.llm_usage.as_ref().and_then(|u| {
                    let profile = model_profile::ModelProfile::for_model(client.model());
                    profile.estimate_cost(u.input_tokens, u.output_tokens)
                });
                history = result.entries;
                transcript.atomic_rewrite(&history)?;
                let mut msg = format!(
                    "  compacted: {} entries → {} entries, ~{} → ~{} tokens",
                    old_count, new_count, old_estimate, new_estimate
                );
                if let Some(cost) = llm_cost {
                    msg.push_str(&format!(" (LLM summarization: ~${cost:.4})"));
                }
                println!("{}", msg.yellow());
            } else {
                let llm_cost = result.llm_usage.as_ref().and_then(|u| {
                    let profile = model_profile::ModelProfile::for_model(client.model());
                    profile.estimate_cost(u.input_tokens, u.output_tokens)
                });
                if let Some(cost) = llm_cost {
                    println!("{}", format!("  nothing to compact (LLM summarization attempt: ~${cost:.4})").yellow());
                } else {
                    println!("{}", "  nothing to compact".dimmed());
                }
            }
            continue;
        }

        rl.add_history_entry(&input)?;

        // Append user message to history and transcript
        let user_entry = TranscriptEntry::user_message(&input);
        history.push(user_entry.clone());
        transcript.append(&user_entry)?;

        // Preflight budget check: compact if estimated tokens exceed threshold
        if let Some(result) = compaction::maybe_compact(&history, &assembler, client.model(), &project_root, &client).await {
            // Always account for LLM usage even if compaction didn't reduce size
            if let Some(u) = &result.llm_usage {
                session_meta.cumulative_input_tokens += u.input_tokens;
                session_meta.cumulative_output_tokens += u.output_tokens;
                let _ = session_meta.save(&sessions_dir);
            }
            if result.compacted {
                let old_estimate = assembler.estimate_tokens(&history);
                let new_estimate = assembler.estimate_tokens(&result.entries);
                let threshold = model_profile::ModelProfile::for_model(client.model()).compaction_threshold();
                let llm_cost = result.llm_usage.as_ref().and_then(|u| {
                    let profile = model_profile::ModelProfile::for_model(client.model());
                    profile.estimate_cost(u.input_tokens, u.output_tokens)
                });
                history = result.entries;
                transcript.atomic_rewrite(&history)?;
                let mut msg = format!(
                    "  auto-compacted: ~{old_estimate} → ~{new_estimate} tokens (threshold: {threshold})",
                );
                if let Some(cost) = llm_cost {
                    msg.push_str(&format!(" [LLM summarization: ~${cost:.4}]"));
                }
                eprintln!("{}", msg.yellow());
            }
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
                &mut |entry: &session::TranscriptEntry| -> anyhow::Result<()> {
                    // Persist each entry to disk before adding to in-memory history.
                    // If disk write fails, return the error so the agent halts —
                    // continuing would risk file edits with no durable checkpoint.
                    transcript.append(entry)?;
                    history.push(entry.clone());
                    Ok(())
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
