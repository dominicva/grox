mod agent;
mod api;
mod checkpoint;
mod command_registry;
mod compaction;
mod context_assembler;
mod file_index;
mod model_profile;
mod permissions;
mod prompt;
mod renderer;
mod repo_context;
mod rewind;
mod session;
mod tools;
mod util;

use agent::Agent;
use anyhow::{Context, Result};
use api::{GrokClient, ReasoningEffort};
use chrono::Utc;
use clap::Parser;
use colored::Colorize;
use context_assembler::ContextAssembler;
use permissions::{PermissionMode, SessionPermissions};
use command_registry::{Command, CommandRegistry, GroxHelper, ThinkToggleHandler};
use rustyline::{Editor, EventHandler, KeyCode, KeyEvent, Modifiers};
use serde_json::json;
use session::{SessionIndex, SessionMeta, Transcript, TranscriptEntry};

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

    /// Set initial reasoning effort (low or high). Only effective on models that support it.
    #[arg(long, num_args = 0..=1, default_missing_value = "low")]
    think: Option<String>,

    /// Do not store conversations on the provider side
    #[arg(long)]
    no_store: bool,
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

    const DEFAULT_MODEL: &str = "grok-4-1-fast-reasoning";

    let user_specified_model = cli.model.is_some() || std::env::var("GROX_MODEL").is_ok();
    let model = cli.model.unwrap_or_else(|| {
        std::env::var("GROX_MODEL").unwrap_or_else(|_| DEFAULT_MODEL.to_string())
    });

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
    let (mut session_meta, mut transcript, mut history, resumed) = if let Some(ref resume_arg) =
        cli.resume
    {
        // Resume an existing session
        let resume_id = resume_arg.trim();
        let meta = if resume_id.is_empty() {
            // No ID specified — resume most recent session for this project
            let project_sessions =
                SessionIndex::list_for_project(&sessions_dir, &project_root.display().to_string())?;
            match project_sessions.into_iter().next() {
                Some(m) => m,
                None => {
                    eprintln!(
                        "{}",
                        "no previous sessions found for this project".red().bold()
                    );
                    std::process::exit(1);
                }
            }
        } else {
            // ID specified — find by prefix match
            let all = SessionIndex::list(&sessions_dir)?;
            let matches: Vec<_> = all
                .into_iter()
                .filter(|s| s.session_id.starts_with(resume_id))
                .collect();
            match matches.len() {
                0 => {
                    eprintln!(
                        "{}",
                        format!("no session found matching '{resume_id}'")
                            .red()
                            .bold()
                    );
                    std::process::exit(1);
                }
                1 => matches.into_iter().next().unwrap(),
                n => {
                    eprintln!(
                        "{}",
                        format!("'{resume_id}' matches {n} sessions — be more specific:")
                            .red()
                            .bold()
                    );
                    for m in &matches {
                        eprintln!(
                            "  {} ({})",
                            &m.session_id[..8],
                            m.last_active.format("%Y-%m-%d %H:%M")
                        );
                    }
                    std::process::exit(1);
                }
            }
        };

        // Warn if resuming a session from a different project
        let current_project = project_root.display().to_string();
        if meta.project_root != current_project {
            eprintln!(
                "{}",
                format!(
                    "  warning: session {} was created in '{}', but you are in '{}'",
                    &meta.session_id[..8],
                    meta.project_root,
                    current_project
                )
                .yellow()
            );
            eprintln!(
                "{}",
                "  tools will operate on the current directory, not the original project".yellow()
            );
        }

        let transcript_path = SessionMeta::transcript_path(&sessions_dir, &meta.session_id);
        let t = Transcript::new(&transcript_path);
        if !transcript_path.exists() {
            eprintln!(
                "{}",
                format!(
                    "  warning: transcript file missing for session {} — starting with empty history",
                    &meta.session_id[..8]
                ).yellow()
            );
            t.create()?;
        }
        let entries = t.read_all().with_context(|| {
            format!(
                "failed to read transcript for session {}",
                &meta.session_id[..8]
            )
        })?;
        if transcript_path.exists()
            && entries.is_empty()
            && std::fs::metadata(&transcript_path)
                .map(|m| m.len() > 0)
                .unwrap_or(false)
        {
            eprintln!(
                "{}",
                format!(
                    "  warning: transcript for session {} appears corrupt — starting with empty history",
                    &meta.session_id[..8]
                ).yellow()
            );
        }
        (meta, t, entries, true)
    } else {
        // New session
        let meta = SessionMeta::new(&model, project_root.display().to_string());
        let t = Transcript::new(SessionMeta::transcript_path(
            &sessions_dir,
            &meta.session_id,
        ));
        t.create()?;
        meta.save(&sessions_dir)?;
        (meta, t, Vec::new(), false)
    };

    // Resolve model for this session: on resume, prefer saved model unless user explicitly overrode
    let model = if resumed && !user_specified_model {
        session_meta.model.clone()
    } else {
        if resumed && session_meta.model != model {
            session_meta.model = model.clone();
        }
        model
    };

    println!("{}", "grox — agentic coding with Grok".bold());
    println!(
        "model: {}  |  project: {}  |  mode: {}  |  type {} to exit",
        model.cyan(),
        project_root.display().to_string().cyan(),
        format!("{permission_mode}").cyan(),
        "/quit".dimmed()
    );
    let no_store = cli.no_store || std::env::var("GROX_NO_STORE").as_deref() == Ok("1");
    if no_store {
        println!(
            "{}",
            "note: grox can read any file on your system. File contents are sent to xAI (store: false — provider asked not to retain)."
                .dimmed()
        );
    } else {
        println!(
            "{}",
            "note: grox can read any file on your system. File contents are sent to xAI and stored for 30 days (use --no-store to opt out)."
                .dimmed()
        );
    }
    if resumed {
        println!(
            "{}",
            format!("session: {} (resumed)", &session_meta.session_id[..8]).dimmed()
        );
        // Show brief summary of previous state
        let user_turns = history
            .iter()
            .filter(|e| matches!(e, TranscriptEntry::UserMessage { .. }))
            .count();
        let assistant_turns = history
            .iter()
            .filter(|e| matches!(e, TranscriptEntry::AssistantMessage { .. }))
            .count();
        println!(
            "{}",
            format!(
                "  {} user turns, {} assistant responses — last active {}",
                user_turns,
                assistant_turns,
                session_meta.last_active.format("%Y-%m-%d %H:%M UTC")
            )
            .dimmed()
        );
    } else {
        println!(
            "{}",
            format!("session: {}", &session_meta.session_id[..8]).dimmed()
        );
    }
    println!();

    let mut session_perms = SessionPermissions::new(permission_mode, project_root.clone());

    let mut client = GrokClient::new(api_key, model, session_meta.session_id.clone());

    if no_store {
        client.set_no_store(true);
    }

    // Set initial reasoning effort from --think flag
    if let Some(ref think_arg) = cli.think {
        let effort = match think_arg.as_str() {
            "low" => Some(ReasoningEffort::Low),
            "high" => Some(ReasoningEffort::High),
            other => {
                eprintln!(
                    "{}",
                    format!("  warning: unknown --think value '{other}', using 'low'").yellow()
                );
                Some(ReasoningEffort::Low)
            }
        };
        let profile = model_profile::ModelProfile::for_model(client.model());
        if profile.supports_reasoning_effort_control {
            client.set_reasoning_effort(effort);
        } else {
            eprintln!(
                "{}",
                format!(
                    "  warning: model {} does not support reasoning effort control, --think ignored",
                    client.model()
                )
                .yellow()
            );
        }
    }

    // Gather repo context
    let repo_ctx = repo_context::RepoContext::gather(&project_root);
    if repo_ctx.truncated {
        eprintln!(
            "{}",
            "  warning: repo context exceeds 10K characters and was truncated".yellow()
        );
    }

    // Load GROX.md custom instructions if present
    let grox_md = util::load_grox_md(&project_root);
    if let Some(ref content) = grox_md
        && content.contains("truncated")
    {
        eprintln!(
            "{}",
            "  warning: GROX.md exceeds 10K characters and was truncated".yellow()
        );
    }

    let repo_ctx_text = if repo_ctx.text.is_empty() {
        None
    } else {
        Some(repo_ctx.text.as_str())
    };
    let system_content =
        prompt::build_system_prompt(&project_root, repo_ctx_text, grox_md.as_deref());

    let system_prompt = json!({
        "role": "system",
        "content": system_content
    });

    let mut assembler = ContextAssembler::new(system_prompt);

    let file_idx = file_index::FileIndex::build(&project_root);

    let config = rustyline::Config::builder()
        .completion_type(rustyline::config::CompletionType::List)
        .build();
    let mut rl = Editor::with_config(config)?;
    rl.set_helper(Some(GroxHelper::with_file_index(file_idx.clone())));
    let mut term_renderer = renderer::TerminalRenderer::new();

    // Bind Ctrl+T to toggle thinking display
    rl.bind_sequence(
        KeyEvent(KeyCode::Char('t'), Modifiers::CTRL),
        EventHandler::Conditional(Box::new(ThinkToggleHandler::new(
            term_renderer.thinking_expanded_handle(),
        ))),
    );

    loop {
        let input = match rl.readline(&format!("{} ", ">>".green().bold())) {
            Ok(line) => line,
            Err(
                rustyline::error::ReadlineError::Interrupted | rustyline::error::ReadlineError::Eof,
            ) => break,
            Err(e) => return Err(e.into()),
        };

        let input = input.trim().to_string();
        if input.is_empty() {
            continue;
        }
        // --- Slash command dispatch via registry ---
        if input.starts_with('/') {
            match CommandRegistry::find(&input) {
                Some((spec, args)) => match spec.command {
                    Command::Quit => break,

                    Command::Model => {
                        let new_model = args.to_string();
                        if new_model.is_empty() {
                            println!("{}", "usage: /model <name>".dimmed());
                        } else {
                            let profile = model_profile::ModelProfile::for_model(&new_model);
                            if !profile.supports_tools {
                                println!(
                                    "{}",
                                    format!("  model {new_model} does not support tool use").red()
                                );
                            } else {
                                client.set_model(new_model.clone());
                                session_meta.model = new_model.clone();
                                println!("  model switched to {}", new_model.cyan());
                            }
                        }
                    }

                    Command::Think => {
                        if args == "display" {
                            let expanded = term_renderer.toggle_thinking_display();
                            let label = if expanded { "expanded" } else { "collapsed" };
                            println!("  thinking display: {}", label.cyan());
                        } else if !args.is_empty() {
                            println!(
                                "{}",
                                format!("  unknown argument: {args}. usage: /think [display]")
                                    .dimmed()
                            );
                        } else {
                            let profile =
                                model_profile::ModelProfile::for_model(client.model());
                            if !profile.supports_reasoning_effort_control {
                                println!(
                                    "{}",
                                    format!(
                                        "  model {} does not support reasoning effort control",
                                        client.model()
                                    )
                                    .dimmed()
                                );
                                if profile.supports_reasoning() {
                                    println!(
                                        "{}",
                                        "  (this model has built-in reasoning that is always active)"
                                            .dimmed()
                                    );
                                }
                            } else {
                                let next = match client.reasoning_effort() {
                                    None => Some(ReasoningEffort::Low),
                                    Some(ReasoningEffort::Low) => Some(ReasoningEffort::High),
                                    Some(ReasoningEffort::High) => None,
                                };
                                client.set_reasoning_effort(next);
                                match next {
                                    Some(effort) => {
                                        println!(
                                            "  reasoning effort: {}",
                                            effort.to_string().cyan()
                                        )
                                    }
                                    None => {
                                        println!("  reasoning effort: {}", "off".dimmed())
                                    }
                                }
                            }
                        }
                    }

                    Command::Status => {
                        let profile = model_profile::ModelProfile::for_model(client.model());
                        let estimated = assembler.estimate_tokens(&history);
                        println!("  model:   {}", client.model().cyan());
                        let mut caps = Vec::new();
                        if profile.returns_plaintext_reasoning {
                            caps.push("plaintext-reasoning");
                        }
                        if profile.returns_encrypted_reasoning {
                            caps.push("encrypted-reasoning");
                        }
                        if profile.supports_reasoning_effort_control {
                            caps.push("effort-control");
                        }
                        if !caps.is_empty() {
                            println!("  caps:    {}", caps.join(", ").cyan());
                        }
                        if profile.supports_reasoning_effort_control {
                            let effort_str = match client.reasoning_effort() {
                                Some(e) => e.to_string(),
                                None => "off".to_string(),
                            };
                            println!("  think:   {}", effort_str.cyan());
                        }
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
                    }

                    Command::Undo => {
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

                        if !parse_error {
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
                        }
                    }

                    Command::Sessions => {
                        let project_str = project_root.display().to_string();
                        match SessionIndex::list_for_project(&sessions_dir, &project_str) {
                            Ok(sessions) if sessions.is_empty() => {
                                println!("{}", "  no sessions found for this project".dimmed());
                            }
                            Ok(sessions) => {
                                println!("  {}", "recent sessions:".bold());
                                for (i, s) in sessions.iter().take(10).enumerate() {
                                    let active = if s.session_id == session_meta.session_id {
                                        " (current)"
                                    } else {
                                        ""
                                    };
                                    let summary_part = if s.summary.is_empty() {
                                        String::new()
                                    } else {
                                        format!(" — {}", s.summary)
                                    };
                                    println!(
                                        "  {}. {} — {} — {}{}{}",
                                        i + 1,
                                        &s.session_id[..8].cyan(),
                                        s.last_active
                                            .format("%Y-%m-%d %H:%M")
                                            .to_string()
                                            .dimmed(),
                                        s.model.dimmed(),
                                        summary_part.dimmed(),
                                        active.green(),
                                    );
                                }
                                println!(
                                    "{}",
                                    "  use /resume <id> to resume a session".dimmed()
                                );
                            }
                            Err(e) => {
                                println!("{}", format!("  failed to list sessions: {e}").red());
                            }
                        }
                    }

                    Command::Resume => {
                        if args.is_empty() {
                            println!("{}", "  usage: /resume <session-id-prefix>".dimmed());
                            println!(
                                "{}",
                                "  use /sessions to see available sessions".dimmed()
                            );
                        } else {
                            let all = match SessionIndex::list(&sessions_dir) {
                                Ok(s) => s,
                                Err(e) => {
                                    println!(
                                        "{}",
                                        format!("  failed to list sessions: {e}").red()
                                    );
                                    continue;
                                }
                            };
                            let matches: Vec<_> = all
                                .into_iter()
                                .filter(|s| s.session_id.starts_with(args))
                                .collect();

                            match matches.len() {
                                0 => {
                                    println!(
                                        "{}",
                                        format!("  no session found matching '{args}'").dimmed()
                                    );
                                }
                                1 => {
                                    let target = &matches[0];
                                    if target.session_id == session_meta.session_id {
                                        println!("{}", "  already in this session".dimmed());
                                    } else {
                                        let current_project =
                                            project_root.display().to_string();
                                        if target.project_root != current_project {
                                            println!(
                                                "{}",
                                                format!(
                                                    "  warning: session was created in '{}', tools will operate on '{}'",
                                                    target.project_root, current_project
                                                ).yellow()
                                            );
                                        }
                                        let t = Transcript::new(SessionMeta::transcript_path(
                                            &sessions_dir,
                                            &target.session_id,
                                        ));
                                        match t.read_all() {
                                            Ok(entries) => {
                                                let user_turns = entries
                                                    .iter()
                                                    .filter(|e| {
                                                        matches!(
                                                            e,
                                                            TranscriptEntry::UserMessage { .. }
                                                        )
                                                    })
                                                    .count();
                                                let assistant_turns = entries
                                                    .iter()
                                                    .filter(|e| {
                                                        matches!(
                                                            e,
                                                            TranscriptEntry::AssistantMessage {
                                                                ..
                                                            }
                                                        )
                                                    })
                                                    .count();
                                                session_meta = target.clone();
                                                history = entries;
                                                transcript =
                                                    Transcript::new(SessionMeta::transcript_path(
                                                        &sessions_dir,
                                                        &session_meta.session_id,
                                                    ));
                                                client.set_session_id(
                                                    session_meta.session_id.clone(),
                                                );
                                                if session_meta.model != client.model() {
                                                    client
                                                        .set_model(session_meta.model.clone());
                                                    println!(
                                                        "  model switched to {} (from resumed session)",
                                                        session_meta.model.cyan()
                                                    );
                                                }
                                                println!(
                                                    "  {} session {} — {} user turns, {} assistant responses",
                                                    "resumed".green(),
                                                    &session_meta.session_id[..8].cyan(),
                                                    user_turns,
                                                    assistant_turns,
                                                );
                                            }
                                            Err(e) => {
                                                println!(
                                                    "{}",
                                                    format!("  failed to read transcript: {e}")
                                                        .red()
                                                );
                                            }
                                        }
                                    }
                                }
                                n => {
                                    println!(
                                        "{}",
                                        format!(
                                            "  '{args}' matches {n} sessions — be more specific:"
                                        )
                                        .dimmed()
                                    );
                                    for m in &matches {
                                        println!(
                                            "    {} ({})",
                                            &m.session_id[..8],
                                            m.last_active.format("%Y-%m-%d %H:%M")
                                        );
                                    }
                                }
                            }
                        }
                    }

                    Command::Compact => {
                        let old_estimate = assembler.estimate_tokens(&history);
                        let old_count = history.len();

                        let heuristic = compaction::heuristic_compact(&history, &project_root);
                        let after_heuristic = assembler.estimate_tokens(&heuristic.entries);
                        let threshold = model_profile::ModelProfile::for_model(client.model())
                            .compaction_threshold();

                        let result = if after_heuristic > threshold {
                            match compaction::llm_compact(&heuristic.entries, client.model(), &client)
                                .await
                            {
                                Ok(llm_result) if llm_result.compacted => llm_result,
                                Ok(llm_result) => compaction::CompactionResult {
                                    llm_usage: llm_result.llm_usage,
                                    ..heuristic
                                },
                                Err(e) => {
                                    eprintln!(
                                        "{}",
                                        format!("  warning: LLM compaction failed: {e}").yellow()
                                    );
                                    heuristic
                                }
                            }
                        } else {
                            heuristic
                        };

                        if let Some(u) = &result.llm_usage {
                            session_meta.cumulative_input_tokens += u.input_tokens;
                            session_meta.cumulative_output_tokens += u.output_tokens;
                            let _ = session_meta.save(&sessions_dir);
                        }

                        if result.compacted {
                            let new_estimate = assembler.estimate_tokens(&result.entries);
                            let new_count = result.entries.len();
                            let llm_cost = result.llm_usage.as_ref().and_then(|u| {
                                let profile =
                                    model_profile::ModelProfile::for_model(client.model());
                                profile.estimate_cost_from_usage(u)
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
                                let profile =
                                    model_profile::ModelProfile::for_model(client.model());
                                profile.estimate_cost_from_usage(u)
                            });
                            if let Some(cost) = llm_cost {
                                println!(
                                    "{}",
                                    format!(
                                        "  nothing to compact (LLM summarization attempt: ~${cost:.4})"
                                    )
                                    .yellow()
                                );
                            } else {
                                println!("{}", "  nothing to compact".dimmed());
                            }
                        }
                    }
                },
                None => {
                    let cmd = input.split_whitespace().next().unwrap_or(&input);
                    println!("{}", format!("  unknown command: {cmd}").dimmed());
                    println!(
                        "{}",
                        "  type / and press Tab for available commands".dimmed()
                    );
                }
            }
            continue;
        }

        rl.add_history_entry(&input)?;

        // Append user message to history and transcript
        let user_entry = TranscriptEntry::user_message(&input);
        history.push(user_entry.clone());
        transcript.append(&user_entry)?;

        // Update last_active on every turn (not just successful API calls)
        session_meta.last_active = Utc::now();
        let _ = session_meta.save(&sessions_dir);

        // Preflight budget check: compact if estimated tokens exceed threshold
        if let Some(result) =
            compaction::maybe_compact(&history, &assembler, client.model(), &project_root, &client)
                .await
        {
            // Always account for LLM usage even if compaction didn't reduce size
            if let Some(u) = &result.llm_usage {
                term_renderer.record_usage(client.model(), u);
                session_meta.cumulative_input_tokens += u.input_tokens;
                session_meta.cumulative_output_tokens += u.output_tokens;
                let _ = session_meta.save(&sessions_dir);
            }
            if result.compacted {
                let old_estimate = assembler.estimate_tokens(&history);
                let new_estimate = assembler.estimate_tokens(&result.entries);
                let threshold =
                    model_profile::ModelProfile::for_model(client.model()).compaction_threshold();
                let llm_cost = result.llm_usage.as_ref().and_then(|u| {
                    let profile = model_profile::ModelProfile::for_model(client.model());
                    profile.estimate_cost_from_usage(u)
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
        term_renderer.begin_turn();
        term_renderer.print_streaming_indicator();

        let agent = Agent::new(&client, &project_root);
        let run_result = agent
            .run(
                api_input,
                &mut term_renderer,
                &mut |name: &str, args: &str| session_perms.authorize(name, args),
                &mut || {
                    // Refresh repo context and file index after mutating tools
                    file_idx.refresh();
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
            .await;

        // Flush any partial line remaining in the markdown buffer
        term_renderer.flush_line_buffer();

        match run_result {
            Ok(result) => {
                if result.text.is_empty() {
                    println!("{}", "(no response from model)".dimmed());
                }
                println!();

                // Update session metadata and display stats
                if let Some(usage) = &result.usage {
                    term_renderer.record_usage(client.model(), usage);
                    term_renderer.print_turn_stats(client.model(), usage);
                    session_meta.cumulative_input_tokens += usage.input_tokens;
                    session_meta.cumulative_output_tokens += usage.output_tokens;
                }
                term_renderer.print_status_line(
                    client.model(),
                    permission_mode.short_name(),
                );
                session_meta.last_active = Utc::now();
                // Best-effort metadata save — don't fail the session on metadata errors
                let _ = session_meta.save(&sessions_dir);
            }
            Err(e) => {
                // Check if the provider rejected the model (404, model-not-found)
                if e.downcast_ref::<api::ModelRejected>().is_some() {
                    let fallback = "grok-3-fast";
                    eprintln!(
                        "\n{}",
                        format!(
                            "  model '{}' rejected by provider — falling back to {fallback}",
                            client.model()
                        )
                        .yellow()
                    );
                    client.set_model(fallback.to_string());
                    session_meta.model = fallback.to_string();
                    let _ = session_meta.save(&sessions_dir);
                    eprintln!("{}", "  please re-send your message.\n".yellow());
                } else {
                    eprintln!("\n{} {e}\n", "error:".red().bold());
                }
            }
        }
    }

    println!("{}", "\ngoodbye.".dimmed());
    Ok(())
}

