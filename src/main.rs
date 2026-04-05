mod agent;
mod api;
mod tools;

use agent::Agent;
use anyhow::{Context, Result};
use api::GrokClient;
use colored::Colorize;
use rustyline::DefaultEditor;
use serde_json::json;
use std::io::{Write, stdout};

#[tokio::main]
async fn main() -> Result<()> {
    let _ = dotenvy::dotenv();

    let api_key =
        std::env::var("XAI_API_KEY").context("XAI_API_KEY environment variable not set")?;

    let model = std::env::var("GROX_MODEL").unwrap_or_else(|_| "grok-3-fast".to_string());

    println!("{}", "grox — agentic coding with Grok".bold());
    println!(
        "model: {}  |  type {} to exit\n",
        model.cyan(),
        "/quit".dimmed()
    );

    let client = GrokClient::new(api_key, model);
    let agent = Agent::new(&client);

    let system_prompt = json!({
        "role": "system",
        "content": "You are Grox, a coding agent powered by Grok. You help developers understand and work with their codebase.

Rules:
- Be concise and direct. Lead with the answer, not the process.
- Do NOT narrate what you are about to do or explain your tool usage. Just use tools silently and respond with findings.
- Do NOT thank the user for letting you read files — you have autonomous access to tools.
- When exploring a codebase, use list_files and file_read proactively to gather context before responding.
- Keep responses short. Use bullet points over paragraphs. Skip preamble."
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
                &mut |token: String| {
                    print!("{token}");
                    let _ = stdout().flush();
                },
                &mut |name: &str, args: &str| {
                    // Compact one-line display: extract the key param for summary
                    let summary = summarize_tool_call(name, args);
                    println!("  {} {}", format!("▸ {name}").cyan(), summary.dimmed());
                    let _ = stdout().flush();
                },
                &mut |_name: &str, output: &str| {
                    // Show a brief result summary, not the full output
                    let lines: Vec<&str> = output.lines().collect();
                    let summary = if lines.len() > 5 {
                        format!("  {} ({} lines)", "✓".green(), lines.len())
                    } else if output.is_empty() {
                        format!("  {} (empty)", "✓".green())
                    } else {
                        format!("  {} ({} bytes)", "✓".green(), output.len())
                    };
                    println!("{}", summary.dimmed());
                },
            )
            .await
        {
            Ok(result) => {
                println!("\n");
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
        _ => args.to_string(),
    }
}
