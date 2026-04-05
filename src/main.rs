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
        "content": "You are Grox, a coding agent powered by Grok. You have access to tools for reading files. Use them to help the developer understand and work with their codebase. Be concise and helpful."
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
                    print!("\n  {} {}\n", format!("[{name}]").cyan(), args.dimmed());
                    let _ = stdout().flush();
                },
                &mut |_name: &str, output: &str| {
                    let display = if output.len() > 500 {
                        format!("{}...", &output[..500])
                    } else {
                        output.to_string()
                    };
                    println!("{}", display.dimmed());
                    print!("\n{} ", "grok:".magenta().bold());
                    let _ = stdout().flush();
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
