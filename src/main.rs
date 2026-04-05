mod api;

use anyhow::{Context, Result};
use api::{GrokClient, Message};
use colored::Colorize;
use rustyline::DefaultEditor;
use std::io::{Write, stdout};

#[tokio::main]
async fn main() -> Result<()> {
    // Load .env file if present (silently ignored if missing)
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

    let mut messages: Vec<Message> = vec![Message {
        role: "system".to_string(),
        content: "You are Grox, a coding assistant powered by Grok. Be concise and helpful."
            .to_string(),
    }];

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

        messages.push(Message {
            role: "user".to_string(),
            content: input,
        });

        print!("\n{} ", "grok:".magenta().bold());
        stdout().flush()?;

        match client
            .stream_chat(&messages, |token| {
                print!("{token}");
                let _ = stdout().flush();
            })
            .await
        {
            Ok(response) => {
                println!("\n");
                messages.push(Message {
                    role: "assistant".to_string(),
                    content: response,
                });
            }
            Err(e) => {
                eprintln!("\n{} {e}\n", "error:".red().bold());
            }
        }
    }

    println!("{}", "\ngoodbye.".dimmed());
    Ok(())
}
