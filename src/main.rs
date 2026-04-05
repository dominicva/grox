mod agent;
mod api;
mod tools;

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

    let system_prompt = json!({
        "role": "system",
        "content": "You are Grox, a coding assistant powered by Grok. Be concise and helpful."
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

        // Build input array for Responses API
        let mut api_input = vec![system_prompt.clone()];
        api_input.push(json!({
            "role": "user",
            "content": input,
        }));

        print!("\n{} ", "grok:".magenta().bold());
        stdout().flush()?;

        use api::GrokApi;
        match client
            .send_turn(
                api_input,
                &[],
                previous_response_id.as_deref(),
                &mut |token: String| {
                    print!("{token}");
                    let _ = stdout().flush();
                },
            )
            .await
        {
            Ok(response) => {
                println!("\n");
                previous_response_id = response.response_id;
            }
            Err(e) => {
                eprintln!("\n{} {e}\n", "error:".red().bold());
            }
        }
    }

    println!("{}", "\ngoodbye.".dimmed());
    Ok(())
}
