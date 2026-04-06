# grox

Agentic coding with Grok.

## What it does

Grox is a CLI coding agent powered by Grok (xAI). It reads your codebase, runs commands, and writes code. It is designed to build understanding before making changes: reading files, exploring structure, and checking patterns rather than making changes too eagerly.

## Permission modes

Grox has four permission modes that control what the agent can do without asking:

- **Default** (`grox`) — Reads are automatic. Writes and shell commands prompt for approval. Destructive commands always prompt.
- **Trust** (`grox --auto-approve-writes`) — Writes inside the project are auto-approved. Everything else prompts.
- **Read-only** (`grox --read-only`) — The agent can only read files and respond. All writes and shell execution are denied.
- **Yolo** (`grox --yolo`) — Everything is auto-approved, including destructive commands. No guardrails.

## Tools

The agent has five tools:

- **file_read** — Read file contents.
- **file_write** — Create or overwrite files. Path-validated to stay inside the project root.
- **list_files** — Explore directory structure.
- **grep** — Search file contents by regex using ripgrep.
- **shell_exec** — Run shell commands in the project root.

## GROX.md

Add a `GROX.md` file in your project root to give the agent custom instructions. This gets appended to the system prompt as a "Project instructions" section. Truncated at 10K characters.

## Architecture

Seven flat Rust modules:

- **main** — REPL, CLI parsing, display formatting
- **agent** — Turn-based agent loop (max 25 turns per message)
- **api** — xAI Responses API client with SSE streaming
- **tools** — Tool definitions and execution
- **permissions** — Permission modes, destructive command detection, approval flow
- **prompt** — System prompt construction (5 sections + optional GROX.md)
- **util** — Project root detection, path validation, output clipping

The agent loop is synchronous and turn-based: send input to the API with a `previous_response_id`, execute any tool calls sequentially, return results, repeat. The server manages conversation history. Requires ripgrep (`rg`) on PATH for the grep tool.
