# grox

Agentic coding with Grok.

Grox is a CLI coding agent powered by [xAI's Grok models](https://x.ai). It reads your codebase, runs commands, and writes code — designed to build understanding before making changes.

## Install

```
cargo install grox-cli
```

Requires [ripgrep](https://github.com/BurntSushi/ripgrep) (`rg`) on your PATH for the grep tool.

## Quickstart

1. Get an API key from [console.x.ai](https://console.x.ai)
2. Set it:
   ```
   export XAI_API_KEY=xai-...
   ```
3. Run grox in any project directory:
   ```
   cd your-project
   grox
   ```

That's it. Grox detects the project root (via `.git`, `Cargo.toml`, `package.json`, etc.), reads a `GROX.md` if present, and starts an interactive session.

## Features

- **Turn-based agent loop** — sends context to the API, executes tool calls sequentially, returns results, repeats (max 25 turns per message)
- **Session persistence** — conversations are saved and can be resumed with `/resume`
- **Context compaction** — automatic summarization when the context window fills up
- **Reasoning support** — plaintext reasoning (grok-3-mini) and encrypted reasoning (grok-4) with configurable effort levels
- **Prompt caching** — session-keyed cache headers for reduced latency and cost
- **Checkpoint & undo** — file mutations are tracked via git blob snapshots; `/undo` restores previous states
- **Syntax-highlighted output** — code blocks are highlighted in the terminal

## Tools

| Tool | Description |
|------|-------------|
| **file_read** | Read file contents. Refuses binary files. |
| **file_write** | Create or overwrite files. Creates parent directories automatically. |
| **file_edit** | Replace a string in a file. Requires exactly one match by default; set `replace_all: true` to replace every occurrence. |
| **list_files** | List directory contents. |
| **grep** | Search file contents by regex using ripgrep. Supports glob filters and case-insensitive mode. |
| **shell_exec** | Run shell commands in the project root. Optional `cwd` and `timeout_secs` (default 60, max 300). |

All file tools are path-validated to stay inside the project root.

## Permission modes

| Mode | Flag | Reads | Writes | Shell | Destructive |
|------|------|-------|--------|-------|-------------|
| **Default** | *(none)* | Auto | Prompt | Prompt | Always prompt |
| **Trust** | `--auto-approve-writes` | Auto | Auto (in project) | Prompt | Always prompt |
| **Read-only** | `--read-only` | Auto | Denied | Denied | Denied |
| **Yolo** | `--yolo` | Auto | Auto | Auto | Auto |

Destructive commands (`rm -rf`, `git push --force`, `DROP TABLE`, etc.) always require confirmation in Default and Trust modes.

## Slash commands

| Command | Description |
|---------|-------------|
| `/think` | Cycle reasoning effort: off &rarr; low &rarr; high &rarr; off. Disabled with a message on models that don't support it. |
| `/model <name>` | Switch to a different model. |
| `/status` | Show model, capabilities, context usage, compaction threshold, tools, and permissions. |
| `/undo [N] [--code\|--conversation\|--both]` | Undo turns. Restores files from checkpoints. |
| `/compact` | Manually trigger context compaction. |
| `/sessions` | List recent sessions for the current project. |
| `/resume [id]` | Resume a previous session by ID prefix. |
| `/quit` | Exit the session. |

## Configuration

### GROX.md

Add a `GROX.md` file in your project root to give the agent custom instructions. It gets appended to the system prompt as a "Project instructions" section. Truncated at 10,000 characters.

### Environment variables

| Variable | Description |
|----------|-------------|
| `XAI_API_KEY` | **(required)** xAI API key |
| `GROX_MODEL` | Default model name (overridden by `--model`) |
| `GROX_NO_STORE` | Set to `1` to prevent conversation storage with the provider |

### CLI flags

```
grox [OPTIONS]

Options:
  --model <NAME>          Override the default model
  --think [low|high]      Set initial reasoning effort
  --no-store              Don't store conversations with the provider
  --auto-approve-writes   Trust mode: auto-approve writes inside the project
  --read-only             Deny all writes and shell execution
  --yolo                  Auto-approve everything, no guardrails
  --resume [SESSION_ID]   Resume a session (defaults to most recent)
  --verbose               Print raw SSE events to stderr
```

## Model support

Grox supports the following model families. Date suffix variants (e.g., `-0309`, `-0415`) resolve automatically via pattern matching.

| Family | Context | Reasoning | Default |
|--------|---------|-----------|---------|
| grok-4-1-fast-reasoning | 2.1M | Encrypted | **Yes** |
| grok-4 reasoning variants | 2.1M | Encrypted | |
| grok-4 multi-agent | 2.1M | Encrypted + effort control | |
| grok-4 non-reasoning | 2.1M | None | |
| grok-3-mini | 131K | Plaintext + effort control | |
| grok-3 / grok-3-fast | 131K | None | |

Unknown model names get a fallback profile (no reasoning, conservative limits). If the provider rejects the requested model, grox falls back to `grok-3-fast`.

## Known limitations

- **ripgrep required** — the grep tool calls `rg` and errors if it's not installed
- **No parallel tool calls** — tools execute sequentially within each turn
- **Binary files refused** — file_read and file_edit detect binary content (null bytes in the first 8KB) and refuse to operate
- **Output clipping** — tool outputs over 30KB are truncated before being sent to the model
- **Context ceiling** — compaction triggers at ~60% of the model's context window (e.g., ~1.2M tokens for grok-4, ~80K for grok-3)
- **GROX.md size cap** — custom instructions are truncated at 10,000 characters

## Contributing

Contributions are welcome. The codebase is 14 flat Rust modules under `src/` — no workspace, no proc macros.

```
cargo test            # run all tests
cargo clippy          # lint
cargo fmt --check     # format check
```
