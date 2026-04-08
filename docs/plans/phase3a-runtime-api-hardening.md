# Plan: Phase 3A — Runtime & API Hardening

> Source PRD: `docs/prd/phase3-foundation-upgrades.md` (Phase 3A section)

## Architectural decisions

Durable decisions that apply across all phases:

- **ModelProfile struct**: Expands with capability booleans (`supports_reasoning_effort_control`, `returns_plaintext_reasoning`, `returns_encrypted_reasoning`, `supports_tools`) and `cached_input_price: f64`. Model resolution uses canonical aliases with pattern/prefix matching — not a rigid name table. Unknown models get sensible fallback defaults (no reasoning, standard pricing, conservative context limit).
- **TranscriptEntry::AssistantMessage**: Gains two optional reasoning fields: `reasoning_content: Option<String>` (plaintext, for grok-3-mini) and `encrypted_reasoning: Option<String>` (opaque blob, for grok-4). Both default to `None` and must deserialize gracefully from old transcripts where they are absent.
- **CompactionSummary handling**: No longer emitted as a separate `{"role": "system"}` message. The context assembler extracts CompactionSummary entries from the transcript and appends the most recent summary to the system prompt content with a `---` separator and `[Context summary from earlier in this conversation]` header. Result: exactly one system message at position 0.
- **TurnResponse struct**: Expands with `reasoning_content: Option<String>` and `encrypted_reasoning: Option<String>` so the agent loop can persist reasoning payloads to the transcript separately from visible content.
- **Usage struct**: Expands with `cached_input_tokens: Option<u64>` and `reasoning_tokens: Option<u64>`. Reasoning tokens are billed at the output token rate.
- **Token estimation**: AssistantMessage `token_estimate()` must account for reasoning payload size (plaintext or encrypted) in addition to visible content. This ensures compaction thresholds and `/status` context reporting remain accurate when reasoning payloads are retained.
- **API request shape**: Gains optional `reasoning: { effort }` object (only for models with `supports_reasoning_effort_control`), `store: bool`, and headers `x-grok-conv-id` / `prompt_cache_key` (both keyed to the session UUID).
- **file_edit tool schema**: Gains optional `replace_all: bool` parameter (default false). When true, replaces all non-overlapping occurrences left-to-right and reports the count.
- **Reasoning payload preservation**: Every retained assistant turn in the transcript must preserve its reasoning payload (plaintext or encrypted). Payloads are dropped only when their containing turn is compacted away — never stripped from a retained turn.
- **Tool execution contract**: Tool functions return a typed `ToolOutcome { success: bool, output: String }` rather than a bare string. Checkpoint emission is driven by comparing pre and post git blob hashes — a checkpoint is emitted whenever the file was actually mutated on disk (`post_hash != pre_hash`), regardless of whether the tool reported success. This ensures partial writes that modify a file before erroring still produce a restore point. Permission denials and validation errors that never touch disk naturally produce matching hashes and are excluded. The `success` field remains available for display and future use.
- **Compaction summary token accounting**: When a CompactionSummary is inlined into the system prompt, its tokens are counted in `system_overhead_estimate` (via `set_system_prompt`). The CompactionSummary entry remains in the persisted transcript for history but is excluded from `estimate_tokens`'s transcript sum to prevent double-counting. This is validated by tests across repeated compaction cycles.
- **Model capability enforcement**: Models with `supports_tools=false` are rejected at `/model` selection time and during default model resolution — not at first API call. Provider-side model rejection errors (404, model-not-found) trigger the fallback to `grok-3-fast`, same as local resolution failures.

---

## Phase 1: Bug Fixes & Lint Compliance

**User stories**: 1, 2, 3 + engineering acceptance criteria

### What to build

Fix three correctness bugs and bring the codebase to clean lint/format status.

**Compaction wire format (story 1):** Change the context assembler so that `CompactionSummary` entries are no longer emitted as separate system messages. Instead, extract the most recent CompactionSummary from the transcript, append it to the system prompt content with a `---` separator, and emit a single system message. When multiple CompactionSummary entries exist, only the most recent is used (earlier summaries were already incorporated by the compaction process).

**shell_exec cwd resolution (story 2):** When the model passes a relative `cwd` value, join it with `project_root` before use. Canonicalize the resolved path and validate it doesn't escape the project root (same containment check as file tools). Absolute `cwd` paths continue to work but must also pass containment.

**Checkpoint emission on failure (story 3):** Refactor tool execution to return a typed `ToolOutcome { success: bool, output: String }` instead of a bare string. Checkpoint emission is driven by comparing pre and post git blob hashes: a checkpoint is emitted whenever the file was actually mutated on disk (`post_hash != pre_hash`), regardless of the tool's success flag. This ensures partial writes that modify a file before erroring still produce a restore point for `/undo`. Permission denials and validation errors that never touch disk naturally produce matching hashes and are excluded without special-casing.

**Compaction summary token accounting:** When the CompactionSummary moves into the system prompt, update the token accounting so that: (1) `set_system_prompt` recalculates `system_overhead_estimate` including the inlined summary, and (2) `estimate_tokens` excludes CompactionSummary entries from the transcript sum since they are no longer emitted as separate messages. This prevents double-counting and ensures compaction thresholds remain accurate across repeated compaction cycles.

**Lint compliance:** Fix all `cargo clippy --all-targets --all-features -- -D warnings` and `cargo fmt --check` failures.

### Acceptance criteria

- [ ] `cargo clippy --all-targets --all-features -- -D warnings` passes
- [ ] `cargo fmt --check` passes
- [ ] `build_messages` with a CompactionSummary produces exactly one system message with the summary appended after a `---` separator
- [ ] Multiple CompactionSummary entries: only the most recent appears in the system message
- [ ] Message ordering: system prompt (with summary) -> conversation history (no CompactionSummary items in the history portion)
- [ ] shell_exec with relative `cwd: "src"` resolves to `{project_root}/src`
- [ ] shell_exec with absolute `cwd` inside project root works
- [ ] shell_exec with `cwd` escaping project root returns an error
- [ ] Tool execution returns a typed `ToolOutcome { success, output }` — not a bare string
- [ ] Tool calls that mutate a file produce checkpoint entries (regardless of success/failure)
- [ ] Tool calls that fail without mutating disk do not produce checkpoint entries
- [ ] Permission-denied tool calls do not produce checkpoint entries
- [ ] Compaction summary tokens are counted in system overhead, not in the transcript sum
- [ ] Repeated compaction cycles do not double-count or omit summary tokens
- [ ] Pre-existing tests continue to pass
- [ ] Existing session transcripts remain loadable

---

## Phase 2: Model Catalog & Capability Profiles

**User stories**: 4, 5

### What to build

Expand ModelProfile from a price/context struct into a capability-aware profile system. Add grok-4 model families and update the default model.

**Capability flags:** Add boolean fields to ModelProfile: `supports_reasoning_effort_control`, `returns_plaintext_reasoning`, `returns_encrypted_reasoning`, `supports_tools`. Add `cached_input_price: f64`.

**Model families:**
- `grok-3-mini` family: `supports_reasoning_effort_control=true`, `returns_plaintext_reasoning=true`, ~131K context, $0.30/$0.50 input/output.
- `grok-3` family (including `grok-3-fast`): no reasoning, ~131K context, $3.00/$15.00.
- `grok-4` reasoning family (names matching `grok-4*reasoning`): `returns_encrypted_reasoning=true`, ~2M context, $2.00/$6.00.
- `grok-4` non-reasoning family (names matching `grok-4*non-reasoning`): no reasoning, ~2M context.
- `grok-4.20-multi-agent` family (names matching `grok-4*multi-agent`): `returns_encrypted_reasoning=true`, `supports_reasoning_effort_control=true`, ~2M context. This is the only grok-4 variant that supports effort control.

**Pattern matching:** Model name resolution uses prefix/pattern matching so date suffix variations (e.g., `-0309`, `-0415`) resolve correctly without code changes.

**Fallback:** Unknown models get sensible defaults (no reasoning, standard pricing, conservative context limit) rather than errors.

**Capability enforcement:** All known model families must have `supports_tools` explicitly assigned. Models with `supports_tools=false` are rejected at `/model` selection time with a clear message ("model X does not support tool use"). Default model resolution also enforces this — a default candidate that lacks tool support is skipped. Provider-side model rejection errors (404, model-not-found, model-not-supported) trigger the same fallback path as local resolution failures.

**Default model:** Change from `grok-3-fast` to `grok-4-1-fast-reasoning`. Fallback to `grok-3-fast` triggers on model resolution failure (unknown model, not found, not supported errors) or provider-side model rejection — not on generic API errors like rate limits.

**Compaction thresholds:** Recalculate for 2M context windows with appropriate effective ceilings.

### Acceptance criteria

- [x] All grok-3, grok-3-mini, grok-4 reasoning, grok-4 non-reasoning, and grok-4 multi-agent families resolve to correct profiles
- [x] Capability flags are accurate for each model family
- [x] Pricing fields (including `cached_input_price`) are set for all known models
- [x] Date suffix variants (e.g., `grok-4-1-fast-reasoning-0415`) resolve correctly via pattern matching
- [x] `supports_tools` is explicitly assigned for every known model family
- [x] `/model` rejects models with `supports_tools=false` with a clear error message
- [x] Unknown model names return a fallback profile (no reasoning, conservative limits) — not an error
- [x] Default model is `grok-4-1-fast-reasoning`
- [x] Provider-side model rejection (404/not-found) triggers fallback to `grok-3-fast`
- [x] Compaction thresholds are recalculated for 2M context models
- [x] `/model <name>` works for new model names
- [x] `/status` displays the correct model profile info

---

## Phase 3: Reasoning Support

**User stories**: 6, 7

### What to build

Wire up reasoning/thinking support end-to-end, driven by the active model's capability flags from Phase 2.

**Request construction:** For models with `supports_reasoning_effort_control` (grok-3-mini, grok-4.20-multi-agent): add `reasoning: { effort: "low" | "high" }` to API requests. Exclude parameters incompatible with reasoning (`presencePenalty`, `frequencyPenalty`, `stop`) when a reasoning model is active.

**Response parsing:** Expand TurnResponse with `reasoning_content: Option<String>` and `encrypted_reasoning: Option<String>` so reasoning payloads are carried separately from visible content. Parse plaintext `reasoning_content` from grok-3-mini responses. For grok-4 reasoning models, request encrypted reasoning via `include: ["reasoning.encrypted_content"]`. Parse `reasoning_tokens` from `usage.output_tokens_details.reasoning_tokens` and include in cost calculation at the output token rate.

**Transcript storage:** Add two optional fields to the AssistantMessage transcript entry: `reasoning_content: Option<String>` (plaintext, grok-3-mini) and `encrypted_reasoning: Option<String>` (opaque blob, grok-4). The agent loop persists whichever field TurnResponse carries. Old transcripts without these fields must load gracefully (both default to `None`).

**Round-tripping:** The context assembler must include reasoning payloads when rebuilding assistant messages for the API — plaintext `reasoning_content` for grok-3-mini turns, encrypted reasoning for grok-4 turns. Every retained assistant turn must preserve its reasoning payload when resent. Payloads are dropped only when compacted away.

**Token estimation:** Update AssistantMessage's `token_estimate()` to include the size of any reasoning payload (plaintext or encrypted) in addition to visible content. This keeps compaction thresholds and `/status` context reporting accurate when reasoning payloads are retained across turns.

**User controls:** Add `/think` slash command to cycle reasoning effort: off -> low -> high -> off. Disable `/think` with an explanatory message on models that don't support it. Add `--think` CLI flag for initial effort setting.

**Display (3A scope):** Plaintext reasoning is printed as-is. Encrypted reasoning shows a one-line `[thinking... N tokens]` indicator. All display-heavy UI (collapsible blocks, dimmed rendering) is deferred to Phase 3B.

### Acceptance criteria

- [ ] `/think` cycles effort on models that support it; shows an explanatory message on models that don't
- [ ] `--think` CLI flag sets initial reasoning effort
- [ ] API requests include `reasoning: { effort }` when appropriate
- [ ] Incompatible parameters are excluded for reasoning models
- [ ] Plaintext `reasoning_content` is parsed and displayed from grok-3-mini responses
- [ ] Encrypted reasoning is requested via `include` for grok-4 reasoning models
- [ ] Encrypted reasoning is stored in the transcript and round-tripped in subsequent requests
- [ ] `reasoning_tokens` are parsed from `usage.output_tokens_details.reasoning_tokens`
- [ ] Cost calculation includes reasoning tokens at the output token rate
- [ ] Old transcripts without reasoning fields load without error
- [ ] Plaintext reasoning from grok-3-mini is round-tripped in subsequent API requests
- [ ] Token estimation for assistant messages includes reasoning payload size
- [ ] Encrypted reasoning displays `[thinking... N tokens]` indicator

---

## Phase 4: Cache Optimization & Privacy

**User stories**: 8, 9, 10

### What to build

Add cache optimization headers, surface cached token savings, and add privacy controls.

**Cache headers:** Send `x-grok-conv-id` header on every API request, using the session UUID as the conversation identifier. Send `prompt_cache_key` using the same session UUID in Responses API requests.

**Cached token display:** Parse cached token counts from the API usage response. Expand the Usage struct with `cached_input_tokens`. Display in the per-turn cost line: `tokens: 1.2k in (800 cached) / 340 out (~$0.0012)`. Use `cached_input_price` from the model profile for cost calculation on cached tokens.

**store: false support:** Add `--no-store` CLI flag and `GROX_NO_STORE=1` environment variable. When enabled, send `store: false` in API requests. Update the existing startup warning about 30-day storage to mention the `--no-store` option.

### Acceptance criteria

- [ ] `x-grok-conv-id` header is set to the session UUID on every API request
- [ ] `prompt_cache_key` is sent in Responses API requests
- [ ] Cached token counts are parsed from the API response
- [ ] Cost display shows cached tokens: `tokens: 1.2k in (800 cached) / 340 out (~$0.0012)`
- [ ] Cost calculation uses `cached_input_price` for cached tokens
- [ ] `--no-store` flag causes `store: false` in API requests
- [ ] `GROX_NO_STORE=1` environment variable works equivalently
- [ ] Startup warning mentions `--no-store` option

---

## Phase 5: file_edit replace_all & Contract Fixtures

**User stories**: 11, 12

### What to build

**replace_all:** Add optional `replace_all` boolean parameter to the file_edit tool schema (default: false). When true, replace all occurrences of `old_string` with `new_string` using non-overlapping left-to-right matching (consistent with `str::replace` semantics). Return message includes the count: "Replaced N occurrences in path/to/file". When false or omitted, behavior is unchanged: exactly one match required, error on zero or multiple matches. Undo uses the existing blob-snapshot mechanism — the file is snapshotted before the edit and restored as a whole.

**Contract test fixtures:** First, extract the SSE event parsing logic from `GrokClient::send_turn` into a standalone, testable function (e.g., `parse_sse_event`) that can be called independently of the HTTP client. Then record 2-3 real xAI Responses API SSE responses (happy path with text + tool call, reasoning model response, error response). Sanitize fixtures: strip API keys, user-identifying content, and non-deterministic fields. Store as fixture files alongside existing tests. Write contract tests that feed these fixtures through the extracted parser to verify it handles the actual wire format.

### Acceptance criteria

- [ ] `replace_all: true` replaces all occurrences and returns "Replaced N occurrences in path/to/file"
- [ ] `replace_all: false` (or omitted) preserves existing single-match behavior
- [ ] `replace_all: true` with zero matches returns an error
- [ ] Non-overlapping left-to-right semantics: replacing "aa" in "aaa" yields one replacement
- [ ] Undo restores the full file from the pre-edit snapshot
- [ ] SSE event parsing logic is extracted into a standalone testable function
- [ ] At least 2 sanitized SSE fixture files exist (happy path, reasoning response)
- [ ] Contract tests feed fixtures through the extracted parser and verify extracted text, tool calls, usage, and reasoning fields

---

## Phase 6: README Rewrite

**User stories**: 13, 14

### What to build

Complete replacement of README.md to match the current feature set after all Phase 3A changes have landed.

**Sections:** Project description, install (`cargo install grox-cli`), quickstart (API key setup + first session in under 60 seconds), features overview, tool list with descriptions, permission modes table, slash commands reference, configuration (GROX.md, env vars, `--no-store`), known limitations, contributing.

No architecture deep-dive in README — that belongs in docs/.

### Acceptance criteria

- [ ] README accurately describes all 6+ tools (including replace_all)
- [ ] Quickstart gets a user from install to working session
- [ ] Permission modes table covers all modes (Default, Trust, ReadOnly, Yolo)
- [ ] Slash commands reference includes `/think`, `/model`, `/undo`, `/compact`, `/sessions`, `/resume`, `/status`
- [ ] Configuration section covers GROX.md, env vars, and `--no-store`
- [ ] Known limitations section exists
- [ ] No references to stale features or incorrect module counts
