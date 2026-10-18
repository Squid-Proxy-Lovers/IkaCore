# Provider reasoning replay (v2.0.0 unreleased)

## OpenAI Responses

Responses reasoning items now survive tool loops, subsequent steps, and snapshot
save/resume. The JSON history envelope is
provider_data: {provider: "responses", items: [...]} on its owning assistant
turn. Items retain their original order, IDs, summaries and encrypted content;
display reasoning remains summary text only. Complete native output messages and
function calls retain their positions relative to reasoning. Final-answer reasoning
is also persisted. Historical reasoning that was discarded by older versions
cannot be reconstructed.

Requests use store: false; reasoning models (or an explicit reasoning effort)
request include: ["reasoning.encrypted_content"]. Non-reasoning OpenAI benchmark
fixtures are unchanged. The existing four-value parser API remains supported:
low-level users should pass their history as the optional fourth argument to
parse_provider_response(provider, data, model_id, message_history) before using
the tool-message appender. The normal sync/async chat paths do this automatically.

Opaque data is never added to display content, tool arguments or CLI output.
Snapshots and explicit JSON request/history debug files may persist it.

Sources: [reasoning and stateless replay](https://developers.openai.com/api/docs/guides/reasoning),
[Responses item schema and completed SSE items](https://developers.openai.com/api/reference/resources/responses/methods/retrieve).

## Codex

Codex shares the Responses history envelope. It always sends stream: true,
store: false and include: ["reasoning.encrypted_content"], without unsupported
max_output_tokens or temperature fields. Completed SSE output items carry the
cipher; output_item.added may contain incomplete data and is never used for replay.
Malformed SSE data is no longer echoed into debug output.

The 21 benchmark checksum checks remain in place. Only the three Codex reasoning
model request fixtures change: codex:10:0, codex:100:10 and codex:1000:10. In every
case the sole difference is the required encrypted-reasoning include. Removing
that field reproduces each original SHA-256 exactly. The other 18 checksums,
including the three non-reasoning OpenAI Responses fixtures, remain unchanged.

Sources: the OpenAI Responses references above and the existing Codex endpoint
request contract in codex_responses.py. Endpoint compatibility is checked separately
by the bounded live replay probe when local login and network are available.

## Anthropic manual extended thinking

Opt in with ProviderConfig(thinking_budget_tokens=1024, max_tokens=4096).
BareBoneModel accepts the same thinking_budget_tokens option; per-model overrides
can set it too. The default None preserves ordinary non-thinking behavior.
Budgets must be integers at least 1024 and strictly less than the effective
max_tokens, including the existing Haiku cap. Invalid values raise before any
network call.

When enabled, requests send thinking: {type: "enabled", budget_tokens: N}, omit
temperature (top_k/top_p are not exposed), and adjust forced/required tool choices
to auto. Tool requirements and completion remain enforced by the agent loop.
This adjustment includes the framework's forced control tools. Supported modern
models may prefer adaptive thinking; this option explicitly selects manual mode.

Complete assistant blocks, including thinking with its signature and
redacted_thinking with its opaque data, are replayed verbatim in their original
positions before tool results. Signed blocks survive JSON history persistence.
Only readable thinking text is returned as reasoning_content.

The only Core wiring additions are the ProviderConfig field and the conversion in
AgentChat.get_barebone; clone_for_run already preserves it through dataclasses.replace.

Sources: [manual budget rules](https://platform.claude.com/docs/en/build-with-claude/extended-thinking),
[thinking, tool choice, sampling and block preservation](https://platform.claude.com/docs/en/build-with-claude/thinking).

## Gemini thought signatures

Function-call signatures were already copied, but signed text parts and thought
parts were flattened or lost. Replay now preserves the whole ordered parts list,
keeping signatures attached to the exact original text/functionCall part, including
parallel calls and both accepted signature spellings. Readable thought text is
reasoning_content rather than user-visible content. Final text-part signatures
are persisted too. This uses generateContent; other providers' streaming APIs
are not added by this change.

HTTP debug previews redact opaque encrypted_content, signatures, redacted thinking
data and OpenRouter encrypted reasoning data on a copy. Gemini's per-call debug
message no longer serializes the signed part. Replay/snapshot originals stay intact.

Source: [Gemini generateContent thought signatures](https://ai.google.dev/gemini-api/docs/generate-content/thought-signatures).

## OpenRouter reasoning passthrough

Previously only a flattened reasoning string survived; reasoning_details was lost.
Now the original reasoning, reasoning_content and reasoning_details fields are
replayed on their owning assistant message, including signatures, encryption,
IDs, indexes and formats. Details-only responses are preserved even without
readable reasoning. Final-answer details are persisted and history can reconstruct
the next request after resume.

Explicit reasoning_effort now sends the unified reasoning object with effort and
exclude: false, including tool requests; it takes precedence over the legacy
per-model reasoning exclusion default. Unconfigured routes retain that default.
Do not change the upstream model while replaying its opaque data.

The shared Chat Completions builder now replays persisted turns, which also
preserves the existing Z.AI reasoning_content behavior with a fresh resumed buffer.

Source: [OpenRouter preserving reasoning and reasoning_details schema](https://openrouter.ai/docs/guides/best-practices/reasoning-tokens).

## DeepSeek verification

The current official thinking-mode contract supersedes older R1-era advice:
requests carrying tools must replay complete reasoning_content from all prior
assistant turns, even turns without a tool call. Without tools, the field is
optional and ignored by the API. The existing adapter already follows this
contract, so its payload policy is deliberately retained. Sync/async two-request
goldens and fresh-buffer replay tests cover both tool-turn and final-turn thoughts.

Sources: [DeepSeek thinking mode](https://api-docs.deepseek.com/guides/thinking_mode/),
[Chat Completions assistant message schema](https://api-docs.deepseek.com/api/create-chat-completion/).

## Z.AI GLM verification

Z.AI continues to replay the original unmodified reasoning_content on assistant
turns, including the Coding Plan endpoint where preserved thinking is enabled by
default. The shared persisted-history reconstruction additionally supports fresh
live buffers after resume. Sync/async payload goldens protect this behavior.

The standard API endpoint defaults to clearing cross-turn reasoning; callers using
that endpoint who need Preserved Thinking must select clear_thinking: false in
their own request customization. This change does not alter the existing Coding
Plan default or add a standard-endpoint configuration API.

Source: [Z.AI thinking mode and preserved thinking](https://docs.z.ai/guides/capabilities/thinking-mode).

## Snapshot and live validation

A parametrized public-API regression interrupts an ask_user tool turn, closes the
agent, opens a distinct agent/store instance, and resumes from the durable SQLite
snapshot. All seven provider routes must replay the original reasoning/signatures
on the next request. Opaque values remain out of CLI output.

The offline scripts/live_replay_repro.py now checks Codex, OpenAI Responses,
Anthropic, DeepSeek, Gemini, OpenRouter and Z.AI, including encrypted/signed fields,
and exits nonzero if any replay fails.

Live Codex gpt-5.6-sol was checked on 2026-10-18 with explicit local-login access
and a 45-second request budget (60-second process alarm, one attempt). The initial
low-effort fixture executed successfully but returned no reasoning item, so it was
inconclusive. A high-effort CRT arithmetic fixture with a harmless local multiplier
tool passed: one encrypted reasoning item preceded the tool, the exact native
item was present in the follow-up input, and the backend accepted it and returned
the correct final product. One tool execution; 596 input + 186 output = 782 tokens.
SSE completion events also returned final-answer reasoning. Only counts, booleans,
event types and usage were recorded; no bearer or encrypted data was printed.

The high-effort probe used reasoning.summary: "auto" solely to observe the live
case; normal library requests do not require a readable summary. Its report is
artifacts/live/reasoning-replay-codex-high.json. This direct runtime probe is the
equivalent reasoning_replay case: the existing live harness only exposes the
reasoning_tools case and otherwise requires an installed wheel.

Codex SSE collection also merges the completed envelope with indexed done events,
keeping items absent from individual events and restoring original output order.
Completed done-event data wins over envelope data for the same item.

## Historical reasoning-branch quality gate (`01249f6`)

After merging ikacore-v2 at aca8c53 into v2-reasoning (merge 01249f6), both
requested gates pass on 2026-10-18:

- python -m pytest -q: 1032 passed, 1 skipped in 16.82 seconds.
- python scripts/quality.py: PASS; Ruff clean, Pyright zero errors, source compile
  checks, 1032 passed / 1 skipped under coverage in 27.74 seconds, all 16 area
  floors intact across 123 modules, wheel package-data and installed import checks.
- Combined statement/branch coverage: 94.82% (the 91% floor is unchanged).
  Separate line coverage is 96.53%; branch coverage is 88.68%.
- All 21 synthetic provider payload SHA-256 checks pass. Only the three Codex
  request hashes described above changed for the encryption include.
- Public snapshot resume coverage passes for all seven reasoning provider routes.

The optional real-provider smoke test is the one normal skip; the bounded live
Codex reasoning probe above was run explicitly. Test servers require loopback
access, so the final gate ran with sandbox escalation. No tests or floors were
disabled, and no production behavior was changed to accommodate the live harness:
the completed grouped-config port was brought in by the requested merge.

The later hardening/dogfood integration is measured separately at `8efd658` in
[merge-validation-before-ergonomics.md](merge-validation-before-ergonomics.md); its higher coverage floors supersede
the historical floors above. The live Codex probe was not rerun for that integration.

The subsequent ergonomics integration uses canonical `stages=`, multi-turn `send()`,
restored completed conversation/accounting and compaction events without discarding native
reasoning/tool exchanges. Current integrated gates and live results are labeled in
[merge-validation.md](merge-validation.md).

## Integrated ergonomics measurements

Validated runtime/test source revision: **`f8c6c1a2105efc6bb2972b075b4fc4bc51829e1c`**.

The integrated runtime (parents `7c410f3` / `a2b7984`, runtime fingerprint
`72a19681dbe20599ee4d96d945fc697cfa36c6215196a231acc687d975256759`) passes **1,386 tests / 2 opt-in skips**,
**100% lines / 97.1816% branches / 99.3842% combined** with all raised floors retained.
All six live Codex apps pass; the matrix reports **14 passed / 0 failed / 1
reasoning_not_exposed skip**. See [merge-validation.md](merge-validation.md) for current
revision labels, soak metrics, semantic integration checks and live limits.
