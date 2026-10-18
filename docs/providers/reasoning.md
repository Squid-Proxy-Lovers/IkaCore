# Provider reasoning replay

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

Sources: the OpenAI Responses references above and the existing Codex endpoint
request contract in codex_responses.py. Endpoint compatibility is checked separately
by the bounded live replay probe when local login and network are available.

## Anthropic extended and adaptive thinking

Opt in with ProviderConfig(thinking_budget_tokens=1024, max_tokens=4096).
BareBoneModel accepts the same thinking_budget_tokens option; per-model overrides
can set it too on models that support manual thinking. The default None preserves
the model default.
Budgets must be integers at least 1024 and strictly less than the effective
max_tokens, including the existing Haiku cap. Invalid values raise before any
network call.

When enabled, requests send thinking: {type: "enabled", budget_tokens: N}, omit
temperature (top_k/top_p are not exposed), and adjust forced/required tool choices
to auto. Tool requirements and completion remain enforced by the agent loop.
This adjustment includes the framework's forced control tools. Supported modern
models may prefer adaptive thinking; this option explicitly selects manual mode.
Opus 4.7/4.8 reject manual budgets locally. Those models also omit temperature
in ordinary requests and summaries. Metadata enables adaptive thinking on
Opus 4.6/4.7/4.8 and Sonnet 4.6: set BareBoneModel.deepthinking = True and,
optionally, reasoning_effort to populate output_config.effort. Adaptive thinking
preserves forced/required tool choices, unlike manual thinking.

Complete assistant blocks, including thinking with its signature and
redacted_thinking with its opaque data, are replayed verbatim in their original
positions before tool results. Signed blocks survive JSON history persistence.
Only readable thinking text is returned as reasoning_content.

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

Gemini 2.5 Pro keeps its dynamic thinking default: it cannot accept a zero
thinking budget. Only the metadata-listed 2.5 Flash families receive the existing
zero budget. Usage includes thought tokens in output and total accounting.

Sources: [Gemini generateContent thought signatures](https://ai.google.dev/gemini-api/docs/generate-content/thought-signatures),
[per-model budgets](https://ai.google.dev/gemini-api/docs/generate-content/thinking),
[usage metadata](https://ai.google.dev/api/generate-content#UsageMetadata).

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

## Validation and limits

Offline native-response and fresh-process snapshot tests cover all seven reasoning routes.
Run python scripts/live_replay_repro.py for the offline replay probe. A live result
validates only its configured provider, model and account. See the
[release validation](../v2/validation/release-2.0.0.md) for reproduced checks and live limits.
