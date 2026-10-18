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
