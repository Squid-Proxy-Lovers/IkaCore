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
