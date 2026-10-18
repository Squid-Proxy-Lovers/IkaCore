# Codex application usage results

Date: **2026-10-18 UTC**. Model: **gpt-5.6-sol**, local ChatGPT login loaded only with
`--use-local-login`. Runtime: fresh non-editable IkaCore 2.0.0 wheel from revision
`aca8c53a57332c0e5ab225628ac7b29bae99d23b`, Python 3.13.15 / HTTPX 0.28.1.
Application implementation: `1f667520889de190ef7848769935bea332b102b7`.
The initial chat draft predates that commit's stale-prompt workaround; the successful rerun
used the committed Python application code. Runtime code did not change between attempts.

Wheel SHA-256: `b864484f9173841568f92a5c6a823b9b29f244771983295e807137f5bd43d226`.
Detailed sanitized checks are in [live-results.json](live-results.json).
No credentials, prompts, raw responses or model answer text are included in these records.

**Latest outcome: six applications passed.** The failed initial chat attempt is retained
below; it was not discarded from the request budget.

| Application / attempt | Outcome | Request boundary invocations | Wall time (s) |
| --- | --- | ---: | ---: |
| research_assistant | passed | 6 | 16.76 |
| code_reviewer | passed | 6 | 14.86 |
| approval_workflow | passed | 11 | 30.99 |
| long_chat | failed; 90-second first-session budget exhausted | 13 | 90.19 |
| parallel_pipeline | passed | 5 | 6.81 |
| provider_switch | passed | 1 | 2.04 |
| long_chat (rerun) | passed | 32 | 115.25 |

**Total: 74 request boundary invocations**, including the failed initial chat and one
expired-deadline probe rejected before dispatch. Actual live HTTP attempts are therefore
at most 74; retries were set to one attempt. Non-Codex providers incurred no live calls.
These are observed application request counts, not provider billing records.

## What passed

- **research_assistant:** synthetic search and fetch tools executed; the separate final
  writer met both citation checks and recommended a pilot. A one-step research stage is
  configured for extension; extension summary invocation is asserted offline. Its live
  extension counter is unavailable and is reported as null rather than inferred.
- **code_reviewer:** both disposable files were read and the structured submission identified
  the empty-input problem. A synthetic lint tool timed out at 5 ms without stopping review;
  1,200-character tool output caps were configured. Offline payload checks verify the cap
  and timeout feedback specifically.
- **approval_workflow:** dependencies blocked preparation at HITL, a process exited after
  saving the question, another process answered YES, and an independent fork answered NO.
  READY and HOLD results were both verified; nothing was published.
- **long_chat:** the corrected app completed twelve turns, verified **11 compaction
  snapshots**, restored its saved post-model turn in a new process, and recalled the
  preference through an IkaMem search tool. The initial draft reached its 90-second
  session bound after 13 invocations. Source investigation also found stale prompt replay
  (DOGFOOD-3). The application now refreshes turn history explicitly and uses a 180-second
  session scope with nested 30-second turn scopes; the successful full demo took 115.25 s.
- **parallel_pipeline:** two-worker fan-out/fan-in retained both cost and risk values, an
  already expired deadline was rejected, and the shared caller-owned HTTPX client remained
  open through clone shutdown. Cooperative SIGINT pre-dispatch cancellation passed offline;
  live in-flight cancellation is not claimed by this application run.
- **provider_switch:** Codex completed its task and consumed its post-model snapshot.
  OpenAI Responses, OpenAI Chat, Anthropic, Gemini, DeepSeek and OpenRouter passed mocked
  route checks with the same agent contract and configuration-only provider selection.

## Reproduce

Follow [clean-wheel setup](README.md), then from the repository root run each module:

```bash
python -m examples.apps.research_assistant.main --live --use-local-login --model gpt-5.6-sol
python -m examples.apps.code_reviewer.main --live --use-local-login --model gpt-5.6-sol
python -m examples.apps.approval_workflow.main --live --use-local-login --model gpt-5.6-sol
python -m examples.apps.long_chat.main --live --use-local-login --model gpt-5.6-sol
python -m examples.apps.parallel_pipeline.main --live --use-local-login --model gpt-5.6-sol
python -m examples.apps.provider_switch.main --live --use-local-login --model gpt-5.6-sol
```

Every invocation defaults to offline without `--live`. The shared session bounds ordinary
apps to 90 seconds and one request attempt; chat sessions use 180 seconds with 30-second
turn bounds. Approval/chat child processes have separate scopes. Individual tool/model
limits also apply. `IKACORE_APP_METRICS=/absolute/path/to/file.jsonl` optionally records only
one boolean request-start row per boundary invocation, including compaction requests.

This is one provider/model/account sample. It does not establish other-provider live
compatibility, in-flight cancellation guarantees, or suitability for untrusted tools.
See [the friction and bug report](../../docs/v2/dogfood-report.md) before adopting v2.
