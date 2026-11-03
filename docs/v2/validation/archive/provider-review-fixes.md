> Historical development evidence. Counts and API examples apply only to the recorded revision.

# Provider review fixes — v2.0.0 unreleased

Worktree: `/private/tmp/ikacore-w-fix-providers`; branch: `v2-fix-providers`;
review base: `1ad479b7e331eff919175b50eae64438872b86a0`.
All changes are local. No push or external review comment was made.
Changes are confined to `src/IkaModel`, provider tests, and documentation.
The virtual environment is local to this worktree; commands use its `PATH` and `VIRTUAL_ENV`.

Official documentation below was fetched on 2026-10-18. Provider request/response
contracts are distinguished from IkaCore policies such as character limits and
cancellation behavior, which are verified against their HTTPX/Python contracts.
Each new bug regression was run before its fix and failed for the reported behavior.
The cancellation regression was rerun against the original implementation after
adding the required `loopback_network` marker, to establish a real-socket failure.
Baseline and targeted logs are retained as `/private/tmp/ikacore-*-before.log` and
`/private/tmp/ikacore-*-after.log` (the usage combined-suite log excludes a nonexistent
initial test selector; the corrected 14-test run passed).

## Findings

1. **Verified: plain assistant reasoning replay.** Commit `1565fa9`.
   Plain live assistants now receive both `reasoning_content` and `provider_data`
   from their matching persisted turn. DeepSeek reconciles those turns before
   appending history and live messages. Repeated equal-text turns retain their
   distinct reasoning once, without mutating either input. Regression:
   `test_provider_review_replay.py`, four cases failed before the fix; 17 targeted
   replay/provider helper tests passed afterward.
   Sources: [DeepSeek thinking](https://api-docs.deepseek.com/guides/thinking_mode/),
   [Z.AI preserved thinking](https://docs.z.ai/guides/capabilities/thinking-mode).

2. **Verified: truncation accepted as completion and destructive compaction.** Commit `89e7bd8`.
   Responses `status=incomplete`, Chat Completions `finish_reason=length`, Anthropic
   `max_tokens`/`model_context_window_exceeded`, and Gemini `MAX_TOKENS` become
   interrupted chat results. `interrupt_data` contains `type=provider_response_truncated`,
   provider, reason, and original Responses `incomplete_details` when present.
   Partial text and usage survive; tools from the truncated turn never execute.
   The signal survives follow-up rounds and usage accumulation. Summary parsers reject
   these signals before replacing any history, in sync and async compaction.
   Codex already rejected incomplete terminal SSE events; its explicit reason-bearing
   exception and lossless compaction behavior remain covered in both modes.
   Regression: `test_provider_review_truncation.py`, 35 cases failed before the fix;
   four Codex controls already passed. Added empty-envelope/filter/context-limit cases.
   Sources: [Responses incomplete events](https://developers.openai.com/api/reference/resources/responses/streaming-events),
   [Chat finish reasons](https://developers.openai.com/api/reference/resources/chat/subresources/completions/methods/create),
   [Claude stop reasons](https://platform.claude.com/docs/en/build-with-claude/handling-stop-reasons),
   [Gemini finish reasons](https://ai.google.dev/api/generate-content#FinishReason),
   [DeepSeek finish reasons](https://api-docs.deepseek.com/api/create-chat-completion/),
   [OpenRouter normalized responses](https://openrouter.ai/docs/api_reference/overview).

3. **Verified: Opus 4.7/4.8 incompatible sampling/manual thinking.** Commit `8dd3e51`.
   Metadata identifies adaptive-capable Claude models and adaptive-only Opus 4.7/4.8.
   Normal turns and summaries omit unsupported temperature. On adaptive-capable models,
   `BareBoneModel.deepthinking=True` enables `thinking={"type":"adaptive"}`; optional
   `reasoning_effort` becomes `output_config.effort`. Opus 4.7/4.8 reject manual
   `thinking_budget_tokens` locally with an actionable error. Older manual budgets keep
   their existing validation; adaptive thinking preserves required tool selection.
   Regression: `test_provider_review_thinking.py`, nine cases failed before the fix;
   dated model IDs, adaptive effort and provider-default effort are covered.
   Source: [Claude thinking and sampling compatibility](https://platform.claude.com/docs/en/build-with-claude/thinking).

4. **Verified: Gemini 2.5 Pro thinking cannot be disabled.** Commit `b7b2052`.
   Metadata permits `thinkingBudget=0` only for Gemini 2.5 Flash families. Pro omits
   the budget and retains dynamic thinking; unsupported generations also omit it.
   Regression: `test_provider_review_gemini_budget.py`, four cases failed before the fix;
   six model/alias controls pass, including Flash and Flash Lite.
   Source: [Gemini per-model thinking budgets](https://ai.google.dev/gemini-api/docs/generate-content/thinking).

5. **Verified: multiple required Anthropic tools used an invalid choice.** Commit `f2703c6`.
   Use `any`, retaining `disable_parallel_tool_use` when configured. Both legacy
   assertions expecting Anthropic `required` were corrected (one in the final test
   follow-up). Regression: `test_provider_review_tool_choice.py` failed before the fix;
   required/forced/automatic/manual-thinking tool choices remain covered.
   Source: [Claude tool choice](https://platform.claude.com/docs/en/agents-and-tools/tool-use/define-tools).

6. **Verified: borrowed AsyncClient request timeout ignored.** Commit `b1b786a`.
   The ordinary async branch passes the request timeout explicitly. Sync already did.
   Regression: `test_provider_review_timeout.py` failed for ordinary async requests
   before the fix. Sync/async, controlled/ordinary requests now match while borrowed
   clients remain open and their default timeout stays unchanged.
   Source: [HTTPX request and client timeouts](https://www.python-httpx.org/advanced/timeouts/).

7. **Verified: async checker-only cancellation waited for completion.** Commit `5ca75c6`.
   Poll controls while awaiting an owned operation task, then cancel and drain that
   task on cancellation/deadline. Registrar callbacks also target the operation.
   Internal controls do not cancel or uncancel the parent. External cancellation
   retains `CancelledError` and its cancellation count, including checker races.
   Regression: `test_provider_review_cancel.py`; real loopback baseline took 0.304 s
   against a 0.150 s bound; the fixed run closes/drains the request within that bound.
   The external/checker race also failed before the fix. All 42 targeted lifecycle/
   Codex tests passed. Caller client ownership is preserved.
   Sources: [Python task cancellation](https://docs.python.org/3/library/asyncio-task.html#task-cancellation),
   [HTTPX async resource ownership](https://www.python-httpx.org/async/).

8. **Verified: tool errors bypassed configured character caps.** Commit `ebb1972`.
   Direct execution, validation, missing-tool and timeout errors now obey the cap.
   The shared batch boundary caps every result, covering planning errors, duplicate/
   limit failures, parallel wrapper exceptions and provider-bound results.
   Regression: `test_provider_review_tool_caps.py`, all 22 cases failed before the fix;
   sync/async and all seven provider formats pass after it. Character limits are
   IkaCore policy; provider docs establish that error bodies enter tool-result history.
   Sources: [OpenAI function outputs](https://developers.openai.com/api/docs/guides/function-calling),
   [Claude tool results/errors](https://platform.claude.com/docs/en/agents-and-tools/tool-use/handle-tool-calls).
   The latter is the current fetched URL replacing the older `/tool-results` link
   included in the implementation commit message.

9. **Verified: Anthropic cache usage omitted.** Commit `9bfad4e`.
   Input/total tokens include uncached input plus cache reads and cache creation,
   including summary accounting. `input_cached_tokens` retains reads;
   `input_cache_creation_tokens` retains creation through `UsageInfo` aggregation.
   Existing four-key usage mappings remain unchanged when creation is zero.
   Regression: `test_provider_review_usage.py` verifies exact totals, both cache
   categories, summary tokens and accumulated usage. Existing cost consumers receive
   complete input counts and the read category; pricing policy in `src/IkaCore`
   is outside this worktree.
   Source: [Claude cache usage accounting](https://platform.claude.com/docs/en/build-with-claude/prompt-caching).

10. **Verified: Gemini thought tokens omitted from output.** Commit `9bfad4e`.
    Output counts candidate plus thought tokens; a total-only fallback subtracts input
    instead of charging it again as output. Provider totals remain authoritative.
    Regression: `test_provider_review_usage.py`, five total usage cases failed before
    the combined accounting fix; explicit totals, inferred totals, thought-only and
    ordinary output are covered.
    Source: [Gemini UsageMetadata](https://ai.google.dev/api/generate-content#UsageMetadata).

11. **Verified: multiline Codex SSE data rejected.** Commit `d9f25f4`.
    Shared sync/async decoding joins data fields with newlines and dispatches only on
    a blank event boundary, resetting event/data buffers. Comments and unknown fields
    are ignored; unfinished EOF frames are discarded. A malformed legacy fixture was
    corrected to use event separators. Regression: `test_provider_review_sse.py`,
    seven cases failed before the fix; LF, CRLF, CR and field/buffer cases pass.
    Follow-up `fcdc22b` corrects the legacy generated byte-boundary fixture to use
    blank event separators and adds a genuinely multiline completion event; its
    70 generated Unicode/chunk/newline cases retain their original assertions.
    All 53 targeted SSE/helper/async-transport tests passed.
    Source: [WHATWG event-stream interpretation](https://html.spec.whatwg.org/multipage/server-sent-events.html#event-stream-interpretation).

12. **Verified: permanent HTTPX failures retried.** Commit `b4f1e88`.
    Ordinary and Codex sync/async routes share explicit transient transport classes:
    timeout, network, remote-protocol and proxy errors. UnsupportedProtocol,
    LocalProtocolError, InvalidURL and TooManyRedirects propagate immediately.
    Regression: `test_provider_review_retries.py`, six permanent-failure cases failed
    before the fix; 36 route/mode/error cases verify exact attempts and exception types.
    All 110 targeted retry/request/Codex tests passed.
    Source: [HTTPX exception hierarchy](https://www.python-httpx.org/exceptions/).

## Payload checksums

All **21/21** SHA-256 checks match the existing fixtures. **No checksum or provider
fixture changed.** The benchmark uses optional tools, Sonnet 4.5, Gemini 2.5 Flash,
and no duplicated persisted/plain assistant pairs, so the intentional fixes leave
its valid payloads unchanged. Separate regression tests cover changed payloads.

Intentional test-fixture corrections: `test_provider_edge_contracts.py` and
`test_optimized_provider_payloads.py` now expect Anthropic `any`;
`test_codex_helpers_extra.py` separates independent SSE frames;
`test_hardening_properties.py` does the same and uses a multiline terminal frame.
No assertion or generator domain was removed.

## Final verification

Source/test revision: `fcdc22bb8063227df5be232210d76200587c5752`. Subsequent
report edits are documentation-only. All checks used the worktree virtual environment,
CPython 3.13.15, HTTPX 0.28.1, httpcore 1.0.9 and the required environment prefix.

- `python -m pytest -q`: **1,525 passed / 2 opt-in skips**, 31.82 seconds.
- `python scripts/quality.py`: **PASS**. Ruff clean; Pyright **0 errors / 0 warnings**;
  all source compilation checks pass. Its coverage suite reports **1,525 passed /
  2 skips**, 48.65 seconds.
- Lines: **9,851 / 9,851 = 100%**. Branches: **2,696 / 2,772 = 97.2583%**.
  Combined statement/branch coverage: **99.3979%**. All **129 modules / 16 areas**
  pass their existing coverage floors, with **zero gate failures**. No floor or
  source exclusion was changed.
- Wheel package contents and imports from the installed wheel outside the source
  tree pass. The built wheel is `ikacore-2.0.0-py3-none-any.whl`.
- All **21/21 provider payload checksums** pass without modifications.
- The first full-suite attempt exposed one invalid preexisting SSE property fixture:
  **1 failed / 1,524 passed / 2 skips**. Commit `fcdc22b` fixes frame construction
  and strengthens its multiline coverage. The full suite was then repeated because
  that fixture changed; the quality script ran once afterward. No runtime change was
  needed for that failure. Original failure output is retained separately.

The live matrix was run **once**, against the installed wheel and the exact requested
model, using explicit local-login access:

```sh
PATH="$PWD/.venv/bin:$PATH" VIRTUAL_ENV="$PWD/.venv" \
python scripts/live_matrix.py --provider codex --use-local-login --model gpt-5.6-sol --case-timeout 45
```

The invocation had a 900-second outer bound; each case had a 45-second limit and
one live request attempt. Result: **14 passed / 0 failed / 1 skipped**, exit **0**.
The skip is `reasoning_tools: reasoning_not_exposed`: this successful tool round
returned no inspectable reasoning, so it does not establish live reasoning replay.
The synthetic `transient_retry` case is identified as such in the matrix artifact.
Text/tools (sync and async), simple/staged agents, fresh-process resume, cancellation,
SSE, usage, deadlines, compaction and workflow HITL passed. The full matrix took about
40 seconds of reported case time.

Retained artifacts (not committed):

- `artifacts/quality/coverage.json`, `coverage-matrix.json`, `coverage-matrix.md`, `tests.xml`.
- `artifacts/live/codex.json`, `artifacts/live/summary.md`.
- `/private/tmp/ikacore-providers-final-pytest.log`, `ikacore-providers-final-quality.log`,
  `ikacore-providers-final-live.log`, `ikacore-providers-before-sse-fixture-pytest.log`.

The matrix runtime fingerprint is
`99718f3efa37e2cf8deb3b1d8639242822a133090ca50d0d7fd0a79fe7aa8b50`.
Nothing was pushed or posted. Core-runtime changes from the parallel worktree are
not part of these measurements and must be validated after integration.
