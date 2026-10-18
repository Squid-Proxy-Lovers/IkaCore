# V2 application ergonomics handoff

Worktree: `/private/tmp/ikacore-w-ergonomics`. Branch: `v2-ergonomics`, based on
`49dbeb3`. Implementation: `ed55da5928451a26e600d59b3266fef33bc92d17`.
All changes are local; no push or external post was made. Every commit carries the
requested Claude Opus 5.5 coauthor trailer.

## Per-item resolution

The [dogfood resolution table](dogfood-report.md#resolution--ergonomics-implementation)
covers all three correctness bugs, every high/medium/low friction item, and the source
lookup documentation gaps individually. None remains deferred. In particular:

- DOGFOOD-1: construction validates callable/signature without running application code.
  All checks receive one `AgentResult` mapping on completion and must return bool.
- DOGFOOD-2: staged/simple answers share validation, bounded correction, explicit status,
  fail-closed mode and snapshot-resumed retry budgets.
- DOGFOOD-3: prompt assignment starts a fresh task; `send()` appends a conversation turn.
  Ordered user turns and final answers survive snapshots and seven provider serializers.
- Grouped token budgets expose compaction events/counters. Completed restore loads chat
  state before returning cached output; pending restore continues its saved turn.
- Agent/workflow fork convenience accepts an answer. Workflow scheduling explicitly names
  blocking threads and concurrency. Results separate output/transcript and retain instances.
- Borrowed client injection survives clones and is never closed. Factories create separately
  owned clients. Worker failures identify nodes/instances and block dependent execution.
- The supported mock adapter replaces app patches. Bounded presets, quiet named logging,
  exact control-tool arguments, concise prompts, lowercase stages and a generated/tested
  API reference resolve the remaining ergonomics issues.

The three original strict repros now pass without xfail; their assertions are unchanged.
Two HTTPX fixtures needed request metadata so `raise_for_status()` could reach the assertions.

## API before / after

Final checks previously required `.get()` to survive a constructor probe and a separate
simple writer for staged output. They now validate the staged result directly:

```python
# Before: callback was also called on an empty history at construction.
def cited(result):
    return "[A]" in result.get("final_message", "")
# A separate writer/application check was needed after staged execution.

# After:
from IkaCore import AgentResult, ExecutionPolicy

def cited(result: AgentResult) -> bool:
    """Include source [A]."""
    return "[A]" in result["final_message"]

policy = ExecutionPolicy.bounded(final_answer_checks=(cited,), fail_closed=True)
# Pass policy=policy to either simple or staged agents.
# result["validation"] == {"status": "passed", "attempts": 1, "failed_checks": []}
# On exhaustion: catch FinalAnswerValidationError; its .result retains the failure.
```

Chat previously rewrote internal history for every turn. The public continuation is now:

```python
# Before:
agent.prompt = message
agent.message_history["first_input"] = {"message": message, "tokens": 0}
# Previous user turns also needed manual stage_input history entries.
result = agent.execution()

# After:
result = agent.send(message)
conversation = agent.transcript  # defensive copy, possibly compacted
agent.prompt = "Independent task"  # explicitly clears transcript/pending turn
```

Compaction and client injection move into grouped configuration:

```python
# Before:
agent.context_budget = 2000
agent.client = caller_client
# Workflow clones needed a custom injection subclass and shutdown detachment.

# After:
from IkaCore import ExecutionLimits, ProviderConfig
limits = ExecutionLimits.bounded(context_budget=2000)  # tokens
provider = ProviderConfig.bounded(client=caller_client)  # borrowed, clone-preserved
owned_provider = ProviderConfig(client_factory=lambda: httpx.Client())
# agent.compaction_count and RuntimeEvent(kind="compaction") expose compactions.
```

Workflow names now describe the scheduling and returned data:

```python
# Before:
results = workflow.run(use_async=True)
output = results["writer"].history  # actually output, not conversation
fork = store.fork_run_from_snapshot(snapshot_id)
results = workflow.resume_from_snapshot(fork["snapshot_id"], resume_input="NO")

# After:
results = workflow.run(parallel=True, concurrency=2)  # blocking worker threads
output = results["writer"].output
transcript = results["writer"].transcript
all_instances = results["writer"].instance_outputs
results = workflow.fork_from_snapshot(snapshot_id, resume_input="NO")
# WorkflowError.errors provides node_name / instance_id / error_type / message.
# WorkflowError.partial_results retains already completed nodes.
```

Offline fixtures no longer patch implementation modules:

```python
# Before: patch request_interface and summarization sync/async functions separately.

# After:
from IkaCore.testing import MockProvider, provider_response
fixture = provider_response("https://api.openai.com/v1/responses", {}, ("agent_end", {"input": "OK"}))
with MockProvider([fixture]) as mock:
    result = agent.execution()
# Native dictionaries or responder(url, payload) support routing and summaries too.
```

Use `stages=[...]` in the agent constructor. Prefer all three `.bounded()` presets:
provider output 1,024 tokens; six steps; 30 seconds/step; three tool rounds; twelve tool
calls; no extensions; one validation correction with two extra steps. A whole-operation
`request_controls(timeout=..., max_retries=1)` remains necessary to bound multiple calls.

## Intentional changes and compatibility decisions

Every item is documented in [CHANGELOG.md 2.0.0](../../CHANGELOG.md) and the
[v2 migration table](README.md#breaking-changes-and-migration):

1. No constructor validator execution or docstring/annotation requirement; non-bool returns
   fail at execution. Staged results validate. Correction uses only `agent_end` and existing
   evidence, avoiding repeat application effects.
2. Prompt assignment clears task history/pending state. `send()` is the conversation API.
   Final answers are now retained in the replay transcript.
3. Completed agent snapshots restore conversation; `resume_input` on a completed snapshot
   raises. Pending snapshots retain their continuation semantics.
4. Direct/client-config assignment borrows; ownership requires a factory or an internal
   client. Clones preserve borrowed references and worker shutdown closes owned resources.
5. Workflow worker failures raise structured `WorkflowError` instead of silently omitting
   results. Request cancellation/deadline exceptions retain their types.
6. `WorkflowResult` uses `output=` plus transcript/instance maps. Use keyword construction
   for extended grouped config/result dataclasses; added fields can shift positional bindings.
7. Default workflow instance workers decrease from 30 to four. Context preparation has a
   separate bounded pool; caller-supplied executors are never replaced by `concurrency=`.
8. The default model prompt is concise. Named diagnostic loggers use NullHandler and the
   library no longer configures global/root logging.

Unrestricted config defaults remain available alongside the bounded presets, as explicitly
allowed by the brief. Capitalized `Stages`, workflow `use_async`, and result `history`
remain deprecated compatibility aliases. Text-only correction deliberately replaces a
full staged rerun to avoid replaying side-effectful tools.

## Historical ergonomics-only installed-wheel results (`ed55da5`)

Fresh wheel SHA-256:
`7d946996cbe3fbfebef67609860c5ca5713e644f37b3d39dc3330aef79ee451e`.
All six offline apps plus cancellation passed (**7 tests**, 1.64 s); offline chat covers
twelve turns. All six live apps used local-login opt-in and `gpt-5.6-sol`, sequentially:

| App | Live outcome | Request boundaries | Seconds |
| --- | --- | ---: | ---: |
| research_assistant | passed; staged citations checked fail-closed | 5 | 17.14 |
| code_reviewer | passed; structured review and timeout | 6 | 13.85 |
| approval_workflow | passed; fresh process and rejection fork | 11 | 30.65 |
| long_chat | passed; four turns, three compactions, resume and recall | 16 | 55.76 |
| parallel_pipeline | passed; fan-in and borrowed client | 5 | 6.87 |
| provider_switch | passed; Codex live, six other routes mocked | 1 | 1.73 |

Total: **44 request boundaries / 126.00 seconds**. These are observed boundaries, not
billing records; one pipeline probe expired before dispatch, so live model HTTP attempts
are at most 43. No failed attempts were discarded. Historical 74-boundary dogfood evidence
is preserved separately; differing live turn counts prevent a performance comparison.
Full reproducible commands and sanitized JSON are in
[LIVE_RESULTS.md](../../examples/apps/LIVE_RESULTS.md).

## Historical ergonomics-only verification (`ed55da5`)

- `python scripts/quality.py`: passed, **1,075 passed / 1 skipped**, no xfails,
  Ruff passed, Pyright zero errors, all source compilation passed.
- Coverage: **94.80% combined**, **96.46% line**, **88.82% branch**. All **127 modules /
  16 areas** passed with unchanged floors. Workflow branch coverage: **93.75%** against 90%.
- Fresh-wheel package data, full module inventory and isolated imports passed.
- Forty-two historical execution scenarios preserve model-call/budget/boundary parity;
  tests explicitly assert the new validation status before comparing the original trace.
- The API reference is generated from public exports/signatures and checked by pytest.
- Final plain `pytest -q`: **1,075 passed / 1 skipped**, no xfails, **19.37 seconds**.

## Integration status

This handoff describes the source branch before integration. Current merged API behavior,
raised coverage floors and sequential validation are recorded in
[merge-validation.md](merge-validation.md). Historical numbers above are not merged results.

## Integrated measurements

The integrated runtime (parents `7c410f3` / `a2b7984`, runtime fingerprint
`72a19681dbe20599ee4d96d945fc697cfa36c6215196a231acc687d975256759`) passes **1,386 tests / 2 opt-in skips**,
**100% lines / 97.1816% branches / 99.3842% combined** with all raised floors retained.
All six live Codex apps pass; the matrix reports **14 passed / 0 failed / 1
reasoning_not_exposed skip**. See [merge-validation.md](merge-validation.md) for current
revision labels, soak metrics, semantic integration checks and live limits.
