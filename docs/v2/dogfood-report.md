# New-user application dogfood report

Current status: all findings are addressed in the [Resolution section](#resolution--ergonomics-implementation).
The original observations and validation numbers below are retained as historical evidence.

Date: 2026-10-18. Runtime: unreleased IkaCore 2.0.0, wheel built from
`aca8c53a57332c0e5ab225628ac7b29bae99d23b`, Python 3.13.15, HTTPX 0.28.1.
Worktree: `/private/tmp/ikacore-w-dogfood`, branch `v2-dogfood`.
No runtime files were changed. No push or external comment was made.

The six applications live under [examples/apps](../../examples/apps/README.md).
They run actual agents, tools, workflows, memory and persistence against offline
provider envelopes, then opt into Codex using `--live --use-local-login --model gpt-5.6-sol`.
They use an installed wheel, not editable imports. The wheel fingerprint, request counts,
per-app live outcomes and limitations are in [LIVE_RESULTS.md](../../examples/apps/LIVE_RESULTS.md).

## Bugs for runtime owners

All three reproductions are strict expected failures under
`src/IkaTest/dogfood_findings/`. They need no credentials. The first two are in
`test_validation_contract.py`; the third is in `test_chat_prompt.py`.

1. **DOGFOOD-1 / high** — `test_final_check_can_read_final_message_without_constructor_keyerror`.
   A documented final-answer validator that reads `result["final_message"]` fails during
   agent construction with `KeyError: 'final_message'`. The constructor executes it with
   an empty history dictionary, whereas the actual simple-run validator gets final output.
   Suggested fix: validate callability/signature without executing application code, and
   specify a single typed result contract. App workaround: `.get("final_message", "")`.
2. **DOGFOOD-2 / high** — `test_staged_execution_invokes_final_answer_check`.
   A valid, configured final-answer check is called at construction but never called on
   a staged agent's final result. This permits an invalid research answer to escape the
   advertised validation. Suggested fix: apply the same final validation policy to staged
   completion, including a bounded correction path. App workaround: a separate simple
   writer and explicit application validation before accepting the result.

3. **DOGFOOD-3 / high** — `test_updated_prompt_reaches_second_execution_request`.
   After a successful execution, assigning a new `agent.prompt` silently sends the old
   first_input to the provider again. This made the first live chat replay its original
   save request rather than advance through new turns. Suggested fix: provide an explicit
   public append/execute-turn API and define prompt assignment behavior. App workaround:
   refresh first_input and retain prior user text as a stage_input history entry. No runtime
   change was made.

These tests deliberately fail until their runtime contracts are fixed. A strict XPASS
requires removing the marker as part of the owning runtime fix; do not weaken the assertion.

## Friction log

Severity: high blocks correct use; medium needs a workaround or source lookup; low is
avoidable confusion or noise. Findings describe observed v2 behavior, not provider claims.

| Severity | Observation | Suggested change / app handling |
| --- | --- | --- |
| High | Final-check constructor receives history while execution receives output. | DOGFOOD-1 above; added a temporary working pattern to USAGE. |
| High | Staged completion skips final checks. | DOGFOOD-2 above; research uses a separate validated writer. |
| High | Exhausting simple validation retries returns the invalid last output without a failure status. | Offer fail-closed behavior or an explicit validation status; document it. Apps revalidate before accepting. This is recorded as policy friction, not an assertion about a promised exception type. |
| Medium | README/USAGE initially taught editable installation only. This conceals package-data and isolated-import failures for an evaluator. | Added fresh-wheel installation, an offline application command and opt-in live command. |
| Medium | Final-check callback shape was not demonstrated; only docstring/bool requirements appeared. | Document exact input and provide a typed example after fixing the constructor contract. |
| High | Changing prompt after execution silently replays the original first_input (DOGFOOD-3); no documented chat turn API exists. | Added a finite multi-turn example and brief USAGE guidance; expose a first-class documented chat-turn API if intended. |
| Medium | Compaction requires assigning context_budget outside the grouped configuration API; its unit and observability were not shown. | Add a documented config field and event/counter; example verifies post_compaction snapshots. |
| Medium | Snapshotting a completed answer returns saved output before restoring conversational state. | Document completed-result retrieval versus pending-turn resume; chat pauses at post_model and restores that turn in a new process. |
| Medium | Memory persistence wording says short-term data disappears when the agent completes, although actual lifetime follows the caller's backend. Snapshots do not persist arbitrary memory backends. | Explain backend lifetime and separate persistence; JsonStorage makes ownership explicit. |
| Medium | Agent fork convenience API has no resume_input argument; workflow fork API is not demonstrated. | Document SnapshotStore.fork_run_from_snapshot followed by workflow.resume_from_snapshot; add a convenience method accepting a human answer. |
| Medium | Workflow parallel execution is named use_async but uses worker threads and remains blocking. AsyncClient instances cannot be shared between those independent loops. | Document scheduling/concurrency precisely; app uses a thread-safe synchronous HTTPX client. |
| Medium | Workflow clones reset `.client`; agent.shutdown closes an attached caller client. No public borrowed-client constructor is documented. | Add explicit client injection/ownership configuration preserved by clones. Example subclass injects on clone and detaches on shutdown. |
| Medium | WorkflowResult.history is execution output, not the transcript suggested by its name. Accessing history['first_input'] raises KeyError. | Rename/document the field and expose actual history separately. App checks fan-in at the public compress_hook. |
| Medium | Worker execution errors can become missing workflow results; the consumer sees KeyError on a downstream node. | Expose per-node structured errors or raise a workflow error with node identity. This was encountered while debugging an offline fixture, not classified as a new correctness bug. |
| Medium | No documented mock provider seam or response fixture format. | Provide one supported testing adapter; common.py currently patches the documented request-function owners and uses provider-native envelopes. |
| Medium | Defaults allow 100 steps, 50,000 output tokens, 900-second steps, two extensions and validation extra steps. Small demos need several explicit bounds. | Show bounded application defaults in quick start; document that requests used for extension/summarization add calls. |
| Low | cli=False suppresses runtime boxes but expected tool timeouts still log warnings; tool callback errors can print full tracebacks. | Document/offer scoped logger behavior; reviewer intentionally demonstrates the timeout. |
| Low | Stage/agent control tools have required argument names that were missing from prose. | Show agent_end(input=...), stage_end(input=...), ask_user(question=...) examples. |
| Low | The prompt automatically encourages very detailed answers, at odds with small application results. | Make default prompt concise or document the PromptConfig override. Apps require short tagged/structured outputs. |
| Low | Stages uses a capitalized constructor argument; names of source owners were hard to guess when docs ran out. | Prefer stages in a future API cleanup and publish signatures with import locations. |

An app-author mistake also surfaced: macOS `/var` versus `/private/var` symlinks require
resolving the temporary repository root before containment checks. The reviewer does so;
this is not an IkaCore finding.

## Source lookups forced by missing documentation

Initial reading was restricted to README.md, docs/USAGE.md, CHANGELOG.md and docs/v2/README.md.
The following is the complete source-lookup inventory, grouped by the question that forced
it. These are documentation gaps, not evidence that new users should read runtime internals.
Some guessed filenames did not exist; searches then located the actual owners.

| Lookup | Why the four documents were insufficient |
| --- | --- |
| `IkaCore/config.py`, `runtime_control.py`, `agents.py` signatures | Needed all grouped fields/defaults, exact keyword shapes and public pause/fork/chat methods. |
| `agent_policy.py`, `agent_runtime_dispatch.py`, `agent_helper_support.py`, `agent_state.py` | Needed validator argument contract; diagnosed the construction KeyError and staged bypass; checked extension/validation behavior and whether extension counters are observable. |
| `agent_execution_helpers.py`, `agent_execution_loop.py`, `prompts.py` | Needed stage control arguments, how a one-step stage exercises an extension, and chat turn/budget behavior. |
| `agent_chat_support.py`, `agent_runtime_foundation.py`, `IkaModel/base.py` | Needed get_barebone/chat_wrapper usage, client handling, default prompt and callback/tool conversion contracts. |
| `tools.py`, `IkaMem/agent_tools.py` | Needed timeout/replay metadata and exact memory tool names/arguments; memory lifetime wording is misleading. |
| `agent_snapshots.py`, `snapshot_store.py`, `snapshot_replay.py`, `snapshot_runs.py` search results | Needed pending versus completed restore, workflow fork with an alternative resume_input, and immutable fork lineage. |
| `workflow.py`, `workflow_types.py`, `workflow_async_executor.py`, `workflow_snapshots.py` signatures, `workflow_async.py` and `workflow_prompt_context.py` search results | Needed parallel scheduler/client ownership, WorkflowResult.history contract, returned snapshot store and workflow context observation. |
| `agent_lifecycle.py`, `agent_initialization.py` search results | Needed clone client reset, shutdown ownership and whether a small public-method subclass could safely lend a client. |
| `IkaModel/request_control.py`, `chat_interface/chat_runtime.py`, `chat_interface/chat_request.py`, `history_replay.py`, `summarization.py` search results | Needed cancellation-checker signature, mock seam, summary HTTP path, compaction markers and chat history semantics. |

As explicitly requested, scripts/live_provider_check.py was also consulted for credential
opt-in, fixture/response shapes and bounded live conventions. pyproject.toml was read for
build/test requirements; scripts/quality.py was read to run the requested quality command.
No provider implementation or existing test suite was read to design the applications.
After the first full pytest run, `src/IkaTest/test_documented_examples.py` was read to
understand a failing newly added USAGE code block: the harness executes Python blocks
sequentially, and the illustrative workflow fork appeared before workflow construction.
The documentation was corrected; the existing harness was not changed.

## Getting started change applied

README.md and docs/USAGE.md now begin application evaluation with a clean venv and built
wheel, a deterministic offline app, then an explicit bounded Codex live command. USAGE
also explains validation limitations, workflow fork/HITL, blocking parallel scheduling,
borrowed-client lifetime, chat turns and separate memory persistence. The runnable
applications are the smallest useful entry point before the full migration guide.

## Validation scope

Offline app tests block accidental HTTP requests, run the real runtime and validate tools,
structured submission, dependency fan-in, HITL continuation/fork, memory save/search,
compaction and provider response parsing. Three strict xfails preserve unfixed runtime bugs.
The parallel cancellation demo exercises SIGINT before dispatch and cancellation checking;
it does not claim to validate in-flight live cancellation or graceful cleanup of arbitrary
uncooperative tools. Live provider compatibility is established only for the listed Codex
model/account; mocked providers are not claimed as live verified.

All six latest Codex application runs passed. The twelve-turn chat rerun observed eleven
compactions and resumed its final turn in a new process before recalling memory. The
initial chat failed at its 90-second session bound; after correcting turn replay at the
application layer, the rerun passed with 30-second turn scopes inside a 180-second session.
All attempts together used 74 request-boundary invocations, including an expired probe.
The full per-app evidence and its limits are in LIVE_RESULTS.md.

Final validation passed:

- `pytest -q`: **980 passed, 1 skipped, 3 xfailed** (18.35 seconds).
- `python scripts/quality.py`: **passed**, including Ruff, Pyright (zero errors), source
  compilation, **980 passed / 1 skipped / 3 xfailed** with coverage, all coverage matrix
  gates (122 modules, 16 areas, zero failures), fresh-wheel data inventory and isolated imports.
- Combined coverage: **94.62%**, above the unchanged 91% gate.
- Fast offline application subset: **7 passed, 3 xfailed** in 1.62 seconds.

The first sandboxed pytest attempt could not bind loopback sockets; it was rerun with
transport-test access. The first full accessible run caught a new illustrative USAGE block
before its workflow definition; that documentation error was corrected and all final
checks passed. No existing tests or gates were weakened.

Implementation commit: `1f667520889de190ef7848769935bea332b102b7`.
The documentation/live-evidence follow-up commit is reported in the handoff; its own SHA
cannot be embedded in the file it commits.

## Resolution — ergonomics implementation

The observations and 980-test/74-call evidence above describe the original dogfood
revision. The following resolutions supersede its API limitations. Implementation starts
from `49dbeb3` on `v2-ergonomics`; current verification and rerun evidence are recorded in
[the application results](../../examples/apps/LIVE_RESULTS.md). No original regression
assertion was weakened. All three strict xfail markers were removed; their HTTPX fixtures
now carry request metadata so `raise_for_status()` reaches the runtime assertions.

| Item | Resolution |
| --- | --- |
| DOGFOOD-1: construction invokes final check | Fixed. Construction checks synchronous callability/signature only. `AgentResult` is the single typed final mapping; callbacks return an actual bool at execution. No docstring or annotation is required. |
| DOGFOOD-2: staged final checks bypassed | Fixed. Simple/staged completion share validation and bounded correction, including snapshot-resumed corrections. Corrections use existing evidence and expose only `agent_end`, avoiding repeated application side effects. The research app validates its staged answer directly. |
| DOGFOOD-3: stale prompt replay | Fixed. Prompt assignment replaces the task and clears transcript/pending continuation. `send(message)` appends an ordered user turn with fresh configured steps; final answers are retained. Seven mocked routes verify turn ordering and no duplication. |
| Exhausted validation returns invalid output silently | Every result reports `validation.status`, `attempts`, `failed_checks`. `fail_closed=True` raises `FinalAnswerValidationError` retaining `.result`. Tests cover both modes, exhaustion, successful correction and correction pause/resume. |
| Fresh-wheel installation hidden by editable setup | README/USAGE/apps retain clean-wheel setup. Offline and live reruns use a newly built non-editable wheel. Quality verifies its complete module/data inventory and isolated imports. |
| Callback shape undocumented | USAGE demonstrates `def has_answer(result: AgentResult) -> bool`; generated API reference lists the result fields and policy signatures. |
| No documented chat-turn API | `agent.send(message)` and `.transcript` replace the app history-editing workaround. `execution()` runs the current task. The long-chat app uses public APIs for every turn and resumed process. |
| Compaction budget outside grouped config; no observability | `ExecutionLimits(context_budget=N)` uses tokens; `compaction_count` and `compaction` RuntimeEvents report successful summaries. Counters persist in snapshots. `post_compaction` snapshots remain available. |
| Completed vs pending snapshot semantics unclear | Completed restore loads transcript, usage/cost and counters, returns saved output without a call, and permits a subsequent `send()`. `resume_input` on a completed agent snapshot is rejected. Pending restore consumes the saved turn/answer first. USAGE and API explain both. |
| Memory lifetime wording misleading | USAGE explicitly ties lifetime to the caller-owned backend. Snapshots do not serialize arbitrary memory stores. The chat app persists JsonStorage separately and recalls it in a fresh process. |
| Agent fork lacks answer; workflow fork undocumented | Agent and workflow `fork_from_snapshot(..., resume_input=...)` fork and answer in one call. Workflow mappings handle multiple interrupted instances. Approval app forks its rejection through this convenience. |
| Workflow use_async means blocking threads | Canonical `run(parallel=True, concurrency=N)` is documented as blocking. Dependencies, fan-in, completion ordering and separate context-preparation workers are explained. `use_async` remains a deprecated alias. Default instance workers decrease to four. |
| Clones discard clients; shutdown closes borrowed client | `ProviderConfig(client=...)` and direct client assignment borrow. Clones retain borrowed references; shutdown leaves them open. `client_factory` creates a distinct owned client per instance. Workers shut down clones. The pipeline subclass/detach workaround is removed. |
| WorkflowResult.history is output, not transcript | Canonical `.output`, `.transcript`, `.instance_outputs` and `.instance_transcripts` separate the contracts and preserve every fan-out instance. `.history` is a deprecated output alias. Transcript survives cached workflow replay. |
| Worker errors become missing results/KeyError | `WorkflowError.errors` has node/instance/type/message identity; `.partial_results` retains completed nodes. Dependents stay blocked. Cancellation/deadline exceptions keep their types. Tests cover both scheduling modes and snapshot failures. |
| No supported mock seam | Public `IkaCore.testing.MockProvider` supports native envelopes/dicts, a responder, scoped/nested sync/async use and workflow workers. `provider_response` builds provider-native envelopes. Apps and executable docs tests use this seam. |
| Large defaults require many bounds | Three `.bounded()` config presets supply 1,024 output tokens, six steps, 30-second steps, three tool rounds, twelve tool calls, no extensions and one correction with two extra steps. Existing unrestricted defaults are retained by design. Whole-operation deadlines and extra extension/forced-answer/summary/compaction calls are explained. |
| cli=False still prints warnings/tracebacks | Library loggers use NullHandler and do not configure root logging; presentation is scoped. Python logging explicitly controls diagnostics. The timeout app stays quiet by default. |
| Control-tool argument names missing | USAGE documents `agent_end(input=...)`, `stage_end(input=...)`, `ask_user(question=...)`, `change_stage(stage_index=..., reason=...)`, including availability. |
| Default prompt demands verbose answers | Default model prompt now asks for clear, concise answers. `PromptConfig(system_prompt=...)` is documented for style overrides. |
| Capitalized Stages and missing API reference | Canonical `stages=`/`.stages`; capitalized compatibility aliases remain so existing v2 callers can migrate incrementally. `docs/API.md` is generated from all package exports, testing adapters and maintained extension entry points; pytest asserts exact export/signature agreement. |
| Source lookups forced by missing docs | The generated API reference supplies signatures/imports/methods and USAGE covers grouped defaults, validation, tools, lifecycle, chat, snapshots, ownership and scheduling. Runtime implementation lookup is no longer required for these documented operations. |

Intentional behavior changes are listed individually in `CHANGELOG.md` 2.0.0 and the
v2 migration table. Two deliberate compatibility choices are retained: unrestricted
configuration defaults remain available alongside the bounded presets, and the old
`Stages`, workflow `use_async`, and result `history` spellings remain documented aliases.
Text-only corrections replace the suggested repeated staged execution to avoid replaying
application tools with side effects.


### Resolution verification

Implementation commit: `ed55da5928451a26e600d59b3266fef33bc92d17`.
The [handoff report](ergonomics-handoff.md) includes API before/after examples and every
intentional behavior change. Fresh-wheel offline apps: **7 passed** (six apps plus SIGINT);
Codex live apps: **6/6 passed**, **44 request boundaries**, four live chat turns with three
compactions (twelve turns remain covered offline). Historical evidence is retained separately.

Final `pytest -q`: **1,075 passed / 1 skipped**, **19.37 s**, no xfails.
`python scripts/quality.py`: **passed**, including the same **1,075 passed / 1 skipped**
under coverage, Ruff, Pyright (zero errors), compilation, wheel data/module inventory and
isolated imports. Combined coverage **94.80%**, line **96.46%**, branch **88.82%**;
**127 modules / 16 areas**, zero coverage gate failures, all floors intact.
