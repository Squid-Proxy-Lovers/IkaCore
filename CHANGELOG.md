# Changelog

## 2.0.0

V2 intentionally breaks compatibility with v1. See the
[migration guide](docs/v2/README.md#breaking-changes-and-migration).

- Removed `CheckpointStore`, `load_checkpoint()`, constructor `checkpoint` and
  `checkpoint_db_path`, stage `checkpoint`, and `execution(checkpoint_uid=...)`.
  Enable the SQLite `SnapshotStore` with `enable_snapshots()` and resume using
  `resume_from_snapshot(snapshot_id, resume_input=...)`. V1 checkpoint databases
  are left intact and are not converted. Finish old runs with v1 before migrating.
- Snapshots record runs, nested frames, immutable state and events; boundaries commit
  atomically. Agent resume, fork/restart, HITL interrupts and workflow selective replay
  use this one backend. Replay cannot restore Python stacks or undo tool side effects;
  unsafe replay requires an explicit opt-in.
- Removed runtime settings `optimize_provider_payloads`, `modern_retries`,
  `dataflow_workflows`, `preserve_workflow_prompts`, `persistent_async_runner` and
  `retry_initial_seconds`, plus `close_thread_runner()`. Use request `wait_seconds`
  for the initial retry delay (default one second).
- Provider history replay and transient-error retries are now standard. Permanent
  HTTP/programming failures propagate. OpenAI uses Responses by default; completion
  controls no longer force tool selection. Sync workflows clone template agents and
  both workflow modes wait for dependencies while preserving each agent's task.
- Removed v1 import/private forwarding facades, the `IkaModel.base.TOKENMAX_MAPPING`
  alias (use `IkaModel.model_metadata`), unused `IkaCheckpointError`, and the unused
  Gemini payload loader (import `gemini_fill_payload` from `IkaModel.gemini.google`). Public packages report the
  installed `ikacore` version and ship `py.typed` markers.
- Examples use environment credentials and avoid printing bearer fragments. Documentation
  now covers functional memory setup, current snapshot APIs and offline validation.
- Validation includes deterministic provider payload, retry, execution, workflow,
  snapshot, memory and cancellation tests; type/lint/coverage gates; and wheel checks.
  Only Codex `gpt-5.6-sol` was verified live: 8/8 checks recorded at `183ce24` on
  2026-10-13 using the installed wheel at runtime revision `344348c`. That result
  predates the fixes below; live checks have not been rerun for the merged tree.
  Other providers/models and long-running application behavior require
  separate live validation. Historical quality/benchmark figures identify their revision.

### Fixed

- Invalidate interrupted downstream transcripts and their old answers when upstream
  workflow instances rerun. Restart with fresh inputs and enforce replay-safety checks.

- Clone interrupted workflow child continuations into forked runs, remapping their frames
  and snapshots, and reject cross-run child snapshots before routing human answers.

- Run exhausted HITL resumes through stage extension or forced finalization before
  completing the stage, so a last-step question cannot bypass the human answer.
- Persist interrupted workflow instances' snapshot IDs and route workflow `resume_input`
  answers to those children. Reuse separately completed child outputs, retain completed
  siblings, and continue dependents in sync and parallel workflows.
- Keep remaining step budgets for older v2 snapshots without `loop_state`, including
  the consumed allowance recorded by the current stage's entry boundary.
- Register synchronous HTTP abort before headers, clamp every per-request I/O timeout
  to the remaining deadline, and close only the owned response/stream. Remove raw
  socket shutdown so cancelling a preloaded response cannot abort another pooled request.
  A blocked synchronous read may observe explicit cancellation only at the next clamped
  I/O timeout; deadlines are per-I/O bounds, not hard wall-clock guarantees. Verify real
  loopback HTTP/HTTPS ownership and header/body timeout behavior.
- Require explicit `allow_unsafe_replay=True` approval before historical snapshot
  resume repeats unsafe tool effects, including nested calls.
- Pause snapshot-enabled workflows when an agent waits for human input, blocking its
  dependents in sync and async runs. Do not reuse interrupted outputs as completed
  results; invalidate their downstream cached results during replay.
- Close only the active HTTP response on synchronous deadline or cancellation, keeping
  caller-owned HTTPX clients usable.
- Keep nested-agent snapshots in the child's frame and restore the parent's execution
  context when the child returns.
- Restore the saved HITL transcript once and append the human answer before continuing.
- Clear stale replay state before fresh executions, including after failed resumes.
- Preserve consumed steps, step limits, extension counts and remaining overall stage
  budgets across resume, including in a new agent. Avoid charging prior steps twice to
  later stages.
- Close SQLite connections when snapshot setup fails.
