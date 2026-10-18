# Final v2 core review fixes

Implemented on `v2-fix-core` from `1ad479b`, exclusively in the assigned worktree.
No provider/transport sources were changed. Every reported finding was reproduced
with a safety assertion that failed before its fix. All commits carry the requested
Claude Opus 5.5 co-author trailer.

1. **Parallel chained agents — `c5e1616`.** The dispatcher reset and executed the
   shared downstream template, so concurrent clones combined their histories.
   Each invocation now executes a fresh downstream clone and shuts it down in
   `finally`. Regression: `test_parallel_chains_isolate_next_agent_and_close_owned_clients`;
   error cleanup: `test_chain_closes_owned_downstream_on_error`.
   Before: both outputs were ALPHABETA. After: ALPHA and BETA, an untouched template,
   and closed owned downstream clients.

2. **Overall budgets — `c0621c8`.** The stage limit forced at least one turn after
   exhaustion, and correction passes replenished consumed task steps. Zero
   allowances now stay zero; stage results carry the remaining allowance;
   explicit extension grants are credited and persisted; corrections retain only
   unspent steps plus their configured extra grant.
   Regressions: `test_backward_stage_transitions_exhaust_overall_budget`,
   `test_validation_correction_does_not_reset_consumed_budget`, and
   `test_restored_validation_correction_keeps_remaining_allowance`.
   Follow-ups: `0d68aba` (fresh correction replay) and `d8211be` (reviewed trace
   expectations). All 42 historical cases were compared against the immutable
   base: 37 preserve calls, results, usage, and summaries byte-for-byte.
   Five staged budget cases intentionally change calls/accounting. The original
   historical fixture remains intact; the corrected fixture pins the new traces.

3. **Delegated interruptions — `00c46eb`.** A paused child result was reduced to
   successful text output. Delegation now raises a structured interrupt containing
   the child's snapshot identity. Parent snapshots preserve the pending tool
   continuation; replay restores and answers the child before continuing the
   parent. Repeated questions update the identity, nested forks remap it, and
   missing/foreign/mismatched identities are rejected. Without snapshots the parent
   pauses with `interrupt_data.resumable=False` and an explanatory error.
   Regressions: `test_delegated_hitl_resumes_child_before_parent`,
   `test_delegated_hitl_without_snapshots_reports_nonresumable_pause`,
   `test_delegated_repeated_questions_update_pending_child_identity`,
   `test_workflow_resumes_delegated_child_in_parent_instance`, and
   `test_fork_preserves_nested_delegated_continuations`.
   Follow-up `22183c2` validates identity, exercises provider result shapes, and
   keeps helpers in the existing snapshot module so the coverage inventory is
   unchanged. Sync/async and staged/simple parent variants pass.

4. **Delegated cancellation — `638c0e3`.** The delegation error adapter caught
   `IkaRequestControlError` and returned it as tool output. It now re-raises before
   generic handling. Regression: `test_delegated_request_controls_propagate`
   covers cancellation and deadlines in sync/async execution, including cancelled
   snapshot status and no subsequent parent request.

   The broader-handler search found five more swallow points in optional memory
   operations/filtering and one in the output sink. All now preserve request
   controls, covered by `test_memory_operations_propagate_request_controls` and
   `test_output_sink_propagates_request_controls`. The three broad workflow error
   adapters already re-raised controls. These were the only additional confirmed
   behavior defects found in the requested sweeps.

5. **Pending later chat turns — `d19f999`.** Snapshots saved the prompt but omitted
   the pending turn used to append input to an existing transcript. They now save
   and restore `turn_message` and its consumed marker. Regression:
   `test_fresh_restore_preserves_pending_later_chat_turn` covers turns 2 and 3 at
   frame_entry, pre_model, and post_model. Input appears exactly once; saved
   post-model responses make no extra request. Follow-up `e599ac4` preserves
   snapshot capture's support for lightweight agent fixtures.

6. **Delegated owned-client cleanup — `51e407c`.** The invocation clone escaped
   without shutdown. Its complete execution/result handling now runs inside
   `try/finally`. Regression: `test_delegated_clones_close_only_owned_clients`
   covers 16 combinations of success/error/cancellation/pause, owned/borrowed
   clients, and synchronous/asynchronous clients. Owned clones close immediately;
   borrowed clients and live template clients remain open.

The shared-state sweep expanded from direct downstream execution to clone
construction, stage wiring, tool metadata, memory backends, output scopes, and
validators. Run histories, accounting dictionaries, counters, rate-limit maps,
stage containers, and validator lists are isolated. Tool callbacks, explicitly
supplied memory backends, and output subscribers are intentionally borrowed;
the framework does not mutate their definitions while executing clones. No
additional accidental runtime sharing was confirmed.

CI regression guard: run `pytest -q src/IkaTest/test_final_core_regressions.py`.
The exact broad-handler audit can be repeated with
`rg -n 'except (Exception|BaseException)' src/IkaCore src/IkaMem`; each new adapter
must preserve request controls, with the behavior regressions above as the guard.

Validation:

- 59 new regression cases, all passing in the final quality run.
- The single standalone full pytest run: 1,442 passed, 2 skipped, 3 failed.
  Its failures identified the lightweight snapshot fixture and two obsolete
  historical trace expectations described above. After correction, the affected
  targeted run passed all 65 cases.
- The single final `python scripts/quality.py` run: 1,445 passed, 2 skipped;
  Ruff passed; compilation passed; Pyright reported zero errors/warnings;
  wheel build, complete package-data inventory, and external import passed.
- Line coverage: 9,923 / 9,923 (100%). Branch coverage: 2,699 / 2,776 (97.23%).
  Combined coverage: 99.39%. All 129 modules / 16 coverage areas passed;
  zero matrix failures. Coverage thresholds and inventories were not relaxed.

Evidence is retained under `artifacts/core-review/`: per-finding before/after
logs, base/patched trace JSON and their differences, the standalone full run,
the targeted correction run, and the complete final quality log. Machine
coverage, JUnit, matrix, and wheel artifacts are under `artifacts/quality/`.
