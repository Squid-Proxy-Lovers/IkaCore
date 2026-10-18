# Durable human approval workflow

Runs plan → approval → prepare with dependencies. The demo launches three separate Python processes: start persists a HITL pause and exits; resume answers YES using resume_input; fork clones the saved workflow boundary and answers NO. Preparation is synthetic and publishes nothing.

From the repository root, after following [wheel setup](../README.md):

```bash
python -m examples.apps.approval_workflow.main --offline
python -m examples.apps.approval_workflow.main --live --use-local-login --model gpt-5.6-sol
```

The dependent remains blocked at the question; a new process resumes to READY; the alternative fork produces HOLD. Completed upstream work is reused. The alternative calls workflow.fork_from_snapshot(snapshot_id, resume_input="NO").

Offline mode runs real agent/tool/workflow/snapshot code against scripted provider-native HTTP response envelopes. It blocks accidental HTTP transport access and needs no credentials. Live mode opts in to the local Codex CLI login (or CODEX_BEARER); prompts and files are synthetic. Request scopes allow one attempt and expire after 90 seconds; per-request agent timeouts are at most 45 seconds. Each subprocess session owns its own scope. The apps print only check results, never model answers or credentials.

See [live outcomes](../../../docs/v2/validation/archive/apps-live-results.md) and [usage guide](../../../docs/USAGE.md).
