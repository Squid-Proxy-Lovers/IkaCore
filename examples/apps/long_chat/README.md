# Preference chat across sessions

Runs twelve short turns through one agent and an explicit IkaMem MemoryTools adapter. A small JSON substring storage backend persists separately from snapshots. A 700-token context budget with compact_at_fraction=0.55 forces compaction. The final turn pauses at post_model; a new process consumes it, then searches memory for the preference.

From the repository root, after following [wheel setup](../README.md):

```bash
python -m examples.apps.long_chat.main --offline
python -m examples.apps.long_chat.main --live --use-local-login --model gpt-5.6-sol
```

Memory tools save and recall blue, at least one post_compaction snapshot exists, and a fresh process resumes the saved turn. Each `send()` appends a user turn. `ExecutionLimits(context_budget=700)` sets the token budget and `compaction_count` reports compactions. Snapshot restore retains the conversation before the next `send()`; assigning `prompt` explicitly starts a fresh task. This is a finite scripted chat loop, with CLI input replaceable by an application UI.

Offline mode runs real agent/tool/workflow/snapshot code against scripted provider-native HTTP response envelopes. It blocks accidental HTTP transport access and needs no credentials. Live mode opts in to the local Codex CLI login (or CODEX_BEARER); prompts and files are synthetic. Request scopes allow one attempt; chat sessions expire after 180 seconds, with nested 30-second turn deadlines; per-request agent timeouts are at most 45 seconds. Each subprocess session owns its own scope. The apps print only check results, never model answers or credentials.

See [live outcomes](../LIVE_RESULTS.md) and [friction and findings](../../../docs/v2/dogfood-report.md).
