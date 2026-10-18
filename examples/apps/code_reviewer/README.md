# Disposable repository reviewer

Creates a temporary repository with an empty-input division bug and long irrelevant notes. Sandboxed read tools inspect it; a structured submit_review tool validates the JSON. A finite synthetic lint probe intentionally exceeds its 5 ms tool timeout. Tool result caps are 1,200 characters.

From the repository root, after following [wheel setup](../README.md):

```bash
python -m examples.apps.code_reviewer.main --offline
python -m examples.apps.code_reviewer.main --live --use-local-login --model gpt-5.6-sol
```

Both files are read, the submitted review names calc.py and its empty-input problem. Offline mode additionally inspects provider payloads for truncation and timeout feedback. The expected timeout warning currently prints even with cli=False.

Offline mode runs real agent/tool/workflow/snapshot code against scripted provider-native HTTP response envelopes. It blocks accidental HTTP transport access and needs no credentials. Live mode opts in to the local Codex CLI login (or CODEX_BEARER); prompts and files are synthetic. Request scopes allow one attempt and expire after 90 seconds; per-request agent timeouts are at most 45 seconds. Each subprocess session owns its own scope. The apps print only check results, never model answers or credentials.

See [live outcomes](../LIVE_RESULTS.md) and [friction and findings](../../../docs/v2/dogfood-report.md).
