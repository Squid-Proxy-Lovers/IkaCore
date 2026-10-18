# Parallel cost/risk pipeline

Uses the workflow parallel scheduler (worker threads) for seed → [cost, risk] → join. Both upstream values must reach the join compression hook. A shared caller-owned synchronous HTTPX client is lent with `ProviderConfig(client=client)`. Clones retain it and shutdown never closes it. `run(parallel=True, concurrency=2)` names the scheduling policy explicitly.

From the repository root, after following [wheel setup](../README.md):

```bash
python -m examples.apps.parallel_pipeline.main --offline
python -m examples.apps.parallel_pipeline.main --live --use-local-login --model gpt-5.6-sol
```

The fan-in result includes both values and the caller client remains open. An expired request is rejected before dispatch. --cancel-demo delivers SIGINT and proves pre-dispatch cancellation without a provider call. During an ordinary run Ctrl-C sets the same cooperative cancellation event. The demo does not claim live in-flight cancellation validation.

Offline mode runs real agent/tool/workflow/snapshot code against scripted provider-native HTTP response envelopes. It blocks accidental HTTP transport access and needs no credentials. Live mode opts in to the local Codex CLI login (or CODEX_BEARER); prompts and files are synthetic. Request scopes allow one attempt and expire after 90 seconds; per-request agent timeouts are at most 45 seconds. Each subprocess session owns its own scope. The apps print only check results, never model answers or credentials.

See [live outcomes](../LIVE_RESULTS.md) and [friction and findings](../../../docs/v2/dogfood-report.md).
