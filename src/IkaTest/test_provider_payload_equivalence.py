"""Compare all provider benchmark payloads against the pre-optimization runtime."""

import hashlib
import json
import runpy
from copy import deepcopy
from pathlib import Path

from IkaModel.chat_helpers_common import PROVIDER_REQUEST_BUILDERS, build_provider_request


def test_all_provider_payloads_match_baseline_without_mutating_inputs():
    harness = runpy.run_path(str(Path(__file__).parents[2] / 'scripts/provider_benchmark.py'))
    baseline = json.loads((Path(__file__).parent / 'fixtures/provider_payload_hashes.json').read_text())
    assert set(harness['PROVIDERS']) == set(PROVIDER_REQUEST_BUILDERS)
    for key, expected in baseline.items():
        provider, messages, tools = key.split(':')
        model, transcript, history = harness['fixture'](provider, int(messages), int(tools))
        original = deepcopy((transcript, history))
        payload = build_provider_request(provider, model, transcript, history)[2]
        canonical = json.dumps(payload, sort_keys=True, separators=(',', ':')).encode()
        assert hashlib.sha256(canonical).hexdigest() == expected, key
        assert (transcript, history) == original, key
