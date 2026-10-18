"""Historical scenarios with reviewed v2 corrections to persisted step accounting."""
import hashlib
import importlib.util
import json
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]


@pytest.mark.parametrize('staged', [False, True])
def test_execution_parity_with_dc9efbd(staged, tmp_path):
    path = ROOT / 'scripts/execution_parity.py'
    spec = importlib.util.spec_from_file_location('execution_parity_fixture', path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    fixtures = Path(__file__).parent / 'fixtures'
    corrected = json.loads((fixtures / 'execution-parity-v2-core-fixes.json').read_text())
    for index, (name, maxsteps, extensions, pause) in enumerate(module.cases()):
        folder = tmp_path / str(index)
        folder.mkdir()
        raw = module.run_case(folder, staged, name, maxsteps, extensions, pause)
        assert raw["result"].pop("validation") == {"status": "not_run", "attempts": 0, "failed_checks": []}
        trace = module.normalize(raw)
        actual = hashlib.sha256(json.dumps(trace, sort_keys=True, separators=(',', ':')).encode()).hexdigest()
        case = f'{"stage" if staged else "simple"}:{name}'
        assert actual == corrected['sha256'][case], f'execution diverged from reviewed budget traces: {case}\n{trace}'
        # Keep the pre-fix model calls, outputs, usage and summaries byte-identical
        # in every scenario that only gained corrected snapshot accounting.
        if case in corrected['preserved_behavior_sha256']:
            observable = {key: value for key, value in trace.items() if key != 'snapshots'}
            digest = hashlib.sha256(json.dumps(observable, sort_keys=True, separators=(',', ':')).encode()).hexdigest()
            assert digest == corrected['preserved_behavior_sha256'][case], case
        assert all(boundary.get('remaining_steps', 0) >= 0 for boundary in trace['snapshots'])
