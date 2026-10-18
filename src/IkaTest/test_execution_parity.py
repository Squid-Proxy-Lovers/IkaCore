"""The same 42 normalized traces produced by the unmodified dc9efbd runtime."""
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
    baseline = json.loads((Path(__file__).parent / 'fixtures/execution-parity-dc9efbd.json').read_text())
    for index, (name, maxsteps, extensions, pause) in enumerate(module.cases()):
        folder = tmp_path / str(index)
        folder.mkdir()
        raw = module.run_case(folder, staged, name, maxsteps, extensions, pause)
        assert raw["result"].pop("validation") == {"status": "not_run", "attempts": 0, "failed_checks": []}
        trace = module.normalize(raw)
        actual = hashlib.sha256(json.dumps(trace, sort_keys=True, separators=(',', ':')).encode()).hexdigest()
        case = f'{"stage" if staged else "simple"}:{name}'
        assert actual == baseline['sha256'][case], f'execution diverged from dc9efbd: {case}\n{trace}'
