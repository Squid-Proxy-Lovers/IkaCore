#!/usr/bin/env python3
"""Check downstream imports and owned type completeness against an installed wheel."""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
# Ignore unknown external HTTPX internals, but do not allow owned annotations to regress.
FLOORS = {"IkaCore": 100.0, "IkaModel": 100.0, "IkaMem": 100.0, "IkaCore.testing": 100.0}


def check(python: Path, output: Path) -> dict[str, float]:
    output.mkdir(parents=True, exist_ok=True)
    env = dict(os.environ)
    env.pop("PYTHONPATH", None)
    scores = {}
    with tempfile.TemporaryDirectory(prefix="ikacore-consumer-") as directory:
        scratch = Path(directory)
        config = scratch / "pyrightconfig.json"
        config.write_text(json.dumps({"typeCheckingMode": "strict", "pythonVersion": "3.11"}))
        consumer = scratch / "consumer.py"
        consumer.write_text(
            "from IkaModel import BareBoneModel, chat, request_controls, IkaRequestCancelled\n"
            "from IkaCore import AgentConfig, ExecutionLimits\n"
            "model = BareBoneModel(model_id='gpt-4o-mini', api_key='fixture', api_url='https://example.invalid')\n"
            "model.system_prompt = 'Answer briefly'\n"
            "limits = ExecutionLimits(maxsteps=2)\n"
            "config = AgentConfig(name='a', description='', prompt='', model_id='', api_key='', limits=limits)\n"
            "response = chat(model, [])\n"
            "with request_controls(timeout=1):\n"
            "    error: Exception = IkaRequestCancelled('offline')\n"
        )
        command = [sys.executable, "-m", "pyright", "--pythonpath", str(python), "--outputjson"]
        valid = subprocess.run([*command, str(consumer)], cwd=scratch, env=env, capture_output=True, text=True)
        (output / "consumer-valid.json").write_text(valid.stdout)
        if valid.returncode:
            raise RuntimeError(f"typed wheel consumer failed: {valid.stdout} {valid.stderr}")
        consumer.write_text(consumer.read_text() +
                            "BareBoneModel(model_id='x', api_key='x', api_url='x', unknown_typo=True)\n")
        invalid = subprocess.run([*command, str(consumer)], cwd=scratch, env=env, capture_output=True, text=True)
        result = json.loads(invalid.stdout)
        (output / "consumer-invalid.json").write_text(invalid.stdout)
        errors = [item for item in result["generalDiagnostics"] if item["severity"] == "error"]
        if invalid.returncode != 1 or len(errors) != 1 or "unknown_typo" not in errors[0]["message"]:
            raise RuntimeError(f"wheel consumer did not reject the unknown keyword: {invalid.stdout}")
        for package, floor in FLOORS.items():
            proc = subprocess.run([*command, "--verifytypes", package, "--ignoreexternal"],
                                  cwd=scratch, env=env, capture_output=True, text=True)
            (output / f"{package}.json").write_text(proc.stdout)
            report = json.loads(proc.stdout)
            score = 100 * report["typeCompleteness"]["completenessScore"]
            scores[package] = score
            if score < floor:
                raise RuntimeError(f"{package} type completeness {score:.2f}% is below {floor:.2f}%")
    (output / "scores.json").write_text(json.dumps(scores, indent=2) + "\n")
    print(json.dumps(scores, indent=2))
    return scores


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--python", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=ROOT / "artifacts/quality/types")
    args = parser.parse_args()
    check(args.python.absolute(), args.output.resolve())
