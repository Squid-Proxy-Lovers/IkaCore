#!/usr/bin/env python3
"""Bounded fact-retention evaluation through the installed-wheel live harness.

Only numeric scores/usage are saved. Each fixture makes one summary and one
follow-up request, without retries. The original login file is never rewritten.
"""

import argparse
import contextlib
import io
import json
import logging
import os
import shutil
import sys
import tempfile
from pathlib import Path
from unittest.mock import patch

from live_provider_check import Probe, installed_identity, observe_usage

from IkaModel.chat_interface.chat_interface import chat, init_message_history
from IkaModel.codex import auth
from IkaModel.request_control import request_controls
from IkaModel.summarization import run_summarization

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src" / "IkaTest"))
from compaction_fixtures import retention_fixture  # noqa: E402


def evaluate(model, bearer, fixtures):
    probe = Probe(model, bearer)
    rows = []
    for index in range(fixtures):
        with request_controls(timeout=60, max_retries=1):
            _facts, history = retention_fixture(index)
            with observe_usage(probe) as usage:
                summary = run_summarization(probe.model(), history)
                assert summary.strip()
                # The follow-up sees exactly the compacted summary plus retained
                # structured entries; none of the discarded source turns.
                followup = "Return a JSON object with keys goal, constraint, result, question, pending. Give the exact stored values from the conversation. For result, give the ARCHIVED probe result. For question, give the unresolved approval question. Preserve numbers and identifiers. No markdown."
                compacted = json.dumps(history, ensure_ascii=False)
                response = chat(probe.model(), [{"role": "user", "content": "Compacted conversation:\n" + compacted + "\n\n" + followup}], init_message_history(), timeout=55)
                text = response["content"].strip()
                if text.startswith("```"):
                    text = text.split("\n", 1)[1].rsplit("```", 1)[0]
                try:
                    answers = json.loads(text)
                except ValueError:
                    answers = {}
                # Check must-keep facts using stable tokens, allowing the model
                # to paraphrase prose without hiding numeric/identifier loss.
                needles = {"goal": [f"sector {index + 17}"], "constraint": ["37", "source"],
                           "result": [f"AMBER-{index + 300}"], "question": ["approv", "transfer"], "pending": ["checksum", "Q9"]}
                scores = {key: all(token.casefold() in str(answers.get(key, "")).casefold() for token in tokens) for key, tokens in needles.items()}
                rows.append({"fixture": index, "facts_retained": sum(scores.values()), "facts_total": 5, "scores": scores,
                             "usage": dict(usage), "summary_tokens": history["summary"]["tokens"]})
            print(f"fixture {index}: {rows[-1]['facts_retained']}/5", flush=True)
    return rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--use-local-login", action="store_true")
    parser.add_argument("--model", default="gpt-5.6-sol")
    parser.add_argument("--fixtures", type=int, choices=range(3, 6), default=4)
    parser.add_argument("--output-dir", type=Path, default=Path("artifacts/hardening/live-compaction"))
    args = parser.parse_args()
    identity = installed_identity()
    logging.disable(logging.CRITICAL)
    bearer = os.environ.get("CODEX_BEARER")
    with tempfile.TemporaryDirectory(prefix="ikacore-eval-login-", dir=ROOT / "artifacts") as directory:
        if not bearer and args.use_local_login:
            copied = Path(directory) / "auth.json"
            shutil.copy2(auth._auth_json_path(), copied)
            copied.chmod(0o600)
            with patch.object(auth, "_auth_json_path", lambda: copied), contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
                bearer = auth.get_bearer()
        if not bearer:
            parser.error("CODEX_BEARER or --use-local-login is required")
        rows = evaluate(args.model, bearer, args.fixtures)
    report = {"model": args.model, "runtime": identity, "fixtures": rows,
              "retained": sum(row["facts_retained"] for row in rows), "total": args.fixtures * 5}
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "retention.json").write_text(json.dumps(report, indent=2) + "\n")
    print(f"retention: {report['retained']}/{report['total']}", flush=True)
    return int(report["retained"] != report["total"])


if __name__ == "__main__":
    raise SystemExit(main())
