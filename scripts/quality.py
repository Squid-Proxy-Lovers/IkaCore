#!/usr/bin/env python3
"""Run the repository's documented local/CI quality checks."""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
import tempfile
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SOURCE_FILES = sorted((ROOT / "src").rglob("*.py"))


def _commands() -> list[tuple[str, list[str]]]:
    return [
        ("ruff check .", [sys.executable, "-m", "ruff", "check", "."]),
        ("python -m py_compile <sources>", [sys.executable, "-m", "py_compile", *map(str, SOURCE_FILES)]),
        ("pyright", [sys.executable, "-m", "pyright"]),
        (
            "pytest --cov --cov-report=term-missing:skip-covered",
            [sys.executable, "-m", "pytest", "--cov", "--cov-report=term-missing:skip-covered"],
        ),
    ]


def _verify_wheel() -> None:
    with tempfile.TemporaryDirectory(prefix="ikacore-wheel-") as temp:
        wheel_dir = Path(temp) / "dist"
        wheel_dir.mkdir()
        print("$ pip wheel . --no-deps --no-build-isolation", flush=True)
        subprocess.run(
            [sys.executable, "-m", "pip", "wheel", ".", "--no-deps", "--no-build-isolation", "-w", str(wheel_dir)],
            cwd=ROOT,
            check=True,
        )
        wheels = list(wheel_dir.glob("*.whl"))
        if len(wheels) != 1:
            raise RuntimeError(f"expected one built wheel, found {len(wheels)}")

        print("$ verify wheel contains required package data", flush=True)
        with zipfile.ZipFile(wheels[0]) as archive:
            names = set(archive.namelist())
        required_suffixes = ("IkaModel/summary_prompt", "IkaModel/data/model_metadata.json")
        missing = [suffix for suffix in required_suffixes if not any(name.endswith(suffix) for name in names)]
        if missing:
            raise RuntimeError(f"wheel is missing package data: {missing}")

        print("$ verify built wheel imports outside source tree", flush=True)
        install_dir = Path(temp) / "installed"
        subprocess.run(
            [sys.executable, "-m", "pip", "install", "--no-deps", "--target", str(install_dir), str(wheels[0])],
            check=True,
        )
        env = dict(os.environ)
        env["PYTHONPATH"] = str(install_dir)
        subprocess.run(
            [sys.executable, "-c", "import IkaCore, IkaModel, IkaMem"],
            cwd=temp,
            env=env,
            check=True,
        )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--list", action="store_true", help="print checks without running them")
    args = parser.parse_args()
    labels = [label for label, _ in _commands()] + [
        "pip wheel . --no-deps --no-build-isolation",
        "verify wheel contains required package data",
        "verify built wheel imports outside source tree",
    ]
    if args.list:
        print("\n".join(labels))
        return 0
    for label, command in _commands():
        print(f"$ {label}", flush=True)
        subprocess.run(command, cwd=ROOT, check=True)
    _verify_wheel()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
