#!/usr/bin/env python3
"""Run the repository's documented local/CI quality checks."""

from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys
import tempfile
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
ARTIFACTS = ROOT / "artifacts" / "quality"
SOURCE_FILES = sorted((ROOT / "src").rglob("*.py"))


def _commands() -> list[tuple[str, list[str]]]:
    return [
        ("ruff check .", [sys.executable, "-m", "ruff", "check", "."]),
        (
            "python -m py_compile <sources>",
            [
                sys.executable,
                "-m",
                "py_compile",
                *map(str, SOURCE_FILES),
            ],
        ),
        ("pyright", [sys.executable, "-m", "pyright"]),
        (
            "pytest --cov --cov-report=term-missing:skip-covered",
            [
                sys.executable,
                "-m",
                "pytest",
                "--cov",
                "--cov-context=test",
                f"--cov-report=json:{ARTIFACTS / 'coverage.json'}",
                f"--junitxml={ARTIFACTS / 'tests.xml'}",
                "--cov-report=term-missing:skip-covered",
            ],
        ),
        ("coverage matrix", [sys.executable, str(ROOT / "scripts" / "coverage_matrix.py")]),
    ]


def _verify_wheel() -> None:
    with tempfile.TemporaryDirectory(prefix="ikacore-wheel-") as temp:
        source_dir = Path(temp) / "source"
        source_dir.mkdir()
        for filename in ("pyproject.toml", "README.md", "LICENSE", "NOTICE"):
            shutil.copy2(ROOT / filename, source_dir / filename)
        shutil.copytree(ROOT / "src", source_dir / "src",
                        ignore=shutil.ignore_patterns("__pycache__", "*.egg-info"))
        wheel_dir = Path(temp) / "dist"
        wheel_dir.mkdir()
        print("$ python -m build --no-isolation (sdist and wheel)", flush=True)
        subprocess.run([sys.executable, "-m", "build", "--no-isolation", "--outdir", str(wheel_dir)],
                       cwd=source_dir, check=True)
        subprocess.run([sys.executable, "-m", "twine", "check", *map(str, wheel_dir.iterdir())], check=True)
        wheels = list(wheel_dir.glob("*.whl"))
        if len(wheels) != 1:
            raise RuntimeError(
                f"expected one built wheel, found {len(wheels)}"
            )

        if ARTIFACTS.exists():
            for previous in ARTIFACTS.glob("*.whl"):
                previous.unlink()
            for artifact in wheel_dir.iterdir():
                shutil.copy2(artifact, ARTIFACTS / artifact.name)
        print("$ verify wheel contains required package data", flush=True)
        with zipfile.ZipFile(wheels[0]) as archive:
            names = set(archive.namelist())
        packages = ("IkaCore", "IkaModel", "IkaMem")
        expected_modules = {path.relative_to(ROOT / "src").as_posix()
                            for package in packages for path in (ROOT / "src" / package).rglob("*.py")}
        packaged_modules = {name for name in names if name.endswith(".py") and name.split("/")[0] in packages}
        if packaged_modules != expected_modules:
            raise RuntimeError(f"wheel/source module mismatch: missing={sorted(expected_modules - packaged_modules)}, "
                               f"unexpected={sorted(packaged_modules - expected_modules)}")
        required_suffixes = (
            "IkaCore/py.typed",
            "IkaModel/py.typed",
            "IkaMem/py.typed",
            "IkaModel/summary_prompt",
            "IkaModel/data/model_metadata.json",
            ".dist-info/licenses/LICENSE",
            ".dist-info/licenses/NOTICE",
        )
        missing = [
            suffix
            for suffix in required_suffixes
            if not any(name.endswith(suffix) for name in names)
        ]
        if missing:
            raise RuntimeError(
                f"wheel is missing package data: {missing}"
            )

        print("$ verify built wheel imports outside source tree", flush=True)
        runtime = Path(temp) / "runtime"
        # Share tool dependencies; the noneditable wheel is installed in this environment's
        # own site-packages and must win over any development installation.
        subprocess.run([sys.executable, "-m", "venv", "--without-pip", str(runtime)],
                       check=True)
        python = runtime / ("Scripts/python.exe" if os.name == "nt" else "bin/python")
        subprocess.run([sys.executable, "-m", "pip", "--python", str(python), "install",
                        "--ignore-installed", str(wheels[0])], check=True)
        env = dict(os.environ)
        env.pop("PYTHONPATH", None)
        subprocess.run([str(python), "-c",
                        "import IkaCore, IkaModel, IkaMem; from importlib.metadata import version; "
                        "assert 'site-packages' in IkaCore.__file__; "
                        "assert IkaCore.__version__ == IkaModel.__version__ == IkaMem.__version__ == version('ikacore')"],
                       cwd=temp, env=env, check=True)
        subprocess.run([sys.executable, str(ROOT / "scripts/check_public_types.py"),
                        "--python", str(python), "--output", str(ARTIFACTS / "types")],
                       cwd=temp, env=env, check=True)
        subprocess.run([str(python), str(ROOT / "scripts/check_offline_wheel.py")],
                       cwd=temp, env=env, check=True)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--list",
        action="store_true",
        help="print checks without running them",
    )
    args = parser.parse_args()
    labels = [label for label, _ in _commands()] + [
        "python -m build --no-isolation (sdist and wheel)",
        "twine check wheel and sdist",
        "installed-wheel public consumer and verifytypes floors",
        "installed-wheel offline docs/examples/apps",
        "verify wheel contains required package data",
        "verify built wheel imports outside source tree",
    ]
    if args.list:
        print("\n".join(labels))
        return 0
    ARTIFACTS.mkdir(parents=True, exist_ok=True)
    for label, command in _commands():
        print(f"$ {label}", flush=True)
        subprocess.run(command, cwd=ROOT, check=True)
    _verify_wheel()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
