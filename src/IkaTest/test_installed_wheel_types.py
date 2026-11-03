"""A downstream checker must see the installed wheel, outside this repository."""
import json
import os
import subprocess
import sys
from pathlib import Path


def test_installed_wheel_rejects_unknown_model_keyword(tmp_path):
    root = Path(__file__).resolve().parents[2]
    env = dict(os.environ)
    env.pop("PYTHONPATH", None)
    subprocess.run([sys.executable, "-m", "pip", "wheel", str(root), "--no-deps",
                    "--no-build-isolation", "-w", str(tmp_path)], env=env,
                   check=True, capture_output=True)
    runtime = tmp_path / "runtime"
    subprocess.run([sys.executable, "-m", "venv", "--system-site-packages", "--without-pip", str(runtime)],
                   check=True, capture_output=True)
    python = runtime / ("Scripts/python.exe" if os.name == "nt" else "bin/python")
    wheel, = tmp_path.glob("*.whl")
    subprocess.run([sys.executable, "-m", "pip", "--python", str(python), "install",
                    "--no-deps", "--ignore-installed", str(wheel)], env=env, check=True, capture_output=True)
    probe = subprocess.check_output([str(python), "-c", "import IkaModel; print(IkaModel.__file__)"],
                                    cwd=tmp_path, env=env, text=True)
    assert str(runtime) in probe and "site-packages" in probe
    snippet = tmp_path / "consumer.py"
    snippet.write_text("from IkaModel import BareBoneModel\n"
                       "BareBoneModel(model_id='x', api_key='x', api_url='x', unknown_typo=True)\n")
    (tmp_path / "pyrightconfig.json").write_text('{"typeCheckingMode": "strict"}')
    result = subprocess.run([sys.executable, "-m", "pyright", "--pythonpath", str(python),
                             "--outputjson", str(snippet)], cwd=tmp_path, env=env,
                            capture_output=True, text=True)
    errors = [item for item in json.loads(result.stdout)["generalDiagnostics"] if item["severity"] == "error"]
    assert result.returncode == 1 and len(errors) == 1
    assert "unknown_typo" in errors[0]["message"]
