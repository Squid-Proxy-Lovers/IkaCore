"""Keep every supported export and public method signature discoverable."""
import importlib.util
from pathlib import Path


def test_public_api_reference_matches_exports_and_signatures():
    root = Path(__file__).resolve().parents[2]
    spec = importlib.util.spec_from_file_location("api_reference", root / "scripts/generate_api_reference.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    assert (root / "docs/API.md").read_text() == module.generate(), "Run python scripts/generate_api_reference.py"
