#!/usr/bin/env python3
"""Test script to verify paths are correct."""

import sys
from pathlib import Path

# Test the path calculation
file_path = Path(__file__)
print(f"File: {file_path}")
print(f"Parent: {file_path.parent}")
print(f"Parent.parent: {file_path.parent.parent}")
print(f"Parent.parent.parent: {file_path.parent.parent.parent}")
print(f"Expected src path: {file_path.parent.parent.parent / 'src'}")

# Add to path
sys.path.insert(0, str(file_path.parent.parent.parent / "src"))
print(f"\nAdded to sys.path: {file_path.parent.parent.parent / 'src'}")

# Try importing
try:
    from IkaCore.agents import IkaBaseAgent
    print("✓ IkaCore.agents import successful")
except Exception as e:
    print(f"✗ IkaCore.agents import failed: {e}")

try:
    from IkaCore.tools import IkaTools
    print("✓ IkaCore.tools import successful")
except Exception as e:
    print(f"✗ IkaCore.tools import failed: {e}")
