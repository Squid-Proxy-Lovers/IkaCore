#!/usr/bin/env python3
"""Compatibility entry point; install a wheel or use live_matrix.py first."""

import sys

from live_provider_check import main

if __name__ == '__main__':
    raise SystemExit(main(['--provider', 'codex', *sys.argv[1:]]))
