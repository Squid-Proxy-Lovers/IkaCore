#!/usr/bin/env python3
"""Measure offline framework footprint and overhead in fresh subprocesses."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import statistics
import subprocess
import sys
import time
import zipfile
from datetime import datetime, timezone
from email.parser import BytesParser
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CASES = ('python_baseline', 'import_public', 'import_httpx', 'import_runtime', 'agent_construct', 'httpx_roundtrip',
         'payload_100_messages', 'tool_sync', 'tool_async', 'agent_sync', 'agent_async', 'snapshot_append',
         'agent_snapshots', 'workflow_sync', 'workflow_parallel')


def summary(values):
    ordered = sorted(values)
    return {'median': statistics.median(values), 'min': ordered[0], 'max': ordered[-1],
            'p95': ordered[min(len(ordered) - 1, int((len(ordered) - 1) * 0.95 + 0.5))]}


def run_worker(python, case, iterations, env):
    started = time.perf_counter_ns()
    result = subprocess.run([python, str(ROOT / 'scripts/benchmark_worker.py'), '--case', case,
                             '--iterations', str(iterations)], env=env, capture_output=True, text=True,
                            check=True, timeout=120)
    value = json.loads(result.stdout)
    value['process_wall_ns'] = time.perf_counter_ns() - started
    return value


def environment(python, env, expected, source=None):
    code = '''import hashlib, importlib.metadata as m, json, os, platform, subprocess, sys
from pathlib import Path
inputs = json.load(sys.stdin)
dist = m.distribution("ikacore")
for name, expected in inputs["hashes"].items():
    path = Path(inputs["source"]) / name if inputs["source"] else dist.locate_file(name)
    if not path.is_file() or hashlib.sha256(path.read_bytes()).hexdigest() != expected:
        raise SystemExit("installed/source file differs from benchmark wheel: " + name)
cpu = platform.processor() or "unknown"
if platform.system() == "Darwin":
    cpu = subprocess.run(["sysctl", "-n", "machdep.cpu.brand_string"], capture_output=True, text=True, check=False).stdout.strip() or cpu
elif Path("/proc/cpuinfo").is_file():
    cpu = next((line.partition(":")[2].strip() for line in Path("/proc/cpuinfo").read_text().splitlines() if line.startswith("model name")), cpu)
rows = []
for d in m.distributions():
    size = sum(d.locate_file(f).stat().st_size for f in d.files or [] if d.locate_file(f).is_file())
    rows.append({"name": d.metadata["Name"], "version": d.version, "installed_bytes": size})
print(json.dumps({"python": platform.python_version(), "platform": platform.system(), "machine": platform.machine(), "cpu": cpu, "logical_cpus": os.cpu_count(), "distributions": rows}))'''
    result = subprocess.run([python, '-c', code], input=json.dumps({'hashes': expected, 'source': source}),
                            capture_output=True, text=True, env=env, check=True)
    return json.loads(result.stdout)


def render(report):
    lines = ['# Offline benchmark', '', f"Environment: Python {report['environment']['python']}, {report['environment']['platform']} {report['environment']['machine']}",
             f"Fresh subprocess samples per case: {report['samples']}; warm operations per sample: {report['iterations']}.", '',
             'No provider latency or billing. Network connections are rejected. Engine cases construct/run/shutdown a fresh agent with HTTPX MockTransport.',
             'Import timings exclude interpreter startup; process wall time is stored separately in JSON. RSS includes Python and the measurement worker.',
             'P95 is across subprocess batch averages, not per-request tail latency. Tiny sample sets are not confidence intervals.', '',
             '| Case | Median ms/op | P95 ms/op | Peak RSS MiB (median) |', '| --- | ---: | ---: | ---: |']
    for case, data in report['cases'].items():
        lines.append(f"| {case} | {data['ns_per_operation']['median'] / 1e6:.4f} | {data['ns_per_operation']['p95'] / 1e6:.4f} | {data['peak_rss_bytes']['median'] / 2**20:.2f} |")
    lines += ['', f"Wheel: {report['wheel']}", '', f"Repeated-run retained allocations: {report['soak']}", '',
              'HTTPX baseline includes constructing a mock client and decoding its response. Snapshot cases include real local SQLite writes.',
              'Workflow cases measure a four-node diamond using pure fixtures; they measure scheduling overhead rather than model execution.', '',
              'Resource-budget failures: ' + (', '.join(report['budget_failures']) or 'none'), '']
    return '\n'.join(lines)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--python', default=sys.executable)
    parser.add_argument('--source', type=Path, help='optional source override; omit for an installed-wheel measurement')
    parser.add_argument('--wheel', type=Path, required=True)
    parser.add_argument('--samples', type=int, default=7)
    parser.add_argument('--iterations', type=int, default=40)
    parser.add_argument('--soak-iterations', type=int, default=500)
    parser.add_argument('--output', type=Path, default=ROOT / 'artifacts/benchmarks')
    args = parser.parse_args()
    if min(args.samples, args.iterations, args.soak_iterations) < 1:
        parser.error('counts must be positive')
    env = dict(os.environ)
    for name in ('PYTHONPATH', 'IKA_DUMP_REQUESTS', 'IKA_TOOL_RESULT_MAX_CHARS', 'IKA_CODEX_TOOL_OUTPUT_MAX_CHARS'):
        env.pop(name, None)
    env['PYTHONDONTWRITEBYTECODE'] = '1'
    if args.source:
        env['PYTHONPATH'] = str(args.source.resolve())
    with zipfile.ZipFile(args.wheel) as archive:
        expected = {name: hashlib.sha256(archive.read(name)).hexdigest() for name in archive.namelist()
                    if not name.endswith('/') and name.split('/')[0] in {'IkaCore', 'IkaModel', 'IkaMem'}}
    report = {'created_at_utc': datetime.now(timezone.utc).isoformat(),
              'environment': environment(args.python, env, expected, str(args.source.resolve()) if args.source else None), 'samples': args.samples, 'iterations': args.iterations,
              'source_override': bool(args.source), 'cases': {}, 'budget_failures': []}
    report['worktree_dirty'] = bool(subprocess.run(['git', 'status', '--porcelain'], cwd=ROOT, capture_output=True, text=True, check=True).stdout)
    report['harness_sha256'] = hashlib.sha256(b''.join((ROOT / 'scripts' / name).read_bytes() for name in
                                                     ('benchmark.py', 'benchmark_worker.py', 'benchmark_cases.py'))).hexdigest()
    report['revision'] = subprocess.run(['git', 'rev-parse', 'HEAD'], cwd=ROOT, capture_output=True, text=True, check=True).stdout.strip()
    for case in CASES:
        raw = [run_worker(args.python, case, args.iterations, env) for _ in range(args.samples)]
        report['cases'][case] = {key: summary([r[key] for r in raw]) for key in ('ns_per_operation', 'peak_rss_bytes', 'process_wall_ns')}
        report['cases'][case]['samples'] = raw
        print(f"{case}: {report['cases'][case]['ns_per_operation']['median'] / 1e6:.4f} ms/op", flush=True)
    report['soak'] = run_worker(args.python, 'agent_soak', args.soak_iterations, env)
    with zipfile.ZipFile(args.wheel) as archive:
        metadata_file = next(name for name in archive.namelist() if name.endswith('.dist-info/METADATA'))
        metadata = BytesParser().parsebytes(archive.read(metadata_file))
        runtime_requires = [r for r in metadata.get_all('Requires-Dist', []) if not re.search(r'\bextra\s*==', r)]
        report['wheel'] = {'sha256': hashlib.sha256(args.wheel.read_bytes()).hexdigest(),
                           'runtime_requires': runtime_requires, 'compressed_bytes': args.wheel.stat().st_size,
                           'uncompressed_bytes': sum(i.file_size for i in archive.infolist()),
                           'python_modules': sum(i.filename.endswith('.py') for i in archive.infolist())}
    # Portable footprint guardrails. Timings are evidence, not noisy cross-machine CI gates.
    if len(runtime_requires) > 1:
        report['budget_failures'].append('more than one direct runtime dependency')
    if report['wheel']['compressed_bytes'] > 300 * 1024:
        report['budget_failures'].append('wheel exceeds 300 KiB')
    if report['wheel']['uncompressed_bytes'] > 1024 * 1024:
        report['budget_failures'].append('installed wheel content exceeds 1 MiB')
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / 'benchmark.json').write_text(json.dumps(report, indent=2) + '\n')
    (args.output / 'benchmark.md').write_text(render(report))
    return int(bool(report['budget_failures']))


if __name__ == '__main__':
    raise SystemExit(main())
