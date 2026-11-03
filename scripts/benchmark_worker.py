"""Fresh-process measurement worker. Results contain synthetic data and no credentials."""

from __future__ import annotations

import argparse
import gc
import json
import resource
import socket
import sys
import tempfile
import threading
import time
from pathlib import Path


def deny_network(*args, **kwargs):
    raise RuntimeError('benchmark attempted a real network connection')


def rss_bytes():
    peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return int(peak if sys.platform == 'darwin' else peak * 1024)


def imports(case):
    import importlib
    start = time.perf_counter_ns()
    if case == 'import_public':
        for name in ('IkaCore', 'IkaModel', 'IkaMem'):
            importlib.import_module(name)
    elif case == 'import_httpx':
        importlib.import_module('httpx')
    elif case == 'import_runtime':
        getattr(importlib.import_module('IkaCore'), 'IkaBaseAgent')
    elif case != 'python_baseline':
        raise ValueError(case)
    return {'ns_per_operation': time.perf_counter_ns() - start}


def measure(operation, iterations):
    import inspect
    if inspect.iscoroutinefunction(operation):
        import asyncio
        async def run():
            for _ in range(3):
                await operation()
            start = time.perf_counter_ns()
            for _ in range(iterations):
                await operation()
            return (time.perf_counter_ns() - start) / iterations
        return asyncio.run(run())
    for _ in range(3):
        operation()
    start = time.perf_counter_ns()
    for _ in range(iterations):
        operation()
    return (time.perf_counter_ns() - start) / iterations


def soak(operation, iterations):
    import tracemalloc
    for _ in range(20):
        operation()
    gc.collect()
    tracemalloc.start()
    retained = [tracemalloc.get_traced_memory()[0]]
    for _ in range(2):
        for _ in range(iterations):
            operation()
        gc.collect()
        retained.append(tracemalloc.get_traced_memory()[0])
    _, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    return {'iterations_per_phase': iterations, 'retained_growth_bytes': [retained[1] - retained[0], retained[2] - retained[1]],
            'traced_peak_bytes': peak}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--case', required=True)
    parser.add_argument('--iterations', type=int, default=40)
    args = parser.parse_args()
    if args.iterations < 1:
        parser.error('iterations must be positive')
    socket.socket.connect = deny_network
    with tempfile.TemporaryDirectory(prefix='ikacore-bench-') as temp:
        if args.case.startswith('import_') or args.case == 'python_baseline':
            result = imports(args.case)
        else:
            from benchmark_cases import setup
            operation, cleanup = setup(args.case, Path(temp))
            try:
                result = soak(operation, args.iterations) if args.case == 'agent_soak' else {
                    'ns_per_operation': measure(operation, args.iterations)}
            finally:
                cleanup()
        result.update({'case': args.case, 'peak_rss_bytes': rss_bytes(), 'threads_before_exit': threading.active_count()})
        print(json.dumps(result))


if __name__ == '__main__':
    main()
