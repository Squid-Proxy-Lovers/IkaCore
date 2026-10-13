#!/usr/bin/env python3
"""Report independent line/branch coverage and verified scenarios for every runtime module."""

from __future__ import annotations

import argparse
import json
import sys
import xml.etree.ElementTree as ET
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
METRICS = ('covered_lines', 'num_statements', 'covered_branches', 'num_branches', 'excluded_lines')


def percent(covered: int, total: int) -> float:
    return 100 * covered / total if total else 100.0


def junit_results(path: Path) -> dict[str, str]:
    results = {}
    for case in ET.parse(path).iter('testcase'):
        parts = case.attrib['classname'].split('.')
        module = next(i for i, part in enumerate(parts) if part.startswith('test_'))
        node = '/'.join(parts[:module + 1]) + '.py::' + '::'.join([*parts[module + 1:], case.attrib['name']])
        result = 'skipped' if case.find('skipped') is not None else 'passed'
        if case.find('failure') is not None or case.find('error') is not None:
            result = 'failed'
        results[node] = result
    return results


def validate_inventory(files: dict, areas: list[dict], source_files: set[str]) -> None:
    assigned = Counter(path for area in areas for path in area['sources'])
    duplicates = [path for path, count in assigned.items() if count != 1]
    if duplicates or set(assigned) != source_files or set(files) != source_files:
        raise ValueError(f'coverage inventory mismatch: duplicates={duplicates}, '
                         f'unassigned={sorted(source_files - set(assigned))}, '
                         f'stale_manifest={sorted(set(assigned) - source_files)}, '
                         f'missing_report={sorted(source_files - set(files))}, '
                         f'stale_report={sorted(set(files) - source_files)}')


def area_result(area: dict, files: dict, tests: dict[str, str]) -> dict:
    totals = {key: sum(files[path]['summary'][key] for path in area['sources']) for key in METRICS}
    observed = {context.split('|')[0] for path in area['sources']
                for contexts in files[path].get('contexts', {}).values() for context in contexts if context}
    scenarios = {}
    for selector in area['required_tests']:
        matches = [name for name in tests if name == selector or name.startswith(selector + '[')]
        scenarios[selector] = bool(matches) and all(tests[name] == 'passed' for name in matches)
    return {**area, **totals, 'line_percent': percent(totals['covered_lines'], totals['num_statements']),
            'branch_percent': percent(totals['covered_branches'], totals['num_branches']),
            'observed_tests': sorted(observed), 'scenarios': scenarios,
            'missing': {path: {'lines': files[path]['missing_lines'], 'branches': files[path].get('missing_branches', [])}
                        for path in area['sources'] if files[path]['missing_lines'] or files[path].get('missing_branches')}}


def build_matrix(coverage: dict, tests: dict[str, str], manifest: dict, source_files: set[str]) -> dict:
    if not coverage['meta'].get('branch_coverage'):
        raise ValueError('branch coverage must be enabled')
    validate_inventory(coverage['files'], manifest['areas'], source_files)
    if not any('contexts' in data for data in coverage['files'].values()):
        raise ValueError('test contexts must be recorded')
    areas = [area_result(area, coverage['files'], tests) for area in manifest['areas']]
    totals = coverage['totals']
    report = {'areas': areas, 'totals': totals, 'test_results': dict(Counter(tests.values())), 'failures': []}
    global_metrics = {'area': 'All runtime', 'line_percent': percent(totals['covered_lines'], totals['num_statements']),
                      'branch_percent': percent(totals['covered_branches'], totals['num_branches']), **manifest}
    for area in [*areas, global_metrics]:
        for kind in ('line', 'branch'):
            floor = area['min_lines' if kind == 'line' else 'min_branches']
            if area[kind + '_percent'] + 1e-9 < floor:
                report['failures'].append(f"{area['area']}: {kind} coverage {area[kind + '_percent']:.2f}% < {floor}%")
    for area in areas:
        report['failures'].extend('Required scenario did not pass: ' + name for name, ok in area['scenarios'].items() if not ok)
    if 'failed' in tests.values():
        report['failures'].append('JUnit contains failed tests')
    return report


def markdown(report: dict) -> str:
    lines = ['# Runtime coverage matrix', '', 'Line and branch percentages are separate. Test counts indicate observed execution, not independent proof of correctness.', '',
             '| Area | Modules | Lines | Branches | Floors L/B | Tests observed | Excluded lines |',
             '| --- | ---: | ---: | ---: | ---: | ---: | ---: |']
    for area in report['areas']:
        lines.append(f"| {area['area']} | {len(area['sources'])} | {area['line_percent']:.2f}% | {area['branch_percent']:.2f}% | "
                     f"{area['min_lines']}/{area['min_branches']}% | {len(area['observed_tests'])} | {area['excluded_lines']} |")
    lines += ['', f"JUnit results: {report['test_results']}", '', '## Required scenarios and known gaps']
    for area in report['areas']:
        lines += ['', '### ' + area['area'], '', area['known_gap'], '']
        lines.extend(f"- {'PASS' if ok else 'MISSING / FAILED / SKIPPED'}: `{name}`" for name, ok in area['scenarios'].items())
        lines.append(f"- Uncovered: {sum(len(v['lines']) for v in area['missing'].values())} lines, "
                     f"{sum(len(v['branches']) for v in area['missing'].values())} branches (exact locations in coverage-matrix.json).")
    lines += ['', '## Gate result', '', '\n'.join(report['failures']) if report['failures'] else 'PASS', '']
    return '\n'.join(lines)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--coverage', type=Path, default=ROOT / 'artifacts/quality/coverage.json')
    parser.add_argument('--junit', type=Path, default=ROOT / 'artifacts/quality/tests.xml')
    parser.add_argument('--manifest', type=Path, default=Path(__file__).with_name('coverage_manifest.json'))
    parser.add_argument('--output', type=Path, default=ROOT / 'artifacts/quality')
    args = parser.parse_args()
    sources = {p.relative_to(ROOT).as_posix() for package in ('IkaCore', 'IkaModel', 'IkaMem')
               for p in (ROOT / 'src' / package).rglob('*.py')}
    report = build_matrix(json.loads(args.coverage.read_text()), junit_results(args.junit),
                          json.loads(args.manifest.read_text()), sources)
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / 'coverage-matrix.json').write_text(json.dumps(report, indent=2) + '\n')
    (args.output / 'coverage-matrix.md').write_text(markdown(report))
    print(f"Coverage matrix: {len(sources)} modules, {len(report['areas'])} areas; "
          f"{len(report['failures'])} gate failures. Report: {args.output / 'coverage-matrix.md'}")
    for failure in report['failures']:
        print(failure, file=sys.stderr)
    return int(bool(report['failures']))


if __name__ == '__main__':
    raise SystemExit(main())
