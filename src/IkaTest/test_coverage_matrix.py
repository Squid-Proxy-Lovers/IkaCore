"""The quality report must fail on uncovered branches or missing scenarios."""

import copy
import runpy
from pathlib import Path

import pytest

MATRIX = runpy.run_path(str(Path(__file__).resolve().parents[2] / 'scripts/coverage_matrix.py'))


def inputs():
    summary = {'covered_lines': 9, 'num_statements': 10, 'covered_branches': 1, 'num_branches': 2, 'excluded_lines': 3}
    coverage = {'meta': {'branch_coverage': True}, 'totals': summary,
                'files': {'src/module.py': {'summary': summary, 'missing_lines': [10], 'missing_branches': [[9, 10]],
                                          'contexts': {'1': ['test.py::test_case|run']}}}}
    area = {'area': 'sample', 'sources': ['src/module.py'], 'required_tests': ['test.py::test_case'],
            'min_lines': 90, 'min_branches': 80, 'known_gap': 'fixture'}
    return coverage, {'test.py::test_case': 'passed'}, {'areas': [area], 'min_lines': 90, 'min_branches': 80}


def test_branch_floor_fails_even_when_line_floor_passes():
    report = MATRIX['build_matrix'](*inputs(), {'src/module.py'})
    assert report['areas'][0]['line_percent'] == 90
    assert report['areas'][0]['branch_percent'] == 50
    assert len(report['failures']) == 2 and all('branch coverage' in f for f in report['failures'])


@pytest.mark.parametrize('mode', ['unassigned', 'duplicate', 'stale', 'missing_report'])
def test_inventory_requires_every_source_exactly_once(mode):
    coverage, tests, manifest = copy.deepcopy(inputs())
    if mode == 'unassigned':
        manifest['areas'][0]['sources'] = []
    elif mode == 'duplicate':
        manifest['areas'].append(manifest['areas'][0])
    elif mode == 'stale':
        manifest['areas'][0]['sources'].append('missing.py')
    else:
        coverage['files'] = {}
    with pytest.raises(ValueError, match='inventory mismatch'):
        MATRIX['build_matrix'](coverage, tests, manifest, {'src/module.py'})


@pytest.mark.parametrize('status', ['skipped', 'failed'])
def test_required_scenarios_must_pass(status):
    coverage, tests, manifest = inputs()
    tests['test.py::test_case'] = status
    report = MATRIX['build_matrix'](coverage, tests, manifest, {'src/module.py'})
    assert any('Required scenario' in failure for failure in report['failures'])
