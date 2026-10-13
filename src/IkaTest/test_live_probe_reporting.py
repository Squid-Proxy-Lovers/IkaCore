"""The opt-in live runner must keep credentials and provider bodies out of reports."""

import json
import runpy
from pathlib import Path

from IkaModel.request_control import effective_max_retries, remaining_request_time


def _runner():
    return runpy.run_path(str(Path(__file__).parents[2] / 'scripts/live_codex_check.py'))


def test_live_probe_redacts_errors_and_captured_output(capsys):
    runner = _runner()
    private = 'SHOULD_NOT_APPEAR_IN_REPORTS'
    def fail():
        print(private)
        raise RuntimeError('codex backend returned 401: ' + private)
    report = runner['run_case']('fixture', fail)
    assert report['status'] == 'failed'
    assert report['error_type'] == 'RuntimeError' and report['http_status'] == 401
    assert private not in json.dumps(report)
    assert capsys.readouterr().out == ''


def test_live_probe_bounds_requests_and_restores_controls():
    runner = _runner()
    def check():
        assert effective_max_retries(99) == 1
        assert 0 < remaining_request_time() <= 60
        return {'verified': True}
    report = runner['run_case']('fixture', check)
    assert report['status'] == 'passed' and report['verified']
    assert effective_max_retries(99) == 99 and remaining_request_time() is None
