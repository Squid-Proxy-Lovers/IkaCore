"""Deterministic generative invariants for persistent state and transcript replay."""

import copy
import json
import random
from collections import Counter

from IkaCore.snapshot_store import SnapshotStore
from IkaModel.history_replay import history_entries_to_replay


def test_snapshot_json_roundtrip_and_append_only_chain(tmp_path):
    rng = random.Random(47219)
    store = SnapshotStore(str(tmp_path / 'state.db'))
    run = store.create_run('agent', 'roundtrip')
    frame = store.create_frame(run, 'agent', 'roundtrip')
    ids = []
    states = []
    for index in range(40):
        state = {'index': index, 'text': ''.join(rng.choices('aΩ🙂\\\n\"', k=rng.randrange(40))),
                 'nested': [None, bool(index % 2), rng.randrange(-100000, 100000), {'values': [rng.random()]}]}
        before = copy.deepcopy(state)
        snapshot = store.create_snapshot(run, frame, 'boundary', state, prev_snapshot_id=ids[-1] if ids else None)
        assert store.load_snapshot(snapshot)['state_json'] == state == before
        ids.append(snapshot)
        states.append(state)
    chain = store.get_snapshot_chain(ids[-1])
    assert [row['snapshot_id'] for row in chain] == ids
    assert [row['state_json'] for row in chain] == states
    assert len(set(ids)) == len(ids)


def test_history_replay_conserves_multiplicity_order_and_inputs():
    rng = random.Random(92014)
    variants = [{'role': 'tool', 'content': str(index), 'tool_call_id': str(index)} for index in range(7)]
    for _ in range(200):
        persisted = rng.choices(variants, k=rng.randrange(30))
        live = rng.choices(variants, k=rng.randrange(30))
        entries = [{'type': 'tool', 'message': json.dumps(item), 'position': i} for i, item in enumerate(persisted)]
        history = {'messages': {str(i): item for i, item in enumerate(entries)}}
        before = copy.deepcopy((history, live))
        remaining = history_entries_to_replay(history, live)
        expected = Counter(json.dumps(item) for item in persisted) - Counter(json.dumps(item) for item in live)
        assert Counter(item['message'] for item in remaining) == expected
        assert [item['position'] for item in remaining] == sorted(item['position'] for item in remaining)
        assert (history, live) == before
