"""Opt-in runtime snapshots alongside the unchanged legacy checkpoint API."""

# pyright: strict

from .checkpoint import CheckpointStore
from .snapshot_data import (
    create_snapshot,
    get_frame_entry_snapshot,
    get_latest_snapshot_for_frame,
    get_snapshot_chain,
    list_snapshots,
    load_snapshot,
)
from .snapshot_events import add_artifact, assess_frame_replay, list_events, list_frame_events, record_event
from .snapshot_frames import (
    create_frame,
    get_call_tree,
    get_frame,
    get_frame_lineage,
    list_frames,
    list_workflow_node_instances,
    update_frame,
)
from .snapshot_replay import fork_run_from_snapshot, restart_frame, resume_from_snapshot
from .snapshot_runs import create_run, get_run, set_run_root_frame, update_run_status
from .snapshot_schema import connect, decode_frame_row, init_schema


class SnapshotStore(CheckpointStore):
    def __init__(self, db_path: str = "runtime-data/snapshots.db") -> None:
        super().__init__(db_path)
        init_schema(self)

    _connect = connect
    _decode_frame_row = staticmethod(decode_frame_row)
    create_run = create_run
    set_run_root_frame = set_run_root_frame
    update_run_status = update_run_status
    get_run = get_run
    create_frame = create_frame
    update_frame = update_frame
    get_frame = get_frame
    list_frames = list_frames
    get_frame_lineage = get_frame_lineage
    get_call_tree = get_call_tree
    list_workflow_node_instances = list_workflow_node_instances
    create_snapshot = create_snapshot
    load_snapshot = load_snapshot
    get_latest_snapshot_for_frame = get_latest_snapshot_for_frame
    get_frame_entry_snapshot = get_frame_entry_snapshot
    list_snapshots = list_snapshots
    get_snapshot_chain = get_snapshot_chain
    record_event = record_event
    list_events = list_events
    list_frame_events = list_frame_events
    assess_frame_replay = assess_frame_replay
    add_artifact = add_artifact
    resume_from_snapshot = resume_from_snapshot
    fork_run_from_snapshot = fork_run_from_snapshot
    restart_frame = restart_frame
