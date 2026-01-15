#!/usr/bin/env python3

import json
import os
import sys
import threading
import time
import signal
from pathlib import Path
from typing import Dict, List, Optional

import asyncio
from fastapi import FastAPI, UploadFile, File, Form, HTTPException, WebSocket, WebSocketDisconnect, Body
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse, HTMLResponse
from fastapi.staticfiles import StaticFiles


ROOT_DIR = Path(__file__).resolve().parent.parent
SQA_DIR = ROOT_DIR / "Squid_Agent"
TMUX_STATUS_DIR = ROOT_DIR / "tmux_status"
LOGS_DIR = ROOT_DIR / "logs_squidagent"
DEFAULT_CUSTOM_DATASET = ROOT_DIR / "Challenges" / "frontend" / "challenges.json"

# Ensure import path for challenge runner
if str(SQA_DIR) not in sys.path:
    sys.path.insert(0, str(SQA_DIR))

from challenge_runner import run_single_challenge  # type: ignore


app = FastAPI(title="SquidCTF Web UI", version="1.0.0")

# Track running challenge processes
running_challenges: Dict[str, threading.Thread] = {}
running_challenges_lock = threading.Lock()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# Static files (frontend)
STATIC_DIR = Path(__file__).parent / "static"
app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")


def _find_datasets() -> List[Path]:
    roots = [ROOT_DIR / "Challenges", ROOT_DIR / "Challenges" / "ctf", ROOT_DIR / "Challenges" / "www"]
    found: List[Path] = []
    for r in roots:
        if not r.exists():
            continue
        for p in r.rglob("challenges.json"):
            found.append(p)
    # Ensure default custom dataset exists
    _ensure_custom_dataset(DEFAULT_CUSTOM_DATASET)
    if DEFAULT_CUSTOM_DATASET not in found:
        found.append(DEFAULT_CUSTOM_DATASET)
    return sorted(found)


def _ensure_custom_dataset(dataset_path: Path) -> None:
    dataset_path.parent.mkdir(parents=True, exist_ok=True)
    if not dataset_path.exists():
        with open(dataset_path, "w", encoding="utf-8") as f:
            json.dump({}, f, indent=2)


def _read_json(path: Path) -> Dict:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _write_json(path: Path, data: Dict) -> None:
    tmp = str(path) + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)
    os.replace(tmp, path)


def _list_challenges(dataset_path: Path) -> Dict[str, Dict]:
    if not dataset_path.exists():
        return {}
    data = _read_json(dataset_path)
    if not isinstance(data, dict):
        return {}
    return data


def _start_run_in_thread(challenge_id: str, dataset: str, base_path: str, flag_format: Optional[str]) -> None:
    def _runner():
        try:
            TMUX_STATUS_DIR.mkdir(exist_ok=True)
            run_single_challenge(challenge_id, dataset, base_path, flag_format, str(TMUX_STATUS_DIR))
        except Exception:
            pass
        finally:
            # Remove from tracking when done
            with running_challenges_lock:
                running_challenges.pop(challenge_id, None)
    t = threading.Thread(target=_runner, daemon=True)
    with running_challenges_lock:
        running_challenges[challenge_id] = t
    t.start()


@app.get("/", response_class=HTMLResponse)
def index() -> HTMLResponse:
    return HTMLResponse((STATIC_DIR / "index.html").read_text(encoding="utf-8"))


@app.get("/api/datasets")
def api_datasets():
    datasets = [str(p) for p in _find_datasets()]
    return {"datasets": datasets, "default": str(DEFAULT_CUSTOM_DATASET)}


@app.get("/api/challenges")
def api_list_challenges(dataset: Optional[str] = None):
    paths = _find_datasets() if not dataset else [Path(dataset)]
    out: Dict[str, Dict] = {}
    for p in paths:
        try:
            out[str(p)] = _list_challenges(p)
        except Exception:
            out[str(p)] = {}
    return out


@app.post("/api/challenges")
async def api_add_challenge(
    challenge_id: str = Form(...),
    category: str = Form(...),
    dataset: Optional[str] = Form(None),
    base_path: Optional[str] = Form(None),
    challenge_dir: Optional[str] = Form(None),
    description: Optional[str] = Form(None),
    archive: Optional[UploadFile] = File(None),
    files: Optional[List[UploadFile]] = File(None),
):
    dataset_path = Path(dataset) if dataset else DEFAULT_CUSTOM_DATASET
    _ensure_custom_dataset(dataset_path)

    # Determine where challenge files reside
    if archive is not None:
        target_dir = (dataset_path.parent / challenge_id)
        target_dir.mkdir(parents=True, exist_ok=True)
        content = await archive.read()
        tmp_zip = target_dir / "upload.zip"
        with open(tmp_zip, "wb") as f:
            f.write(content)
        import zipfile
        with zipfile.ZipFile(tmp_zip, 'r') as zf:
            zf.extractall(target_dir)
        tmp_zip.unlink(missing_ok=True)
        ch_dir = target_dir
    elif files:
        target_dir = (dataset_path.parent / challenge_id)
        target_dir.mkdir(parents=True, exist_ok=True)
        for uf in files:
            try:
                # Sanitize filename
                filename = os.path.basename(uf.filename or "")
                if not filename:
                    continue
                dest = target_dir / filename
                content = await uf.read()
                with open(dest, "wb") as f:
                    f.write(content)
            except Exception:
                pass
        ch_dir = target_dir
    elif challenge_dir is not None:
        ch_dir = Path(challenge_dir)
        if not ch_dir.exists():
            raise HTTPException(status_code=400, detail="challenge_dir does not exist")
    else:
        # Create minimal challenge folder with placeholder challenge.json
        ch_dir = (dataset_path.parent / challenge_id)
        ch_dir.mkdir(parents=True, exist_ok=True)
        cj = ch_dir / "challenge.json"
        if not cj.exists():
            with open(cj, "w", encoding="utf-8") as f:
                json.dump({
                    "name": challenge_id,
                    "category": category,
                    "description": description or f"Added via web UI",
                    "files": [],
                }, f, indent=2)

    # Ensure challenge.json exists and contains an up-to-date files list and description
    try:
        cj = ch_dir / "challenge.json"
        files_list: List[str] = []
        for root, _, filenames in os.walk(ch_dir):
            for fn in filenames:
                if fn == "challenge.json":
                    continue
                full = Path(root) / fn
                rel = os.path.relpath(str(full), str(ch_dir))
                files_list.append(rel)

        meta: Dict[str, any] = {}
        if cj.exists():
            try:
                existing = _read_json(cj)
                if isinstance(existing, dict):
                    meta = existing
            except Exception:
                meta = {}
        # Populate defaults and updates
        meta.setdefault("name", challenge_id)
        meta.setdefault("category", category)
        if description:
            meta["description"] = description
        meta.setdefault("description", f"Added via web UI")
        meta["files"] = sorted(files_list)

        with open(cj, "w", encoding="utf-8") as f:
            json.dump(meta, f, indent=2)
    except Exception:
        pass

    data = _list_challenges(dataset_path)
    rel_path = os.path.relpath(str(ch_dir), str(dataset_path.parent))
    data[challenge_id] = {"category": category, "path": rel_path}
    _write_json(dataset_path, data)

    return {"ok": True, "dataset": str(dataset_path), "challenge_id": challenge_id, "dir": str(ch_dir)}


@app.post("/api/run")
def api_run(
    challenge_id: str = Body(...),
    dataset: str = Body(...),
    base_path: Optional[str] = Body(None),
    flag_format: Optional[str] = Body(None),
):
    ds_path = Path(dataset)
    if not ds_path.exists():
        raise HTTPException(status_code=400, detail="dataset not found")
    base = base_path if base_path else str(ds_path.parent)
    challenge_id = challenge_id.replace(" ", "-")
    _start_run_in_thread(challenge_id, str(ds_path), base, flag_format)
    return {"started": True, "challenge_id": challenge_id}


@app.post("/api/stop")
def api_stop(challenge_id: str = Body(..., embed=True)):
    """Stop a running challenge by updating its status to 'stopping'."""
    challenge_id = challenge_id.replace(" ", "-")
    
    # Check if challenge is running
    status_file = TMUX_STATUS_DIR / f"{challenge_id}.json"
    if not status_file.exists():
        raise HTTPException(status_code=404, detail="challenge not found")
    
    try:
        status = _read_json(status_file)
        current_state = status.get('state', '')
        
        if current_state not in ['starting', 'running']:
            raise HTTPException(status_code=400, detail=f"challenge is not running (state: {current_state})")
        
        # Update status to 'stopping' - challenge_runner will detect this
        status['state'] = 'stopping'
        _write_json(status_file, status)
        
        return {"stopped": True, "challenge_id": challenge_id}
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"error stopping challenge: {str(e)}")


@app.get("/api/status")
def api_status():
    TMUX_STATUS_DIR.mkdir(exist_ok=True)
    statuses: List[Dict] = []
    for p in TMUX_STATUS_DIR.glob("*.json"):
        try:
            statuses.append(_read_json(p))
        except Exception:
            pass
    return {"statuses": sorted(statuses, key=lambda s: s.get("start_time", 0))}


@app.get("/api/flags")
def api_flags():
    """Aggregate captured flags from JSON transcripts in logs_squidagent.

    Returns a list of objects: {challenge_id, flag, success, time}
    """
    results: List[Dict] = []
    if LOGS_DIR.exists():
        for p in LOGS_DIR.glob("*.json"):
            try:
                data = _read_json(p)
            except Exception:
                continue

            challenge_id = p.stem
            success = bool(data.get("success", False))
            time_taken = data.get("time_taken")
            end_time = data.get("end_time")
            transcript = data.get("transcript", []) or []

            flag_value: Optional[str] = None
            # Prefer explicit meta.flag_submitted entries, use the last non-empty one
            for ev in transcript:
                meta = ev.get("meta") if isinstance(ev, dict) else None
                if isinstance(meta, dict) and meta.get("flag_submitted"):
                    flag_value = meta.get("flag_submitted")
            # Fallback: look for tool_result with a 'flag' field from validation tools
            if not flag_value:
                for ev in transcript:
                    tr = ev.get("tool_result") if isinstance(ev, dict) else None
                    if isinstance(tr, dict) and isinstance(tr.get("result"), dict):
                        maybe_flag = tr["result"].get("flag")
                        if maybe_flag:
                            flag_value = maybe_flag

            # Only include entries where we found a flag
            if flag_value:
                results.append({
                    "challenge_id": challenge_id,
                    "flag": flag_value,
                    "success": success,
                    "time": end_time,
                    "time_taken": time_taken,
                })

    # Sort most recent first when end_time available
    results.sort(key=lambda x: (x.get("time") or 0), reverse=True)
    return {"flags": results}


@app.get("/api/logs/{challenge_id}")
def api_logs(challenge_id: str):
    # Prefer runner log in tmux_status, else transcript JSON in logs_squidagent
    log_txt = TMUX_STATUS_DIR / f"{challenge_id}.log"
    if log_txt.exists():
        return FileResponse(str(log_txt))

    # Fallback to JSON transcript
    if LOGS_DIR.exists():
        for p in LOGS_DIR.glob("*.json"):
            if challenge_id.lower().replace(" ", "-") in p.stem:
                return FileResponse(str(p))

    raise HTTPException(status_code=404, detail="log not found")


def _log_file_for_challenge(challenge_id: str) -> Optional[Path]:
    p = TMUX_STATUS_DIR / f"{challenge_id}.log"
    if p.exists():
        return p
    return None


@app.get("/api/logs/info/{challenge_id}")
def api_logs_info(challenge_id: str):
    p = _log_file_for_challenge(challenge_id)
    if not p:
        raise HTTPException(status_code=404, detail="log not found")
    try:
        return {"size": p.stat().st_size}
    except Exception:
        raise HTTPException(status_code=500, detail="could not stat log")


@app.get("/api/logs/chunk/{challenge_id}")
def api_logs_chunk(challenge_id: str, offset: int, length: int):
    if offset < 0 or length <= 0 or length > 4 * 1024 * 1024:
        raise HTTPException(status_code=400, detail="invalid range")
    p = _log_file_for_challenge(challenge_id)
    if not p:
        raise HTTPException(status_code=404, detail="log not found")
    try:
        size = p.stat().st_size
        if offset >= size:
            return {"offset": offset, "data": "", "eof": True}
        with open(p, 'r', encoding='utf-8', errors='ignore') as f:
            f.seek(offset, os.SEEK_SET)
            data = f.read(length)
        return {"offset": offset, "data": data, "eof": (offset + len(data) >= size)}
    except Exception:
        raise HTTPException(status_code=500, detail="could not read chunk")


@app.websocket("/ws/logs/{challenge_id}")
async def ws_logs(websocket: WebSocket, challenge_id: str):
    await websocket.accept()
    file_path = _log_file_for_challenge(challenge_id)

    try:
        # Wait for file to exist if the run just started
        wait_start = time.time()
        while file_path is None or not file_path.exists():
            await asyncio.sleep(0.2)
            file_path = _log_file_for_challenge(challenge_id)
            if time.time() - wait_start > 60:
                await websocket.send_text("[log] waiting for log file...\n")
                break

        if file_path and file_path.exists():
            # Start tailing from end (frontend loads history via chunk API)
            with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                f.seek(0, os.SEEK_END)
                while True:
                    line = f.readline()
                    if line:
                        await websocket.send_text(line)
                    else:
                        await asyncio.sleep(0.25)
        else:
            # Fallback: no text log, try JSON transcript bursts
            if LOGS_DIR.exists():
                # Stream minimal updates by re-sending whole file when it changes
                last_sent_size = 0
                target_file = None
                for p in LOGS_DIR.glob("*.json"):
                    if challenge_id.lower().replace(" ", "-") in p.stem:
                        target_file = p
                        break
                while True:
                    if target_file and target_file.exists():
                        try:
                            sz = target_file.stat().st_size
                            if sz != last_sent_size:
                                last_sent_size = sz
                                await websocket.send_text(f"[json_update]{target_file.read_text(encoding='utf-8')}\n")
                        except Exception:
                            pass
                    await asyncio.sleep(1.0)
            else:
                await websocket.send_text("[log] no logs available\n")
                await asyncio.sleep(1.0)
    except WebSocketDisconnect:
        return
    except Exception:
        try:
            await websocket.close()
        except Exception:
            pass

@app.get("/healthz")
def healthz():
    return {"ok": True, "time": time.time()}


def main() -> None:
    import uvicorn
    port = int(os.environ.get("PORT", "8000"))
    uvicorn.run(app, host="0.0.0.0", port=port, reload=False)


if __name__ == "__main__":
    main()


