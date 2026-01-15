import heapq
import threading
import time
import traceback
import uuid
from dataclasses import asdict, dataclass, field
from typing import Any, Callable, Dict, List, Optional, Set

JobHandler = Callable[["JobContext"], None]


@dataclass(order=True)
class _QItem:
    priority: int
    created_at: float
    job_id: str = field(compare=False)


@dataclass
class Job:
    id: str
    job_type: str
    payload: Dict[str, Any] = field(default_factory=dict)
    priority: int = 100                # lower number == higher priority
    depends_on: Set[str] = field(default_factory=set)
    status: str = "queued"             # queued|waiting|running|completed|failed
    error: Optional[str] = None
    trace: Optional[Dict[str, Any]] = None
    output: Optional[Any] = None
    created_at: float = field(default_factory=time.time)
    started_at: Optional[float] = None
    finished_at: Optional[float] = None
    unique_key: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        data = asdict(self)
        data["depends_on"] = sorted(self.depends_on)
        return data


class JobContext:
    def __init__(self, queue: "JobQueue", job: Job):
        self._queue = queue
        self.job = job

    @property
    def id(self) -> str:
        return self.job.id

    @property
    def job_type(self) -> str:
        return self.job.job_type

    @property
    def payload(self) -> Dict[str, Any]:
        return self.job.payload

    def update_trace(self, trace: Dict[str, Any]) -> None:
        self._queue.update_trace(self.id, trace)

    @property
    def output(self) -> Optional[Any]:
        return self.job.output

    def set_output(self, output: Any) -> None:
        self._queue.update_output(self.id, output)

    def get_output(self, job_id: str) -> Optional[Any]:
        return self._queue.get_output(job_id)

    def enqueue_child(
        self,
        job_type: str,
        payload: Optional[Dict[str, Any]] = None,
        *,
        priority: Optional[int] = None,
        depends_on: Optional[List[str]] = None,
    ) -> str:
        deps = depends_on if depends_on is not None else [self.id]
        prio = priority if priority is not None else self.job.priority
        return self._queue.enqueue(
            job_type=job_type,
            payload=payload or {},
            priority=prio,
            depends_on=deps,
        )


class JobQueue:
    def __init__(self, workers: int = 2):
        self._lock = threading.RLock()
        self._ready: List[_QItem] = []
        self._jobs: Dict[str, Job] = {}
        self._waiting: Dict[str, Job] = {}
        self._done: Set[str] = set()
        self._handlers: Dict[str, JobHandler] = {}
        self._workers = workers
        self._threads: List[threading.Thread] = []
        self._cv = threading.Condition(self._lock)
        self._running = False
        self._unique_jobs: Dict[str, str] = {}

    def register_handler(self, job_type: str, handler: JobHandler) -> None:
        with self._lock:
            self._handlers[job_type] = handler

    def job_types(self) -> List[str]:
        with self._lock:
            return sorted(self._handlers)

    def start(self) -> None:
        with self._lock:
            if self._running:
                return
            self._running = True
            for i in range(self._workers):
                thread = threading.Thread(target=self._worker_loop, name=f"worker-{i}", daemon=True)
                thread.start()
                self._threads.append(thread)

    def stop(self) -> None:
        with self._lock:
            self._running = False
            self._cv.notify_all()
        for thread in self._threads:
            thread.join(timeout=2)

    def enqueue(
        self,
        *,
        job_type: str,
        payload: Optional[Dict[str, Any]] = None,
        priority: int = 100,
        depends_on: Optional[List[str]] = None,
        unique_key: Optional[str] = None,
    ) -> str:
        payload = payload or {}
        depends = set(depends_on or [])

        with self._lock:
            if job_type not in self._handlers:
                raise ValueError(f"No handler registered for job type '{job_type}'")
            if unique_key:
                existing_id = self._unique_jobs.get(unique_key)
                if existing_id:
                    existing_job = self._jobs.get(existing_id)
                    if existing_job and existing_job.status in {"queued", "waiting", "running", "completed"}:
                        return existing_job.id
                    if existing_job and existing_job.status == "failed":
                        self._unique_jobs.pop(unique_key, None)

            job_id = str(uuid.uuid4())
            job = Job(
                id=job_id,
                job_type=job_type,
                payload=payload,
                priority=priority,
                depends_on=depends,
                unique_key=unique_key,
            )
            self._jobs[job_id] = job
            if unique_key:
                self._unique_jobs[unique_key] = job_id
            if self._deps_satisfied(job.depends_on):
                heapq.heappush(self._ready, _QItem(priority=priority, created_at=job.created_at, job_id=job_id))
                job.status = "queued"
                self._cv.notify()
            else:
                job.status = "waiting"
                self._waiting[job_id] = job
        return job_id

    def list_jobs(self) -> Dict[str, Dict[str, Any]]:
        with self._lock:
            # do not include trace in the listing
            return {job_id: {**job.to_dict(), "trace": None} for job_id, job in self._jobs.items()}

    def get_job(self, job_id: str) -> Optional[Dict[str, Any]]:
        with self._lock:
            job = self._jobs.get(job_id)
            return job.to_dict() if job else None

    def get_trace(self, job_id: str) -> Optional[Dict[str, Any]]:
        with self._lock:
            job = self._jobs.get(job_id)
            return job.trace if job else None

    def update_trace(self, job_id: str, trace: Dict[str, Any]) -> None:
        with self._lock:
            job = self._jobs.get(job_id)
            if job:
                job.trace = trace

    def update_output(self, job_id: str, output: Any) -> None:
        with self._lock:
            job = self._jobs.get(job_id)
            if job:
                job.output = output

    def get_output(self, job_id: str) -> Optional[Any]:
        with self._lock:
            job = self._jobs.get(job_id)
            return job.output if job else None

    def _deps_satisfied(self, deps: Set[str]) -> bool:
        return all(dep in self._done for dep in deps)

    def _maybe_unblock_waiting(self) -> None:
        ready_jobs = [job for job in self._waiting.values() if self._deps_satisfied(job.depends_on)]
        for job in ready_jobs:
            job.status = "queued"
            heapq.heappush(self._ready, _QItem(priority=job.priority, created_at=job.created_at, job_id=job.id))
            del self._waiting[job.id]
        if ready_jobs:
            self._cv.notify_all()

    def _worker_loop(self) -> None:
        while True:
            with self._lock:
                while self._running and not self._ready:
                    self._cv.wait(timeout=0.5)
                if not self._running and not self._ready:
                    return
                queue_item = heapq.heappop(self._ready)
                job = self._jobs[queue_item.job_id]
                handler = self._handlers.get(job.job_type)
                if handler is None:
                    job.status = "failed"
                    job.error = f"No handler for type {job.job_type}"
                    self._done.add(job.id)
                    continue
                job.status = "running"
                job.started_at = time.time()

            context = JobContext(self, job)

            try:
                handler(context)
                with self._lock:
                    job.status = "completed"
                    job.finished_at = time.time()
                    self._done.add(job.id)
            except Exception as exc:
                with self._lock:
                    job.status = "failed"
                    job.error = str(exc) + "\n" + "".join(traceback.format_exception(type(exc), exc, exc.__traceback__))
                    job.finished_at = time.time()
                    self._done.add(job.id)
                    if job.unique_key:
                        self._unique_jobs.pop(job.unique_key, None)
            finally:
                with self._lock:
                    self._maybe_unblock_waiting()
