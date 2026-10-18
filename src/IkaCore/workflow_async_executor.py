"""Thread-pool executor used by asynchronous workflow runs."""

from __future__ import annotations

import concurrent.futures
import threading
from typing import Any, Callable, Coroutine, Dict, List, Optional, cast

from IkaCore.agents import IkaBaseAgent
from IkaCore.events import current_output as get_cli_output
from IkaCore.runtime_control import RuntimePauseRequested
from IkaModel.async_runner import run_async
from IkaModel.runtime_errors import IkaRequestControlError
from IkaModel.worker_context import submit_with_controls


def _run_async_workflow_agent(
    node_name: str,
    agent: IkaBaseAgent,
    context: str,
    instance_id: int,
) -> Dict[str, Any]:
    try:
        cli = get_cli_output()
        cli.set_instance_id(instance_id)

        if context:
            agent.inject_workflow_context(context)

        async_execution = getattr(agent, "async_execution", None)
        if (getattr(agent, "use_async", False) and callable(async_execution)
                and not getattr(agent, "_workflow_snapshot_instance", False)):
            result = run_async(cast(Callable[[], Coroutine[Any, Any, Any]], async_execution))
        else:
            result = agent.execution()
        return {
            "node_name": node_name,
            "instance_id": instance_id,
            "result": result,
            "success": True,
            "transcript": getattr(agent, "transcript", getattr(agent, "message_history", {})),
        }
    except RuntimePauseRequested as paused:
        return {"node_name": node_name, "instance_id": instance_id, "success": False,
                "interrupted": True, "result": {"status": "paused", "snapshot_id": paused.snapshot_id}}
    except IkaRequestControlError:
        raise
    except Exception as e:
        return {
            "node_name": node_name,
            "instance_id": instance_id,
            "result": {"error": str(e), "error_type": type(e).__name__},
            "success": False,
        }

    finally:
        shutdown = getattr(agent, "shutdown", None)
        if callable(shutdown):
            shutdown()


class AsyncWorkflowExecutor:
    def __init__(self, max_workers: int = 30):
        self.max_workers = max_workers
        self.executor = concurrent.futures.ThreadPoolExecutor(max_workers=max_workers)
        self.pending_tasks: Dict[str, concurrent.futures.Future] = {}
        self.completed_results: Dict[str, Any] = {}
        self.lock = threading.Lock()

    def schedule_node(
        self,
        node_name: str,
        agent: IkaBaseAgent,
        context: str = "",
        instance_id: int = 0,
    ) -> concurrent.futures.Future:
        task_id = f"{node_name}_{instance_id}"
        future = submit_with_controls(self.executor, _run_async_workflow_agent, node_name, agent, context, instance_id)

        with self.lock:
            self.pending_tasks[task_id] = future

        return future

    def wait_for_completion(
        self,
        futures: List[concurrent.futures.Future],
        timeout: Optional[float] = None,
    ) -> List[Any]:
        results = []
        for future in concurrent.futures.as_completed(futures, timeout=timeout):
            try:
                result = future.result()
                results.append(result)
                with self.lock:
                    task_id = f"{result['node_name']}_{result['instance_id']}"
                    self.completed_results[task_id] = result
                    if task_id in self.pending_tasks:
                        del self.pending_tasks[task_id]
            except IkaRequestControlError:
                raise
            except Exception as e:
                results.append({
                    "result": {"error": str(e)},
                    "success": False,
                })
        return results

    def drain(self) -> None:
        with self.lock:
            pending = list(self.pending_tasks.values())

        if pending:
            self.wait_for_completion(pending)

    def shutdown(self, wait: bool = True) -> None:
        self.executor.shutdown(wait=wait)

