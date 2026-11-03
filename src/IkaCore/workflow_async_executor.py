"""Thread-pool executor used by asynchronous workflow runs."""

from __future__ import annotations

import asyncio
import concurrent.futures
import threading
from typing import Any, Callable, Coroutine, Dict, List, Optional, cast

from IkaCore.agents import IkaBaseAgent
from IkaCore.cli_output import get_cli_output


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
        if getattr(agent, "use_async", False) and callable(async_execution):
            result = asyncio.run(cast(Callable[[], Coroutine[Any, Any, Any]], async_execution)())
        else:
            result = agent.execution()
        return {
            "node_name": node_name,
            "instance_id": instance_id,
            "result": result,
            "success": True,
        }
    except Exception as e:
        return {
            "node_name": node_name,
            "instance_id": instance_id,
            "result": {"error": str(e)},
            "success": False,
        }


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
        future = self.executor.submit(_run_async_workflow_agent, node_name, agent, context, instance_id)

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

