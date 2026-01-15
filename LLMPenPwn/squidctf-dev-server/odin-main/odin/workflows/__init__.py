import importlib
import inspect
import logging
import pkgutil
from typing import Dict, Type

from ..runtime.workflow import Workflow

_LOG = logging.getLogger(__name__)
WORKFLOW_REGISTRY: Dict[str, Type[Workflow]] = {}


def _register_workflow(wf_cls: Type[Workflow]) -> None:
    if wf_cls is Workflow or wf_cls.__name__ == "Workflow":
        return

    name = wf_cls.__name__.lower()
    if name.endswith("workflow"):
        name = name[:-8]

    _LOG.debug("Registering workflow '%s' → %s", name, wf_cls.__name__)
    WORKFLOW_REGISTRY[name] = wf_cls


def _discover_workflows() -> None:
    for module_info in pkgutil.iter_modules(__path__, __name__ + "."):
        try:
            module = importlib.import_module(module_info.name)
            for _, obj in inspect.getmembers(module, inspect.isclass):
                if issubclass(obj, Workflow) and obj.__module__ == module_info.name:
                    _register_workflow(obj)
        except Exception as e:
            _LOG.error("Error loading workflow module %s: %s", module_info.name, e)


_discover_workflows()


