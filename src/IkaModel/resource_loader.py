"""Typed access to optional resources supplied by an embedding application."""

# pyright: strict

from collections.abc import Callable
from importlib import import_module
from typing import cast


def read_embedded_resource(path: str) -> str:
    module = import_module("src.resources")
    try:
        reader = cast(Callable[[str], str], getattr(module, "read_text"))
    except AttributeError as error:
        raise ImportError("src.resources does not provide read_text") from error
    return reader(path)
