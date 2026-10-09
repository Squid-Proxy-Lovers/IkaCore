"""Compatibility import for the older Gemini payload-builder module path."""

from .gemini.google import gemini_fill_payload as gemini_fill_payload

__all__ = ["gemini_fill_payload"]
