"""
Revien — Memory that returns.
Graph-based memory engine for AI systems.
"""

# ONNX Runtime 1.29.0 and later (first released 2026-08-17) on Linux and macOS
# posts telemetry to Microsoft, from the import alone, unless
# ORT_DISABLE_TELEMETRY is set when onnxruntime is FIRST imported (it reads the
# variable then; disable_telemetry_events() afterwards does not stop it). This
# runs before any other Revien import, so the variable is in place before
# anything Revien loads can import onnxruntime, and every child process
# inherits it through os.environ. A value the user already set is kept; an
# empty one (or only the ASCII whitespace ORT trims) counts as unset. Whether
# the switch was in time for this process is reported by
# revien.semantic._ort.ort_telemetry_state().
import os as _os
import sys as _sys

_ORT_TELEMETRY_ENV = "ORT_DISABLE_TELEMETRY"
# What the environment held before Revien touched it (None = unset).
_ORT_TELEMETRY_PRESET = _os.environ.get(_ORT_TELEMETRY_ENV)
# onnxruntime already imported (by the host app) before Revien could set it.
_ORT_PREIMPORTED = "onnxruntime" in _sys.modules
if not (_ORT_TELEMETRY_PRESET or "").strip(" \t\n\r\v\f"):
    _os.environ[_ORT_TELEMETRY_ENV] = "1"
del _os, _sys

__version__ = "0.3.0"
