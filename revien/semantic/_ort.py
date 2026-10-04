"""ONNX Runtime telemetry off-switch.

ORT's Windows builds emit telemetry events through the OS (TraceLogging/ETW).
Revien's default path promises zero egress, so we turn emission off before any
model/session is created. Events already emitted, and whether the OS
diagnostics service uploads them, are outside Revien's control.
"""

from typing import Optional

_DONE = False


def _disable() -> Optional[bool]:
    """True = ran, False = onnxruntime present but no switch, None = absent."""
    global _DONE
    if _DONE:
        return True
    try:
        import onnxruntime
    except Exception:
        return None
    try:
        fn = getattr(onnxruntime, "disable_telemetry_events", None)
        if fn is None:
            return False
        fn()
    except Exception:
        return False
    _DONE = True
    return True


def disable_ort_telemetry() -> bool:
    """Idempotent, never raises. True when telemetry emission was disabled."""
    return bool(_disable())


def ort_telemetry_state() -> Optional[bool]:
    """True/False for reports; None when onnxruntime is not installed."""
    return _disable()
