"""ONNX Runtime telemetry: the off-switches, and an honest report of whether
they took in this process.

ONNX Runtime has two telemetry paths, each with its own switch:

* Linux and macOS builds from 1.29.0 on (first released 2026-08-17) embed
  Microsoft's 1DS telemetry SDK. Importing onnxruntime is enough, no session
  is needed: about 9 s after the import it posts events to
  https://mobile.events.data.microsoft.com/OneCollector/1.0, then again a few
  seconds later. They carry a persistent device id, OS name and version, CPU
  model, total memory, network type, container/VM detection and the full
  Python executable path (which holds the user name). Events it has not sent yet are queued in
  ~/.cache/Microsoft/DeveloperTools/.onnxruntime/ and sent by the next ORT
  process that runs long enough. The one switch measured to stop it is the
  environment variable ORT_DISABLE_TELEMETRY, which ORT reads when onnxruntime
  is FIRST imported; set after that, it is too late. revien/__init__.py sets
  it before anything Revien loads can import onnxruntime.
  disable_telemetry_events() does NOT stop this path (measured).
* Windows builds emit TraceLogging/ETW events through the OS;
  onnxruntime.disable_telemetry_events() turns that emission off. Events
  already emitted, and whether the OS diagnostics service uploads them, are
  outside Revien's control.

ort_telemetry_state() is True only when both switches took: the variable held
a value ORT treats as off when onnxruntime was first imported, and
disable_telemetry_events() ran. When a host application imported onnxruntime
before Revien without the variable, or the user set it to a value ORT does not
treat as off, this process cannot stop it: the state is False and Revien logs
one warning that names the fix.
"""

import logging
import os
from typing import Optional, Tuple

import revien as _pkg

log = logging.getLogger("revien.semantic")

ENV_VAR = "ORT_DISABLE_TELEMETRY"
# The values onnxruntime 1.31.0 treats as "off", compared after trimming
# ASCII whitespace and ignoring ASCII case. Measured on Linux: each gave 0
# connections and no ~/.cache/Microsoft directory. Measured as NOT off (1
# connection each): unset, "", "0", "false", "no", "off", "t", "2", "01",
# "1 1". A value outside this set is reported as not off, whatever a later
# ORT might accept.
OFF_VALUES = frozenset({"1", "true", "yes", "y", "on"})
# What ORT trims from either end before comparing: exactly ASCII space, \t,
# \n, \r, \v and \f (each measured, leading and trailing, as off with "1").
# NOT trimmed (measured: telemetry stays on): a no-break space (U+00A0) and
# \x1c, both of which Python's str.strip() would remove, and \x01.
ORT_WHITESPACE = " \t\n\r\v\f"
FIX = (f"set {ENV_VAR}=1 before importing onnxruntime (in the environment, "
       "or in os.environ before the first `import onnxruntime`)")

_DONE = False        # disable_telemetry_events() ran in this process
_UNSEEN = object()
_SEEN_ENV = _UNSEEN  # the variable when Revien first imported onnxruntime
_WARNED = False


def env_value_is_off(value: Optional[str]) -> bool:
    """True when ONNX Runtime treats this ORT_DISABLE_TELEMETRY value as off."""
    if value is None:
        return False
    v = value.strip(ORT_WHITESPACE)
    return v.isascii() and v.lower() in OFF_VALUES


def _disable() -> Optional[bool]:
    """True = ran, False = onnxruntime present but no switch, None = absent."""
    global _DONE, _SEEN_ENV
    if _DONE:
        return True
    if _SEEN_ENV is _UNSEEN:
        # Read before the import below: when Revien is the first importer of
        # onnxruntime, this is exactly the value ORT is about to read.
        _SEEN_ENV = os.environ.get(ENV_VAR)
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


def _shown(value: Optional[str]) -> str:
    return "unset" if value is None else f"{ENV_VAR}={value[:40]!r}"


def _env_problem() -> Optional[str]:
    """Why the variable did not take, from what was known when onnxruntime was
    first imported; None when it took."""
    if getattr(_pkg, "_ORT_PREIMPORTED", False):
        # onnxruntime was loaded before revien/__init__ ran: what counts is the
        # value the environment held then, before Revien touched it.
        value = getattr(_pkg, "_ORT_TELEMETRY_PRESET", None)
        if env_value_is_off(value):
            return None
        if not (value or "").strip(ORT_WHITESPACE):
            return (f"onnxruntime was imported before revien without {ENV_VAR} "
                    "(ONNX Runtime reads it only at its first import, so "
                    "Revien setting it afterwards is too late)")
        return (f"onnxruntime was imported before revien with {_shown(value)}, "
                "which ONNX Runtime does not treat as off")
    value = _SEEN_ENV if _SEEN_ENV is not _UNSEEN else os.environ.get(ENV_VAR)
    if env_value_is_off(value):
        return None
    if value is None or not value.strip(ORT_WHITESPACE):
        return (f"{ENV_VAR} was {'unset' if value is None else 'empty'} when "
                "onnxruntime was imported")
    return f"{_shown(value)} is set, which ONNX Runtime does not treat as off"


def _problem(ran: bool) -> Optional[Tuple[str, str]]:
    """(why telemetry is not off, the fix), or None when it is off."""
    env = _env_problem()
    if env is not None:
        return env, FIX
    if not ran:
        return ("onnxruntime.disable_telemetry_events() is missing or failed "
                "(the Windows ETW path)", "upgrade onnxruntime")
    return None


def _warn_once(why: str, fix: str) -> None:
    global _WARNED
    if _WARNED:
        return
    _WARNED = True
    log.warning("ONNX Runtime telemetry is NOT off in this process: %s. "
                "Fix: %s.", why, fix)


def disable_ort_telemetry() -> bool:
    """Idempotent, never raises. True when ONNX Runtime telemetry is off in
    this process (see ort_telemetry_state)."""
    return bool(ort_telemetry_state())


def ort_telemetry_state() -> Optional[bool]:
    """True when ONNX Runtime telemetry is off in this process, False when it
    is not (logs one warning naming the fix), None when onnxruntime is not
    installed."""
    ran = _disable()
    if ran is None:
        return None
    problem = _problem(ran)
    if problem is not None:
        _warn_once(*problem)
        return False
    return True


def ort_telemetry_problem() -> Optional[str]:
    """One line for reports: why telemetry is not off, and the fix; None when
    it is off or onnxruntime is not installed. Never raises."""
    ran = _disable()
    if ran is None:
        return None
    problem = _problem(ran)
    return None if problem is None else f"{problem[0]}. Fix: {problem[1]}"
