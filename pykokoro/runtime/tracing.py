from __future__ import annotations

import time
from contextlib import contextmanager

from ..types import Trace, TraceEvent


@contextmanager
def trace_timing(trace: Trace | None, stage: str, name: str):
    t0 = time.perf_counter()
    try:
        yield
    finally:
        if trace is not None:
            ms = (time.perf_counter() - t0) * 1000.0
            trace.events.append(TraceEvent(stage=stage, name=name, ms=ms))
