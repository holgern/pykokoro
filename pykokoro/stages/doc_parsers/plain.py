from __future__ import annotations

import re
from typing import TYPE_CHECKING

from ...types import BoundaryEvent
from ..protocols import DocumentResult

if TYPE_CHECKING:
    from ...pipeline_config import PipelineConfig
    from ...types import Trace


class PlainTextDocumentParser:
    def parse(self, text: str, cfg: PipelineConfig, trace: Trace) -> DocumentResult:
        _ = (cfg, trace)
        boundaries: list[BoundaryEvent] = []
        for match in re.finditer(r"\n\s*\n", text):
            if match.start() == 0:
                continue
            boundary_pos = match.start() - 1
            if boundary_pos < 0:
                continue
            boundaries.append(
                BoundaryEvent(
                    pos=boundary_pos,
                    kind="pause",
                    duration_s=None,
                    attrs={"strength": "p"},
                )
            )
        return DocumentResult(clean_text=text, boundary_events=boundaries)
