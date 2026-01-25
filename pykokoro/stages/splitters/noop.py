from __future__ import annotations

from typing import TYPE_CHECKING

from ...types import Segment, Trace
from ..protocols import DocumentResult

if TYPE_CHECKING:
    from ...pipeline_config import PipelineConfig

__all__ = ["NoopSplitter"]


class NoopSplitter:
    def split(
        self, doc: DocumentResult, cfg: PipelineConfig, trace: Trace
    ) -> list[Segment]:
        _ = (cfg, trace)
        return [
            Segment(
                id="p0_s0_c0_seg0",
                text=doc.clean_text,
                char_start=0,
                char_end=len(doc.clean_text),
                paragraph_idx=0,
                sentence_idx=0,
                clause_idx=0,
            )
        ]
