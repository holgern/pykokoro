from __future__ import annotations

from ...pipeline_config import PipelineConfig
from ...types import Segment, Trace
from ..base import DocumentResult, Splitter


class CompatSplitter(Splitter):
    def split(
        self, doc: DocumentResult, cfg: PipelineConfig, trace: Trace
    ) -> list[Segment]:
        text = doc.clean_text
        return [
            Segment(
                id="compat_0",
                text=text,
                char_start=0,
                char_end=len(text),
                paragraph_idx=0,
                sentence_idx=0,
            )
        ]
