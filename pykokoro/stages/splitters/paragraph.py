from __future__ import annotations

import re
from typing import TYPE_CHECKING

from ...types import Segment, Trace
from ..protocols import DocumentResult

if TYPE_CHECKING:
    from ...pipeline_config import PipelineConfig


class ParagraphSplitter:
    def split(
        self, doc: DocumentResult, cfg: PipelineConfig, trace: Trace
    ) -> list[Segment]:
        _ = (cfg, trace)
        text = doc.clean_text
        if not text:
            return []

        segments: list[Segment] = []
        para_idx = 0
        pattern = re.compile(r"(?:^|\n\s*\n)(?P<para>.*?)(?=\n\s*\n|\Z)", re.DOTALL)
        for match in pattern.finditer(text):
            para_text = match.group("para")
            if not para_text.strip():
                continue
            start = match.start("para")
            end = match.end("para")
            segments.append(
                Segment(
                    id=f"p{para_idx}_s0_c0_seg0",
                    text=para_text,
                    char_start=start,
                    char_end=end,
                    paragraph_idx=para_idx,
                    sentence_idx=0,
                    clause_idx=0,
                )
            )
            para_idx += 1

        if not segments:
            return [
                Segment(
                    id="p0_s0_c0_seg0",
                    text=text,
                    char_start=0,
                    char_end=len(text),
                    paragraph_idx=0,
                    sentence_idx=0,
                    clause_idx=0,
                )
            ]

        return segments
