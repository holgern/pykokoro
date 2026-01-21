from __future__ import annotations

from ...pipeline_config import PipelineConfig
from ...types import Trace
from ..base import DocumentResult


class SsmdCompatDocumentParser:
    def parse(self, text: str, cfg: PipelineConfig, trace: Trace) -> DocumentResult:
        return DocumentResult(clean_text=text)
