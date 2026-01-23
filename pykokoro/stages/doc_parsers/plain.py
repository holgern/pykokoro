from __future__ import annotations

from typing import TYPE_CHECKING

from ..protocols import DocumentResult

if TYPE_CHECKING:
    from ...pipeline_config import PipelineConfig
    from ...types import Trace


class PlainTextDocumentParser:
    def parse(self, text: str, cfg: PipelineConfig, trace: Trace) -> DocumentResult:
        _ = (cfg, trace)
        return DocumentResult(clean_text=text)
