from __future__ import annotations

from typing import Protocol

from ...types import Annotation, Segment
from .kokorog2p import PhonemeSegment


class G2P(Protocol):
    def phonemize(
        self,
        *,
        segments: list[Segment],
        clean_texts: list[str],
        annotations: list[list[Annotation]],
    ) -> list[PhonemeSegment]: ...
