from __future__ import annotations

from typing import Protocol

from ...types import Segment


class Splitter(Protocol):
    def split(self, text: str, *, max_chars: int | None) -> list[Segment]:
        """Split text into stable segments with original char offsets."""
        ...
