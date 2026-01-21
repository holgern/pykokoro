from __future__ import annotations

from ...types import Segment


class NoSplitSplitter:
    def split(self, text: str, *, max_chars: int | None) -> list[Segment]:
        return [
            Segment(
                id="p0s0",
                text=text,
                char_start=0,
                char_end=len(text),
                paragraph_idx=0,
                sentence_idx=0,
            )
        ]
