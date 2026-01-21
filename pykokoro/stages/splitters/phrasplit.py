from __future__ import annotations

from .base import Splitter
from ...types import Segment


class PhrasplitSplitter:
    """Adapter over phrasplit.

    NOTE: This is a stub. Wire it to phrasplit.split_text / Segment outputs and
    ensure char offsets refer to the original input string.
    """

    def split(self, text: str, *, max_chars: int | None) -> list[Segment]:
        # Lazy import so optional dependency
        try:
            from phrasplit import split_text  # type: ignore
        except Exception:
            # Fallback: single segment
            return [Segment(id="p0s0", text=text, char_start=0, char_end=len(text), paragraph_idx=0, sentence_idx=0)]

        # phrasplit returns Segment objects without guaranteed offsets today.
        # Recommended: update phrasplit to provide offsets; until then, best-effort.
        parts = split_text(text, max_length=max_chars or 10**9)
        segments: list[Segment] = []
        cursor = 0
        p_idx = 0
        s_idx = 0
        for i, p in enumerate(parts):
            # Best-effort offset search from current cursor
            j = text.find(p, cursor)
            if j < 0:
                j = cursor
            segments.append(Segment(id=f"p{p_idx}s{s_idx}", text=p, char_start=j, char_end=j + len(p), paragraph_idx=p_idx, sentence_idx=s_idx))
            cursor = j + len(p)
            s_idx += 1
        return segments
