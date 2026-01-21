from __future__ import annotations

from typing import Any

import numpy as np

from ...config import PipelineConfig
from ...types import Annotation, PhonemeSegment, Segment


class KokoroG2PAdapter:
    """Adapter around kokorog2p.

    Goals:
    - accept clean_text + span annotations (already parsed)
    - apply overrides by spans (NOT by word-string matching)
    - output token_ids when possible
    """

    def __init__(self, config: PipelineConfig) -> None:
        self.config = config
        self._g2p = None

    def _load(self):
        if self._g2p is not None:
            return self._g2p
        try:
            import kokorog2p  # type: ignore
        except Exception as e:
            raise RuntimeError("kokorog2p is not installed") from e

        # Placeholder: adapt to kokorog2p public API
        self._g2p = kokorog2p
        return self._g2p

    def phonemize(
        self,
        *,
        segments: list[Segment],
        clean_texts: list[str],
        annotations: list[list[Annotation]],
    ) -> list[PhonemeSegment]:
        g2p = self._load()

        out: list[PhonemeSegment] = []
        for seg, clean, ann in zip(segments, clean_texts, annotations):
            # TODO: implement span-based overrides.
            # Temporary: just phonemize whole string.
            phonemes = g2p.phonemize(clean, lang=self.config.lang) if hasattr(g2p, "phonemize") else clean

            token_ids = None
            if hasattr(g2p, "phonemes_to_ids"):
                try:
                    token_ids = np.asarray(g2p.phonemes_to_ids(phonemes), dtype=np.int64)
                except Exception:
                    token_ids = None

            out.append(
                PhonemeSegment(
                    segment_id=seg.id,
                    phonemes=None if token_ids is not None else str(phonemes),
                    token_ids=token_ids,
                    meta={"lang": seg.meta.get("lang", self.config.lang)},
                )
            )
        return out
