from __future__ import annotations

from typing import Any

import numpy as np

from ...phonemes import PhonemeSegment, text_to_phoneme_segments
from ...pipeline_config import PipelineConfig
from ...tokenizer import Tokenizer
from ...types import Trace, TraceEvent
from ..base import DocumentResult, G2PAdapter


class TokenizerCompatG2PAdapter(G2PAdapter):
    def __init__(self, tokenizer: Tokenizer | None = None) -> None:
        self._tokenizer = tokenizer

    def phonemize(
        self,
        segments: list[Any],
        doc: DocumentResult,
        cfg: PipelineConfig,
        trace: Trace,
    ) -> list[PhonemeSegment]:
        tokenizer = self._tokenizer or Tokenizer(
            config=cfg.tokenizer_config, espeak_config=cfg.espeak_config
        )
        generation = cfg.generation

        if generation.is_phonemes:
            phonemes = doc.clean_text
            tokens = tokenizer.tokenize(phonemes)
            return [
                PhonemeSegment(
                    text=phonemes,
                    phonemes=phonemes,
                    tokens=tokens,
                    lang=generation.lang,
                )
            ]

        rng = np.random.default_rng(generation.random_seed)
        phoneme_segments = text_to_phoneme_segments(
            text=doc.clean_text,
            tokenizer=tokenizer,
            lang=generation.lang,
            pause_mode=generation.pause_mode,
            pause_clause=generation.pause_clause,
            pause_sentence=generation.pause_sentence,
            pause_paragraph=generation.pause_paragraph,
            pause_variance=generation.pause_variance,
            rng=rng,
            short_sentence_config=cfg.short_sentence_config,
        )

        boundary_count = sum(
            1 for seg in phoneme_segments if seg.pause_after > 0 or seg.pause_before > 0
        )
        if boundary_count:
            trace.events.append(
                TraceEvent(
                    stage="g2p",
                    name="ssmd_boundary",
                    ms=0.0,
                    details={"count": boundary_count},
                )
            )

        return phoneme_segments
