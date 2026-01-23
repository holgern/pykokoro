from __future__ import annotations

from types import SimpleNamespace
from typing import Any, cast

from pykokoro.audio_generator import AudioGenerator
from pykokoro.constants import MAX_PHONEME_LENGTH
from pykokoro.types import PhonemeSegment


class DummyTokenizer:
    def __init__(self, factor: int) -> None:
        self.factor = factor

    def tokenize(self, text: str):
        return list(range(len(text) * self.factor))

    def detokenize(self, tokens):
        if not tokens:
            return ""
        return "a" * max(1, len(tokens) // self.factor)


def test_split_phonemes_uses_token_count():
    tokenizer = DummyTokenizer(factor=300)
    generator = AudioGenerator(
        session=cast(Any, SimpleNamespace()),
        tokenizer=cast(Any, tokenizer),
    )

    batches = generator.split_phonemes("hi")

    assert len(batches) > 1


def test_preprocess_pipeline_splits_segments():
    tokenizer = DummyTokenizer(factor=MAX_PHONEME_LENGTH)
    generator = AudioGenerator(
        session=cast(Any, SimpleNamespace()),
        tokenizer=cast(Any, tokenizer),
    )
    segment = PhonemeSegment(
        id="seg_1",
        segment_id="seg_1",
        phoneme_id=0,
        text="hello",
        phonemes="aa",
        tokens=[],
        pause_before=0.3,
        pause_after=0.7,
    )

    processed = generator._preprocess_segments(
        [segment], enable_short_sentence_override=False
    )

    assert len(processed) == 2
    assert processed[0].id == "seg_1_ph0"
    assert processed[1].id == "seg_1_ph1"
    assert processed[0].pause_before == 0.3
    assert processed[0].pause_after == 0.0
    assert processed[1].pause_before == 0.0
    assert processed[1].pause_after == 0.7
    assert len(processed[0].tokens) == MAX_PHONEME_LENGTH
    assert len(processed[1].tokens) == MAX_PHONEME_LENGTH
