from __future__ import annotations

from types import SimpleNamespace

from pykokoro.audio_generator import AudioGenerator


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
        session=SimpleNamespace(),
        tokenizer=tokenizer,
    )

    batches = generator.split_phonemes("hi")

    assert len(batches) > 1
