import numpy as np

import pykokoro.onnx_backend as onnx_backend
import pykokoro.pipeline as pipeline
from pykokoro.pipeline import KokoroPipeline
from pykokoro.pipeline_config import PipelineConfig
from pykokoro.stages.protocols import DocumentResult
from pykokoro.types import PhonemeSegment, Segment, Trace


class DummyDocumentParser:
    def parse(self, text: str, cfg: PipelineConfig, trace: Trace) -> DocumentResult:
        _ = cfg, trace
        segment = Segment(
            id="seg_0",
            text=text,
            char_start=0,
            char_end=len(text),
            paragraph_idx=0,
            sentence_idx=0,
            clause_idx=0,
        )
        return DocumentResult(clean_text=text, segments=[segment])


class DummyG2P:
    def phonemize(self, segments, doc, cfg: PipelineConfig, trace: Trace):
        _ = doc, trace
        out = []
        for segment in segments:
            out.append(
                PhonemeSegment(
                    id=f"{segment.id}_ph0",
                    segment_id=segment.id,
                    phoneme_id=0,
                    text=segment.text,
                    phonemes="a",
                    tokens=[1],
                    lang=cfg.generation.lang,
                    char_start=segment.char_start,
                    char_end=segment.char_end,
                    paragraph_idx=segment.paragraph_idx,
                    sentence_idx=segment.sentence_idx,
                    clause_idx=segment.clause_idx,
                )
            )
        return out


def _patch_kokoro(monkeypatch):
    created: list[object] = []
    closed: list[object] = []

    class FakeKokoro:
        def __init__(self, **kwargs):
            _ = kwargs
            created.append(self)

        def close(self) -> None:
            closed.append(self)

    class FakePhonemeProcessor:
        def __init__(self, kokoro):
            self._kokoro = kokoro

        def process(self, phoneme_segments, cfg, trace):
            _ = cfg, trace
            return phoneme_segments

    class FakeAudioGenerator:
        def __init__(self, kokoro):
            self._kokoro = kokoro

        def generate(self, phoneme_segments, cfg, trace):
            _ = cfg, trace
            return phoneme_segments

    class FakeAudioPostprocessor:
        def __init__(self, kokoro):
            self._kokoro = kokoro

        def postprocess(self, phoneme_segments, cfg, trace):
            _ = phoneme_segments, cfg, trace
            return np.zeros(1, dtype=np.float32)

    monkeypatch.setattr(onnx_backend, "Kokoro", FakeKokoro)
    monkeypatch.setattr(pipeline, "OnnxPhonemeProcessorAdapter", FakePhonemeProcessor)
    monkeypatch.setattr(pipeline, "OnnxAudioGenerationAdapter", FakeAudioGenerator)
    monkeypatch.setattr(
        pipeline, "OnnxAudioPostprocessingAdapter", FakeAudioPostprocessor
    )

    return created, closed


def test_pipeline_close_idempotent(monkeypatch):
    created, closed = _patch_kokoro(monkeypatch)
    cfg = PipelineConfig()
    pipeline_instance = KokoroPipeline(
        cfg, doc_parser=DummyDocumentParser(), g2p=DummyG2P()
    )

    pipeline_instance.run("Hello")
    pipeline_instance.run("World")

    assert len(created) == 1

    pipeline_instance.close()
    pipeline_instance.close()

    assert len(closed) == 1


def test_pipeline_context_manager_closes(monkeypatch):
    created, closed = _patch_kokoro(monkeypatch)
    cfg = PipelineConfig()
    with KokoroPipeline(cfg, doc_parser=DummyDocumentParser(), g2p=DummyG2P()) as pipe:
        pipe.run("Hello")

    assert len(created) == 1
    assert len(closed) == 1
