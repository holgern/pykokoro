import os

import numpy as np
import pytest

from pykokoro import KokoroPipeline, PipelineConfig
from pykokoro.generation_config import GenerationConfig
from pykokoro.pipeline_config import PipelineConfig as PipelineConfigType
from pykokoro.stages.g2p.kokorog2p import PhonemeSegment
from pykokoro.stages.base import DocumentResult
from pykokoro.stages.doc_parsers.ssmd import SsmdDocumentParser
from pykokoro.stages.splitters.phrasplit import PhrasplitSplitter
from pykokoro.types import Trace


class DummySynth:
    def synthesize(self, phoneme_segments, cfg, trace):
        return np.zeros(240, dtype=np.float32)


class DummyDocParser:
    def parse(self, text, cfg, trace):
        return DocumentResult(clean_text=text)


class DummyG2P:
    def __init__(self) -> None:
        self.last_lang = None

    def phonemize(self, segments, doc, cfg, trace):
        self.last_lang = cfg.generation.lang
        return [
            PhonemeSegment(
                text=doc.clean_text,
                phonemes="a",
                tokens=[],
                lang=cfg.generation.lang,
            )
        ]


def test_pipeline_imports():
    assert KokoroPipeline is not None
    assert PipelineConfig is not None


def test_modular_ssmd_parser_spans_and_breaks():
    parser = SsmdDocumentParser()
    cfg = PipelineConfigType()
    trace = Trace()
    doc = parser.parse("[Bonjour](fr) le monde.", cfg, trace)
    assert doc.clean_text == "Bonjour le monde."
    assert any(span.attrs.get("lang") == "fr" for span in doc.annotation_spans)

    doc_break = parser.parse("Hello ...500ms world", cfg, trace)
    assert any(
        boundary.duration_s == 0.5 for boundary in doc_break.boundary_events
    )


def test_phrasplit_offsets_match_slices():
    pytest.importorskip("phrasplit")
    parser = SsmdDocumentParser()
    cfg = PipelineConfigType()
    trace = Trace()
    doc = parser.parse("Hello world. Second sentence.", cfg, trace)
    splitter = PhrasplitSplitter()
    segments = splitter.split(doc, cfg, trace)
    for segment in segments:
        assert segment.text == doc.clean_text[segment.char_start : segment.char_end]


def test_phrasplit_language_model_from_lang():
    splitter = PhrasplitSplitter()
    assert splitter._language_model_from_lang("en") == "en_core_web_sm"
    assert splitter._language_model_from_lang("en-us") == "en_core_web_sm"
    assert splitter._language_model_from_lang("de") == "de_core_news_sm"
    assert splitter._language_model_from_lang("zh-cn") == "zh_core_web_sm"


def test_pipeline_run_overrides_lang():
    cfg = PipelineConfig()
    g2p = DummyG2P()
    pipe = KokoroPipeline(
        cfg,
        doc_parser=DummyDocParser(),
        g2p=g2p,
        synth=DummySynth(),
    )
    res = pipe.run("Hallo", lang="de")
    assert g2p.last_lang == "de"
    assert res.segments[0].text == "Hallo"

    res = pipe.run("Salut", generation=GenerationConfig(lang="fr"), lang="it")
    assert g2p.last_lang == "it"
    assert res.segments[0].text == "Salut"


@pytest.mark.skipif(
    os.getenv("PYKOKORO_ONNX_SMOKE") != "1",
    reason="Enable with PYKOKORO_ONNX_SMOKE=1",
)
def test_onnx_smoke():
    cfg = PipelineConfig()
    res = KokoroPipeline(cfg).run("Hello")
    assert res.audio.size > 0
