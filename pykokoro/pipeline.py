from __future__ import annotations

from dataclasses import replace
from typing import Any

from .constants import SAMPLE_RATE
from .pipeline_config import PipelineConfig
from .runtime.tracing import trace_timing
from .stages.base import DocumentParser, G2PAdapter, Splitter, Synthesizer
from .stages.doc_parsers.ssmd import SsmdDocumentParser
from .stages.g2p.tokenizer import TokenizerAdapter
from .stages.splitters.phrasplit import PhrasplitSplitter
from .stages.synth.onnx import OnnxSynthesizerAdapter
from .types import AudioResult, Trace


class KokoroPipeline:
    def __init__(
        self,
        config: PipelineConfig,
        *,
        doc_parser: DocumentParser | None = None,
        splitter: Splitter | None = None,
        g2p: G2PAdapter | None = None,
        synth: Synthesizer | None = None,
    ) -> None:
        self.config = config
        self.doc_parser = doc_parser or SsmdDocumentParser()
        self.splitter = splitter or PhrasplitSplitter()
        self.g2p = g2p or TokenizerAdapter()
        self.synth = synth or OnnxSynthesizerAdapter()

    def run(self, text: str, **overrides: Any) -> AudioResult:
        if overrides:
            lang = overrides.pop("lang", None)
            if lang is not None:
                generation = overrides.get("generation")
                if generation is None:
                    generation = replace(self.config.generation, lang=lang)
                else:
                    generation = replace(generation, lang=lang)
                overrides["generation"] = generation
            cfg = replace(self.config, **overrides)
        else:
            cfg = self.config
        trace = Trace()

        with trace_timing(trace, "doc", "parse"):
            doc = self.doc_parser.parse(text, cfg, trace)
            trace.warnings.extend(doc.warnings)

        with trace_timing(trace, "split", "split"):
            segments = self.splitter.split(doc, cfg, trace)

        with trace_timing(trace, "g2p", "phonemize"):
            phoneme_segments = self.g2p.phonemize(segments, doc, cfg, trace)

        with trace_timing(trace, "synth", "synthesize"):
            audio = self.synth.synthesize(phoneme_segments, cfg, trace)

        return AudioResult(
            audio=audio,
            sample_rate=SAMPLE_RATE,
            segments=segments,
            trace=trace if cfg.return_trace else None,
        )

    def __call__(self, text: str, **overrides: Any) -> AudioResult:
        return self.run(text, **overrides)
