from __future__ import annotations

from dataclasses import replace
from typing import Any

from .constants import SAMPLE_RATE
from .pipeline_config import PipelineConfig
from .runtime.tracing import trace_timing
from .stages.base import DocumentParser, G2PAdapter, Splitter, Synthesizer
from .stages.doc_parsers.ssmd import SsmdDocumentParser
from .stages.doc_parsers.ssmd_compat import SsmdCompatDocumentParser
from .stages.g2p.tokenizer import TokenizerAdapter
from .stages.g2p.tokenizer_compat import TokenizerCompatG2PAdapter
from .stages.splitters.compat import CompatSplitter
from .stages.splitters.phrasplit import PhrasplitSplitter
from .stages.synth.onnx import OnnxSynthesizerAdapter
from .types import AudioResult, PhonemeSegment, Segment, Trace


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
        if config.mode == "modular":
            self.doc_parser = doc_parser or SsmdDocumentParser()
            self.splitter = splitter or PhrasplitSplitter()
            self.g2p = g2p or TokenizerAdapter()
        else:
            self.doc_parser = doc_parser or SsmdCompatDocumentParser()
            self.splitter = splitter or CompatSplitter()
            self.g2p = g2p or TokenizerCompatG2PAdapter()
        self.synth = synth or OnnxSynthesizerAdapter()

    def _build_segments_from_phonemes(
        self, text: str, phoneme_segments: list[PhonemeSegment]
    ) -> list[Segment]:
        segments: list[Segment] = []
        cursor = 0
        for idx, seg in enumerate(phoneme_segments):
            seg_text = seg.text
            if seg_text:
                pos = text.find(seg_text, cursor)
                if pos < 0:
                    pos = cursor
                start = pos
                end = pos + len(seg_text)
                cursor = end
            else:
                start = cursor
                end = cursor
            segments.append(
                Segment(
                    id=f"seg_{idx}",
                    text=seg_text,
                    char_start=start,
                    char_end=end,
                    paragraph_idx=seg.paragraph,
                    sentence_idx=seg.sentence
                    if isinstance(seg.sentence, int)
                    else None,
                )
            )
        return segments

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

        if cfg.mode == "compat":
            with trace_timing(trace, "doc", "parse"):
                doc = self.doc_parser.parse(text, cfg, trace)
                trace.warnings.extend(doc.warnings)

            with trace_timing(trace, "g2p", "phonemize"):
                phoneme_segments = self.g2p.phonemize([], doc, cfg, trace)

            segments = self._build_segments_from_phonemes(text, phoneme_segments)

            with trace_timing(trace, "synth", "synthesize"):
                audio = self.synth.synthesize(phoneme_segments, cfg, trace)

            return AudioResult(
                audio=audio,
                sample_rate=SAMPLE_RATE,
                segments=segments,
                trace=trace if cfg.return_trace else None,
            )

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
