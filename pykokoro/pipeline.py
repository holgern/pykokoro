from __future__ import annotations

import time
from dataclasses import replace
from typing import Any

import numpy as np

from .config import PipelineConfig
from .exceptions import ConfigurationError
from .runtime.cache import Cache, NullCache
from .runtime.tracing import trace_timing
from .types import Annotation, AudioResult, PhonemeSegment, Segment, Trace

from .stages.splitters.base import Splitter
from .stages.splitters.noop import NoSplitSplitter
from .stages.splitters.phrasplit import PhrasplitSplitter

from .stages.markup.base import MarkupParser
from .stages.markup.plain import PlainTextParser
from .stages.markup.ssmd import SsmdMarkupParser

from .stages.g2p.base import G2P
from .stages.g2p.kokorog2p import KokoroG2PAdapter

from .stages.synth.base import Synthesizer
from .stages.synth.onnx import OnnxSynthesizer


class KokoroPipeline:
    def __init__(
        self,
        config: PipelineConfig,
        *,
        splitter: Splitter | None = None,
        markup: MarkupParser | None = None,
        g2p: G2P | None = None,
        synthesizer: Synthesizer | None = None,
        cache: Cache | None = None,
    ) -> None:
        self.config = config
        self.cache = cache or (Cache.from_dir(config.cache_dir) if config.cache_dir else NullCache())

        self.splitter = splitter or self._default_splitter(config)
        self.markup = markup or self._default_markup(config)
        self.g2p = g2p or KokoroG2PAdapter(config)
        self.synthesizer = synthesizer or OnnxSynthesizer(config)

    def _default_splitter(self, cfg: PipelineConfig) -> Splitter:
        if cfg.split == "none" or cfg.max_chars is None:
            return NoSplitSplitter()
        if cfg.split in ("auto", "phrasplit"):
            return PhrasplitSplitter()
        raise ConfigurationError(f"Unknown split mode: {cfg.split}")

    def _default_markup(self, cfg: PipelineConfig) -> MarkupParser:
        if cfg.markup == "none":
            return PlainTextParser()
        if cfg.markup in ("auto", "ssmd"):
            return SsmdMarkupParser()
        raise ConfigurationError(f"Unknown markup mode: {cfg.markup}")

    def run(self, text: str, **overrides: Any) -> AudioResult:
        cfg = replace(self.config, **overrides) if overrides else self.config
        trace = Trace() if cfg.return_trace else None

        # 1) split
        with trace_timing(trace, "split", "split"):
            segments = self.splitter.split(text, max_chars=cfg.max_chars)
        if trace:
            trace.segments = segments

        # 2) parse markup per segment -> clean_text + annotations
        clean_texts: list[str] = []
        per_segment_annotations: list[list[Annotation]] = []
        with trace_timing(trace, "markup", "parse"):
            for seg in segments:
                clean, ann = self.markup.parse(seg.text, default_lang=cfg.lang)
                clean_texts.append(clean)
                per_segment_annotations.append(ann)
        if trace:
            trace.clean_texts = clean_texts
            trace.annotations = per_segment_annotations

        # 3) g2p
        with trace_timing(trace, "g2p", "phonemize"):
            phoneme_segments = self.g2p.phonemize(
                segments=segments,
                clean_texts=clean_texts,
                annotations=per_segment_annotations,
            )
        if trace:
            trace.phoneme_segments = phoneme_segments

        # 4) synth
        with trace_timing(trace, "synth", "synthesize"):
            audio, sr = self.synthesizer.synth(phoneme_segments, segments=segments)

        return AudioResult(audio=audio, sample_rate=sr, segments=segments, trace=trace)

    def __call__(self, text: str, **overrides: Any) -> AudioResult:
        return self.run(text, **overrides)
