from __future__ import annotations

import logging
from dataclasses import replace
from pathlib import Path
from typing import Any

from .constants import SAMPLE_RATE
from .pipeline_config import PipelineConfig
from .runtime.tracing import trace_timing
from .stages.audio_generation.onnx import OnnxAudioGenerationAdapter
from .stages.audio_postprocessing.onnx import OnnxAudioPostprocessingAdapter
from .stages.doc_parsers.ssmd import SsmdDocumentParser
from .stages.g2p.kokorog2p import KokoroG2PAdapter
from .stages.phoneme_processing.onnx import OnnxPhonemeProcessorAdapter
from .stages.protocols import (
    AudioGeneratorStage,
    AudioPostprocessor,
    DocumentParser,
    G2PAdapter,
    PhonemeProcessor,
    Splitter,
)
from .stages.splitters.phrasplit import PhrasplitSplitter
from .types import AudioResult, Trace

logger = logging.getLogger(__name__)


class KokoroPipeline:
    def __init__(
        self,
        config: PipelineConfig,
        *,
        doc_parser: DocumentParser | None = None,
        splitter: Splitter | None = None,
        g2p: G2PAdapter | None = None,
        phoneme_processing: PhonemeProcessor | None = None,
        audio_generation: AudioGeneratorStage | None = None,
        audio_postprocessing: AudioPostprocessor | None = None,
    ) -> None:
        self.config = config
        self.doc_parser = doc_parser or SsmdDocumentParser()
        self.splitter = splitter or PhrasplitSplitter()
        self.g2p = g2p or KokoroG2PAdapter()
        self.phoneme_processing = phoneme_processing
        self.audio_generation = audio_generation
        self.audio_postprocessing = audio_postprocessing

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
            logger.debug("Parsing document")
            doc = self.doc_parser.parse(text, cfg, trace)
            trace.warnings.extend(doc.warnings)

        with trace_timing(trace, "split", "split"):
            logger.debug("Splitting document text")
            segments = self.splitter.split(doc, cfg, trace)

        with trace_timing(trace, "g2p", "phonemize"):
            logger.debug("Phonemizing %d segments", len(segments))
            phoneme_segments = self.g2p.phonemize(segments, doc, cfg, trace)

        phoneme_processor = self.phoneme_processing
        audio_generator = self.audio_generation
        audio_postprocessor = self.audio_postprocessing

        if (
            phoneme_processor is None
            or audio_generator is None
            or audio_postprocessor is None
        ):
            from .onnx_backend import Kokoro

            model_path = Path(cfg.model_path) if cfg.model_path else None
            voices_path = Path(cfg.voices_path) if cfg.voices_path else None
            kokoro = Kokoro(
                model_path=model_path,
                voices_path=voices_path,
                model_quality=cfg.model_quality,
                model_source=cfg.model_source,
                model_variant=cfg.model_variant,
                provider=cfg.provider,
                provider_options=cfg.provider_options,
                session_options=cfg.session_options,
                tokenizer_config=cfg.tokenizer_config,
                espeak_config=cfg.espeak_config,
                short_sentence_config=cfg.short_sentence_config,
            )
            if phoneme_processor is None:
                phoneme_processor = OnnxPhonemeProcessorAdapter(kokoro)
            if audio_generator is None:
                audio_generator = OnnxAudioGenerationAdapter(kokoro)
            if audio_postprocessor is None:
                audio_postprocessor = OnnxAudioPostprocessingAdapter(kokoro)

        assert phoneme_processor is not None
        assert audio_generator is not None
        assert audio_postprocessor is not None

        with trace_timing(trace, "phoneme_processing", "preprocess"):
            logger.debug("Preprocessing %d phoneme segments", len(phoneme_segments))
            phoneme_segments = phoneme_processor.process(phoneme_segments, cfg, trace)

        with trace_timing(trace, "audio_generation", "generate"):
            logger.debug(
                "Generating audio for %d phoneme segments", len(phoneme_segments)
            )
            phoneme_segments = audio_generator.generate(phoneme_segments, cfg, trace)

        with trace_timing(trace, "audio_postprocessing", "postprocess"):
            logger.debug("Postprocessing %d phoneme segments", len(phoneme_segments))
            audio = audio_postprocessor.postprocess(phoneme_segments, cfg, trace)

        return AudioResult(
            audio=audio,
            sample_rate=SAMPLE_RATE,
            segments=segments,
            phoneme_segments=phoneme_segments,
            trace=trace if cfg.return_trace else None,
        )

    def __call__(self, text: str, **overrides: Any) -> AudioResult:
        return self.run(text, **overrides)
