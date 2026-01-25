from __future__ import annotations

import logging
from dataclasses import replace
from pathlib import Path
from typing import TYPE_CHECKING, Any

from .constants import SAMPLE_RATE
from .generation_config import GenerationConfig
from .onnx_backend import LANG_CODE_TO_ONNX
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
)
from .types import AudioResult, Segment, Trace

logger = logging.getLogger(__name__)

if TYPE_CHECKING:
    from .onnx_backend import Kokoro


def _default_lang_from_voice(cfg: PipelineConfig) -> PipelineConfig:
    if not isinstance(cfg.voice, str):
        return cfg
    if "," in cfg.voice or ":" in cfg.voice:
        return cfg
    default_lang = GenerationConfig().lang
    if cfg.generation.lang != default_lang:
        return cfg
    voice_key = cfg.voice.split("_", 1)[0].strip().lower()
    if not voice_key:
        return cfg
    voice_lang = LANG_CODE_TO_ONNX.get(voice_key[0])
    if not voice_lang or voice_lang == cfg.generation.lang:
        return cfg
    generation = replace(cfg.generation, lang=voice_lang)
    return replace(cfg, generation=generation)


class KokoroPipeline:
    def __init__(
        self,
        config: PipelineConfig,
        *,
        doc_parser: DocumentParser | None = None,
        g2p: G2PAdapter | None = None,
        phoneme_processing: PhonemeProcessor | None = None,
        audio_generation: AudioGeneratorStage | None = None,
        audio_postprocessing: AudioPostprocessor | None = None,
    ) -> None:
        self.config = config
        self.doc_parser = doc_parser or SsmdDocumentParser()
        self.g2p = g2p or KokoroG2PAdapter()
        self.phoneme_processing = phoneme_processing
        self.audio_generation = audio_generation
        self.audio_postprocessing = audio_postprocessing
        self._kokoro: Kokoro | None = None
        self._kokoro_config_key: tuple[object, ...] | None = None
        self._owns_kokoro = False
        self._owns_phoneme_processing = False
        self._owns_audio_generation = False
        self._owns_audio_postprocessing = False

    def __enter__(self) -> KokoroPipeline:
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.close()

    def close(self) -> None:
        if self._owns_phoneme_processing:
            self._close_stage(self.phoneme_processing)
            self.phoneme_processing = None
            self._owns_phoneme_processing = False
        if self._owns_audio_generation:
            self._close_stage(self.audio_generation)
            self.audio_generation = None
            self._owns_audio_generation = False
        if self._owns_audio_postprocessing:
            self._close_stage(self.audio_postprocessing)
            self.audio_postprocessing = None
            self._owns_audio_postprocessing = False
        if self._kokoro is not None and self._owns_kokoro:
            self._kokoro.close()
        self._kokoro = None
        self._kokoro_config_key = None

    def _kokoro_key(self, cfg: PipelineConfig) -> tuple[object, ...]:
        model_path = str(cfg.model_path) if cfg.model_path else None
        voices_path = str(cfg.voices_path) if cfg.voices_path else None
        return (
            model_path,
            voices_path,
            cfg.model_quality,
            cfg.model_source,
            cfg.model_variant,
            cfg.provider,
            cfg.provider_options,
            cfg.session_options,
            cfg.tokenizer_config,
            cfg.espeak_config,
            cfg.short_sentence_config,
        )

    @staticmethod
    def _close_stage(stage: object | None) -> None:
        if stage is None:
            return
        close = getattr(stage, "close", None)
        if callable(close):
            close()

    def _ensure_kokoro(self, cfg: PipelineConfig) -> tuple[Kokoro, bool]:
        kokoro_key = self._kokoro_key(cfg)
        if self._kokoro is not None and self._kokoro_config_key == kokoro_key:
            return self._kokoro, False
        if self._kokoro is not None and self._owns_kokoro:
            self._kokoro.close()
        from .onnx_backend import Kokoro

        self._kokoro = Kokoro(
            model_path=Path(cfg.model_path) if cfg.model_path else None,
            voices_path=Path(cfg.voices_path) if cfg.voices_path else None,
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
        self._kokoro_config_key = kokoro_key
        self._owns_kokoro = True
        return self._kokoro, True

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
        cfg = _default_lang_from_voice(cfg)
        trace = Trace()

        with trace_timing(trace, "doc", "parse"):
            logger.debug("Parsing document")
            doc = self.doc_parser.parse(text, cfg, trace)
            trace.warnings.extend(doc.warnings)
        segments = doc.segments
        if not segments and doc.clean_text:
            segments = [
                Segment(
                    id="p0_s0_c0_seg0",
                    text=doc.clean_text,
                    char_start=0,
                    char_end=len(doc.clean_text),
                    paragraph_idx=0,
                    sentence_idx=0,
                    clause_idx=0,
                )
            ]

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
            kokoro, kokoro_changed = self._ensure_kokoro(cfg)
            if phoneme_processor is None or (
                kokoro_changed and self._owns_phoneme_processing
            ):
                phoneme_processor = OnnxPhonemeProcessorAdapter(kokoro)
                if self.phoneme_processing is None or self._owns_phoneme_processing:
                    self.phoneme_processing = phoneme_processor
                    self._owns_phoneme_processing = True
            if audio_generator is None or (
                kokoro_changed and self._owns_audio_generation
            ):
                audio_generator = OnnxAudioGenerationAdapter(kokoro)
                if self.audio_generation is None or self._owns_audio_generation:
                    self.audio_generation = audio_generator
                    self._owns_audio_generation = True
            if audio_postprocessor is None or (
                kokoro_changed and self._owns_audio_postprocessing
            ):
                audio_postprocessor = OnnxAudioPostprocessingAdapter(kokoro)
                if self.audio_postprocessing is None or self._owns_audio_postprocessing:
                    self.audio_postprocessing = audio_postprocessor
                    self._owns_audio_postprocessing = True

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
