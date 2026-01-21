from __future__ import annotations

from dataclasses import asdict

from ...pipeline_config import PipelineConfig
from ...runtime.cache import cache_from_dir, make_g2p_key
from ...runtime.spans import slice_boundaries, slice_spans
from ...tokenizer import Tokenizer
from ...types import Segment, Trace
from ..base import DocumentResult, G2PAdapter
from .kokorog2p import PhonemeSegment


class TokenizerAdapter(G2PAdapter):
    def __init__(self, tokenizer: Tokenizer | None = None) -> None:
        self._tokenizer = tokenizer

    def phonemize(
        self,
        segments: list[Segment],
        doc: DocumentResult,
        cfg: PipelineConfig,
        trace: Trace,
    ) -> list[PhonemeSegment]:
        tokenizer = self._tokenizer or Tokenizer(
            config=cfg.tokenizer_config, espeak_config=cfg.espeak_config
        )
        cache = cache_from_dir(cfg.cache_dir)
        generation = cfg.generation
        out: list[PhonemeSegment] = []

        for segment in segments:
            warnings: list[str] = []
            seg_spans = slice_spans(
                doc.annotation_spans,
                segment.char_start,
                segment.char_end,
                overlap_mode=cfg.overlap_mode,
                warnings=warnings,
            )
            seg_boundaries = slice_boundaries(
                doc.boundary_events, segment.char_start, segment.char_end
            )
            trace.warnings.extend(warnings)

            lang = generation.lang
            phoneme_override = None
            ssmd_metadata: dict[str, str] = {}
            seg_len = max(0, segment.char_end - segment.char_start)

            for span in seg_spans:
                span_lang = span.attrs.get("lang")
                if span_lang:
                    lang = span_lang
                if span.char_start == 0 and span.char_end == seg_len:
                    phoneme_override = span.attrs.get("ph") or span.attrs.get(
                        "phonemes"
                    )
                self._apply_span_metadata(span.attrs, ssmd_metadata)

            cache_key = make_g2p_key(
                text=segment.text,
                lang=lang,
                tokenizer_config=asdict(cfg.tokenizer_config)
                if cfg.tokenizer_config
                else None,
                phoneme_override=phoneme_override,
                model_quality=cfg.model_quality,
                model_source=cfg.model_source,
                model_variant=cfg.model_variant,
            )
            cached = cache.get(cache_key)
            if cached is not None:
                phonemes = cached.get("phonemes", "")
                tokens = cached.get("tokens", [])
            else:
                if phoneme_override:
                    phonemes = phoneme_override
                else:
                    phonemes = tokenizer.phonemize(segment.text, lang=lang)
                tokens = tokenizer.tokenize(phonemes) if phonemes else []
                cache.set(cache_key, {"phonemes": phonemes, "tokens": tokens})

            pause_before, pause_after = self._resolve_pauses(seg_boundaries, generation)

            out.append(
                PhonemeSegment(
                    text=segment.text,
                    phonemes=phonemes,
                    tokens=list(tokens),
                    lang=lang,
                    paragraph=segment.paragraph_idx or 0,
                    sentence=segment.sentence_idx,
                    pause_before=pause_before,
                    pause_after=pause_after,
                    ssmd_metadata=ssmd_metadata or None,
                )
            )

        return out

    def _apply_span_metadata(
        self, attrs: dict[str, str], metadata: dict[str, str]
    ) -> None:
        if not attrs:
            return
        if "voice" in attrs:
            metadata["voice_name"] = attrs["voice"]
        if "voice_name" in attrs:
            metadata["voice_name"] = attrs["voice_name"]
        if "prosody_rate" in attrs:
            metadata["prosody_rate"] = attrs["prosody_rate"]
        if "rate" in attrs:
            metadata.setdefault("prosody_rate", attrs["rate"])
        if "prosody_pitch" in attrs:
            metadata["prosody_pitch"] = attrs["prosody_pitch"]
        if "pitch" in attrs:
            metadata.setdefault("prosody_pitch", attrs["pitch"])
        if "prosody_volume" in attrs:
            metadata["prosody_volume"] = attrs["prosody_volume"]
        if "volume" in attrs:
            metadata.setdefault("prosody_volume", attrs["volume"])
        if "lang" in attrs:
            metadata["language"] = attrs["lang"]
        if "ph" in attrs:
            metadata["phonemes"] = attrs["ph"]
        if "phonemes" in attrs:
            metadata["phonemes"] = attrs["phonemes"]

    def _resolve_pauses(self, boundaries, generation):
        pause_before = 0.0
        pause_after = 0.0
        for boundary in boundaries:
            if boundary.kind != "pause":
                continue
            duration = boundary.duration_s
            if duration is None:
                strength = boundary.attrs.get("strength")
                if strength == "c":
                    duration = generation.pause_clause
                elif strength == "s":
                    duration = generation.pause_sentence
                elif strength == "p":
                    duration = generation.pause_paragraph
                elif strength == "w":
                    duration = 0.15
                elif strength == "n":
                    duration = 0.0
            if duration is None:
                continue
            if boundary.pos == 0:
                pause_before = max(pause_before, duration)
            else:
                pause_after = max(pause_after, duration)
        return pause_before, pause_after
