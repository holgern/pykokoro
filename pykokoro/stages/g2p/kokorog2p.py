from __future__ import annotations

from ...phonemes import PhonemeSegment
from ...pipeline_config import PipelineConfig
from ...runtime.cache import cache_from_dir, make_g2p_key
from ...runtime.spans import slice_spans
from ...types import Segment, Trace
from ..base import DocumentResult, G2PAdapter


class KokoroG2PAdapter(G2PAdapter):
    def __init__(self) -> None:
        self._g2p = None

    def _load(self):
        if self._g2p is not None:
            return self._g2p
        try:
            import kokorog2p  # type: ignore
        except Exception as exc:
            raise RuntimeError("kokorog2p is not installed") from exc
        self._g2p = kokorog2p
        return self._g2p

    def phonemize(
        self,
        segments: list[Segment],
        doc: DocumentResult,
        cfg: PipelineConfig,
        trace: Trace,
    ) -> list[PhonemeSegment]:
        g2p = self._load()
        cache = cache_from_dir(cfg.cache_dir)
        generation = cfg.generation
        out: list[PhonemeSegment] = []

        for segment in segments:
            span_list = slice_spans(
                doc.annotation_spans,
                segment.char_start,
                segment.char_end,
                overlap_mode=cfg.overlap_mode,
            )
            lang = generation.lang
            seg_len = max(0, segment.char_end - segment.char_start)
            phoneme_override = None
            for span in span_list:
                if span.attrs.get("lang"):
                    lang = span.attrs["lang"]
                if span.char_start == 0 and span.char_end == seg_len:
                    phoneme_override = span.attrs.get("ph") or span.attrs.get(
                        "phonemes"
                    )

            cache_key = make_g2p_key(
                text=segment.text,
                lang=lang,
                tokenizer_config=None,
                phoneme_override=phoneme_override,
                kokorog2p_version=getattr(g2p, "__version__", None),
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
                    tokens = []
                else:
                    result = g2p.phonemize_to_result(
                        segment.text,
                        lang=lang,
                        return_phonemes=True,
                        return_ids=True,
                    )
                    phonemes = getattr(result, "phonemes", None) or getattr(
                        result, "phoneme", ""
                    )
                    tokens = getattr(result, "ids", None) or getattr(
                        result, "token_ids", []
                    )
                    warnings = getattr(result, "warnings", None)
                    if warnings:
                        trace.warnings.extend(list(warnings))
                cache.set(cache_key, {"phonemes": phonemes, "tokens": tokens})

            out.append(
                PhonemeSegment(
                    text=segment.text,
                    phonemes=str(phonemes),
                    tokens=list(tokens),
                    lang=lang,
                    paragraph=segment.paragraph_idx or 0,
                    sentence=segment.sentence_idx,
                )
            )

        return out
