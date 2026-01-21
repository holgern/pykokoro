from __future__ import annotations

from ...pipeline_config import PipelineConfig
from ...types import BoundaryEvent, Segment, Trace
from ..base import DocumentResult, Splitter


class PhrasplitSplitter(Splitter):
    def split(
        self, doc: DocumentResult, cfg: PipelineConfig, trace: Trace
    ) -> list[Segment]:
        text = doc.clean_text
        language_model = self._language_model_from_lang(cfg.generation.lang)
        try:
            import phrasplit  # type: ignore
        except Exception:
            return [
                Segment(
                    id="p0_s0_c0_seg0",
                    text=text,
                    char_start=0,
                    char_end=len(text),
                    paragraph_idx=0,
                    sentence_idx=0,
                    clause_idx=0,
                )
            ]

        override_ranges = self._override_ranges(doc.annotation_spans)
        ranges = self._hard_ranges(text, doc.boundary_events, override_ranges)
        segments: list[Segment] = []
        seg_idx = 0
        sentence_idx = 0

        for start, end in ranges:
            if end <= start:
                continue
            chunk = text[start:end]
            if (start, end) in override_ranges:
                split_items = [(chunk, 0, len(chunk), None, None, None)]
            else:
                split_items = self._split_with_offsets(phrasplit, chunk, language_model)
                if not split_items:
                    split_items = [(chunk, 0, len(chunk), None, None, None)]

            cursor = 0

            chunk_len = len(chunk)

            for item in split_items:
                seg_text, seg_start, seg_end, para, sent, clause = item
                if seg_text is None:
                    continue

                if (
                    seg_start is None
                    or seg_end is None
                    or seg_start < 0
                    or seg_end < seg_start
                    or seg_end > chunk_len
                ):
                    found = chunk.find(seg_text, cursor) if seg_text else -1
                    if found >= 0:
                        seg_start = found
                        seg_end = found + len(seg_text)
                    else:
                        seg_start = cursor
                        seg_end = cursor + len(seg_text)

                seg_start = max(0, min(seg_start, chunk_len))
                seg_end = max(seg_start, min(seg_end, chunk_len))

                abs_start = start + seg_start
                abs_end = start + seg_end
                resolved_sentence = sent if sent is not None else sentence_idx
                if sent is None:
                    sentence_idx += 1
                else:
                    sentence_idx = max(sentence_idx, sent + 1)
                resolved_paragraph = para if para is not None else 0
                resolved_clause = clause if clause is not None else 0
                segment_id = (
                    f"p{resolved_paragraph}"
                    f"_s{resolved_sentence}"
                    f"_c{resolved_clause}"
                    f"_seg{seg_idx}"
                )
                segments.append(
                    Segment(
                        id=segment_id,
                        text=text[abs_start:abs_end],
                        char_start=abs_start,
                        char_end=abs_end,
                        paragraph_idx=resolved_paragraph,
                        sentence_idx=resolved_sentence,
                        clause_idx=resolved_clause,
                    )
                )
                seg_idx += 1
                cursor = max(cursor, seg_end)

        if not segments:
            segments.append(
                Segment(
                    id="p0_s0_c0_seg0",
                    text=text,
                    char_start=0,
                    char_end=len(text),
                    paragraph_idx=0,
                    sentence_idx=0,
                    clause_idx=0,
                )
            )
        return segments

    def _hard_ranges(
        self,
        text: str,
        boundaries: list[BoundaryEvent],
        override_ranges: set[tuple[int, int]],
    ) -> list[tuple[int, int]]:
        positions = {b.pos for b in boundaries}
        for start, end in override_ranges:
            positions.add(start)
            positions.add(end)
        positions = sorted(positions)
        ranges: list[tuple[int, int]] = []
        start = 0
        for pos in positions:
            pos = max(0, min(len(text), pos))
            if pos > start:
                ranges.append((start, pos))
            start = pos
        if start < len(text):
            ranges.append((start, len(text)))
        if not ranges:
            ranges.append((0, len(text)))
        return ranges

    def _override_ranges(self, spans) -> set[tuple[int, int]]:
        ranges: set[tuple[int, int]] = set()
        for span in spans:
            if "ph" in span.attrs or "phonemes" in span.attrs:
                ranges.add((span.char_start, span.char_end))
        return ranges

    def _split_with_offsets(self, phrasplit_module, text: str, language_model: str):
        kwargs: dict[str, object] = {
            "mode": "sentence",
            "language_model": language_model,
        }
        for key in ("apply_corrections", "split_on_colon"):
            kwargs[key] = True

        if hasattr(phrasplit_module, "split_with_offsets"):
            try:
                segments = phrasplit_module.split_with_offsets(text, **kwargs)
            except (OSError, TypeError):
                try:
                    segments = phrasplit_module.split_with_offsets(
                        text,
                        mode="sentence",
                        language_model=language_model,
                    )
                except OSError:
                    return []
        elif hasattr(phrasplit_module, "iter_split_with_offsets"):
            try:
                segments = list(
                    phrasplit_module.iter_split_with_offsets(text, **kwargs)
                )
            except (OSError, TypeError):
                try:
                    segments = list(
                        phrasplit_module.iter_split_with_offsets(
                            text,
                            mode="sentence",
                            language_model=language_model,
                        )
                    )
                except OSError:
                    return []
        else:
            return []

        out = []
        for seg in segments:
            seg_text = getattr(seg, "text", None)
            start = getattr(seg, "start", None)
            end = getattr(seg, "end", None)
            para = getattr(seg, "paragraph", None)
            sent = getattr(seg, "sentence", None)
            clause = getattr(seg, "clause", None)
            if seg_text is None and start is not None and end is not None:
                seg_text = text[start:end]
            if seg_text is None:
                continue
            out.append((seg_text, start, end, para, sent, clause))
        return out

    def _language_model_from_lang(self, lang: str | None) -> str:
        lang_code = (lang or "en").lower()
        for sep in ("-", "_"):
            if sep in lang_code:
                lang_code = lang_code.split(sep, 1)[0]
                break
        if not lang_code:
            lang_code = "en"
        web_langs = {"en", "zh"}
        size = "sm"
        if lang_code in web_langs:
            return f"{lang_code}_core_web_{size}"
        return f"{lang_code}_core_news_{size}"
