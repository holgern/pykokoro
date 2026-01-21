from pykokoro.runtime.spans import slice_spans
from pykokoro.types import AnnotationSpan


def test_slice_spans_snap_punctuation_and_adjacent():
    spans = [
        AnnotationSpan(char_start=0, char_end=6, attrs={"lang": "en"}),
        AnnotationSpan(char_start=6, char_end=11, attrs={"lang": "fr"}),
    ]

    sliced = slice_spans(spans, 0, 11, overlap_mode="snap")

    assert [(span.char_start, span.char_end) for span in sliced] == [(0, 6), (6, 11)]


def test_slice_spans_strict_drops_partial_with_warning():
    spans = [AnnotationSpan(char_start=0, char_end=6, attrs={"lang": "en"})]
    warnings: list[str] = []

    sliced = slice_spans(
        spans,
        0,
        5,
        overlap_mode="strict",
        warnings=warnings,
    )

    assert sliced == []
    assert any("Dropped partial annotation span" in warning for warning in warnings)
