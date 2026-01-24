from pykokoro.runtime.spans import slice_boundaries, slice_spans
from pykokoro.types import AnnotationSpan, BoundaryEvent


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


def test_slice_boundaries_assigns_to_right_segment_at_join():
    boundaries = [BoundaryEvent(pos=5, kind="pause", duration_s=0.5, attrs={"a": "b"})]

    left = slice_boundaries(boundaries, 0, 5, doc_end=10)
    right = slice_boundaries(boundaries, 5, 10, doc_end=10)

    assert left == []
    assert [(b.pos, b.duration_s) for b in right] == [(0, 0.5)]


def test_slice_boundaries_keeps_initial_pause():
    boundaries = [BoundaryEvent(pos=0, kind="pause", duration_s=0.2, attrs={})]

    sliced = slice_boundaries(boundaries, 0, 4, doc_end=4)

    assert [(b.pos, b.duration_s) for b in sliced] == [(0, 0.2)]


def test_slice_helpers_copy_attrs():
    span = AnnotationSpan(char_start=0, char_end=2, attrs={"lang": "en"})
    boundary = BoundaryEvent(pos=1, kind="pause", duration_s=0.1, attrs={"k": "v"})

    sliced_span = slice_spans([span], 0, 2)[0]
    sliced_boundary = slice_boundaries([boundary], 0, 2, doc_end=2)[0]

    span.attrs["lang"] = "fr"
    boundary.attrs["k"] = "changed"

    assert sliced_span.attrs["lang"] == "en"
    assert sliced_boundary.attrs["k"] == "v"


def test_slice_boundaries_keeps_doc_end_pause():
    boundaries = [BoundaryEvent(pos=10, kind="pause", duration_s=0.4, attrs={})]

    sliced = slice_boundaries(boundaries, 5, 10, doc_end=10)

    assert [(b.pos, b.duration_s) for b in sliced] == [(5, 0.4)]
