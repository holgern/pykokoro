from pykokoro.generation_config import GenerationConfig
from pykokoro.pipeline_config import PipelineConfig
from pykokoro.stages.g2p.kokorog2p import KokoroG2PAdapter
from pykokoro.stages.protocols import DocumentResult
from pykokoro.types import AnnotationSpan, Segment, Trace


def test_phoneme_override_requires_exact_span():
    text = "Hello world"
    doc = DocumentResult(
        clean_text=text,
        annotation_spans=[
            AnnotationSpan(char_start=0, char_end=len(text), attrs={"ph": "OVERRIDE"})
        ],
    )
    segments = [
        Segment(
            id="seg_0",
            text="Hello",
            char_start=0,
            char_end=5,
            paragraph_idx=0,
            sentence_idx=0,
        ),
        Segment(
            id="seg_1",
            text="world",
            char_start=6,
            char_end=11,
            paragraph_idx=0,
            sentence_idx=1,
        ),
    ]
    cfg = PipelineConfig(generation=GenerationConfig(lang="en-us"))
    trace = Trace()

    phoneme_segments = KokoroG2PAdapter().phonemize(segments, doc, cfg, trace)

    assert all(seg.phonemes != "OVERRIDE" for seg in phoneme_segments)
    assert any("Skipped phoneme override" in warning for warning in trace.warnings)
