from pykokoro.generation_config import GenerationConfig
from pykokoro.pipeline_config import PipelineConfig
from pykokoro.stages.g2p.kokorog2p import KokoroG2PAdapter
from pykokoro.stages.protocols import DocumentResult
from pykokoro.types import Segment, Trace


def test_kokorog2p_preserves_punctuation():
    text = (
        "'That's ridiculous!' I protested. "
        "'I'm not gonna stand here and let you insult me! "
        "What's your problem anyway?'"
    )
    cfg = PipelineConfig(generation=GenerationConfig(lang="en-us"))
    doc = DocumentResult(clean_text=text)
    segments = [
        Segment(
            id="seg_0",
            text=text,
            char_start=0,
            char_end=len(text),
            paragraph_idx=0,
            sentence_idx=0,
        )
    ]
    g2p = KokoroG2PAdapter()
    trace = Trace()
    phoneme_segments = g2p.phonemize(segments, doc, cfg, trace)

    assert len(phoneme_segments) == 1
    phonemes = phoneme_segments[0].phonemes
    assert "wˌʌts" in phonemes  # "What's" should be preserved
    assert "!" in phonemes
    assert "?" in phonemes
    assert "." in phonemes
    assert "'" not in phonemes
    assert '"' not in phonemes
    assert "“" in phonemes or "”" in phonemes
