from pykokoro.pipeline import KokoroPipeline
from pykokoro.pipeline_config import PipelineConfig
from pykokoro.runtime.cache import make_g2p_key


def test_make_g2p_key_changes_with_is_phonemes():
    base = make_g2p_key(
        text="Hello",
        lang="en-us",
        is_phonemes=False,
        tokenizer_config=None,
        phoneme_override=None,
        kokorog2p_version="1.0",
    )
    alt = make_g2p_key(
        text="Hello",
        lang="en-us",
        is_phonemes=True,
        tokenizer_config=None,
        phoneme_override=None,
        kokorog2p_version="1.0",
    )

    assert base != alt


def test_kokoro_key_changes_with_model_quality():
    pipeline = KokoroPipeline(PipelineConfig())
    key_fp32 = pipeline._kokoro_key(PipelineConfig(model_quality="fp32"))
    key_fp16 = pipeline._kokoro_key(PipelineConfig(model_quality="fp16"))

    assert key_fp32 != key_fp16
