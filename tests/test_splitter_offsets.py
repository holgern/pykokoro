import sys
from types import SimpleNamespace

from pykokoro.generation_config import GenerationConfig
from pykokoro.pipeline_config import PipelineConfig
from pykokoro.stages.base import DocumentResult
from pykokoro.stages.splitters.phrasplit import PhrasplitSplitter
from pykokoro.types import Trace


def test_phrasplit_splitter_handles_missing_offsets():
    text = "Hello world. How are you?"
    cfg = PipelineConfig(generation=GenerationConfig(lang="en-us"))
    doc = DocumentResult(clean_text=text)

    class DummySplitter(PhrasplitSplitter):
        def _split_with_offsets(self, phrasplit_module, text, language_model):
            return [
                ("Hello world.", None, None, None, None, None),
                ("How are you?", None, None, None, None, None),
            ]

    splitter = DummySplitter()
    dummy_module = SimpleNamespace()

    with patch_sys_modules({"phrasplit": dummy_module}):
        segments = splitter.split(doc, cfg, Trace())

    assert len(segments) == 2
    assert segments[0].text == "Hello world."
    assert segments[1].text == "How are you?"
    assert segments[0].char_start == 0
    assert segments[1].char_start == segments[0].char_end + 1


class patch_sys_modules:
    def __init__(self, updates: dict[str, object]) -> None:
        self._updates = updates
        self._originals: dict[str, object | None] = {}

    def __enter__(self):
        for key, value in self._updates.items():
            self._originals[key] = sys.modules.get(key)
            sys.modules[key] = value
        return self

    def __exit__(self, exc_type, exc, tb):
        for key, original in self._originals.items():
            if original is None:
                sys.modules.pop(key, None)
            else:
                sys.modules[key] = original
        return False
