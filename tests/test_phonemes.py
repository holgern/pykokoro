"""Tests for pykokoro.phonemes module."""

import pytest

from pykokoro.phonemes import (
    PhonemeSegment,
    phonemize_text_list,
    split_and_phonemize_text,
)
from pykokoro.tokenizer import Tokenizer, create_tokenizer


class TestPhonemeSegment:
    """Tests for PhonemeSegment dataclass."""

    def test_create_basic(self):
        """Test basic segment creation."""
        segment = PhonemeSegment(
            text="hello",
            phonemes="həˈloʊ",
            tokens=[50, 83, 156, 54, 57, 135],
        )
        assert segment.text == "hello"
        assert segment.phonemes == "həˈloʊ"
        assert len(segment.tokens) == 6
        assert segment.lang == "en-us"  # Default

    def test_create_with_lang(self):
        """Test segment creation with custom language."""
        segment = PhonemeSegment(
            text="hello",
            phonemes="həˈləʊ",
            tokens=[50, 83, 156, 54, 83, 135],
            lang="en-gb",
        )
        assert segment.lang == "en-gb"

    def test_to_dict(self):
        """Test conversion to dictionary."""
        segment = PhonemeSegment(
            text="hello",
            phonemes="həˈloʊ",
            tokens=[1, 2, 3],
            lang="en-us",
        )
        d = segment.to_dict()
        assert d["text"] == "hello"
        assert d["phonemes"] == "həˈloʊ"
        assert d["tokens"] == [1, 2, 3]
        assert d["lang"] == "en-us"

    def test_from_dict(self):
        """Test creation from dictionary."""
        d = {
            "text": "hello",
            "phonemes": "həˈloʊ",
            "tokens": [1, 2, 3],
            "lang": "en-us",
        }
        segment = PhonemeSegment.from_dict(d)
        assert segment.text == "hello"
        assert segment.phonemes == "həˈloʊ"
        assert segment.tokens == [1, 2, 3]
        assert segment.lang == "en-us"

    def test_from_dict_default_lang(self):
        """Test creation from dictionary without lang."""
        d = {
            "text": "hello",
            "phonemes": "həˈloʊ",
            "tokens": [1, 2, 3],
        }
        segment = PhonemeSegment.from_dict(d)
        assert segment.lang == "en-us"  # Default

    def test_format_readable(self):
        """Test human-readable formatting."""
        segment = PhonemeSegment(
            text="hello",
            phonemes="həˈloʊ",
            tokens=[1, 2, 3],
        )
        readable = segment.format_readable()
        assert readable == "hello [həˈloʊ]"


class TestHelperFunctions:
    """Tests for helper functions."""

    @pytest.fixture
    def tokenizer(self):
        """Create a tokenizer instance."""
        return Tokenizer()

    def test_phonemize_text_list(self, tokenizer):
        """Test phonemizing a list of texts."""
        texts = ["hello", "world"]
        segments = phonemize_text_list(texts, tokenizer)

        assert len(segments) == 2
        assert segments[0].text == "hello"
        assert segments[1].text == "world"
        assert all(len(s.phonemes) > 0 for s in segments)
        assert all(len(s.tokens) > 0 for s in segments)


class TestSplitAndPhonemizeText:
    """Tests for split_and_phonemize_text() standalone function."""

    @pytest.fixture
    def tokenizer(self):
        """Create a tokenizer instance."""
        return create_tokenizer()

    def test_basic_sentence_splitting(self, tokenizer):
        """Test basic text splitting with sentence mode."""
        text = "Hello world. How are you?"
        segments = split_and_phonemize_text(text, tokenizer, split_mode="sentence")

        assert len(segments) >= 1
        assert all(isinstance(s, PhonemeSegment) for s in segments)
        assert all(s.phonemes for s in segments)
        assert all(s.tokens for s in segments)

    def test_paragraph_mode(self, tokenizer):
        """Test paragraph splitting mode."""
        text = "First paragraph.\n\nSecond paragraph."
        segments = split_and_phonemize_text(text, tokenizer, split_mode="paragraph")

        assert len(segments) >= 1
        assert all(s.phonemes for s in segments)

    def test_clause_mode(self, tokenizer):
        """Test clause splitting mode."""
        text = "Hello, world. This is a test, with commas."
        segments = split_and_phonemize_text(text, tokenizer, split_mode="clause")

        # Clause mode should create more segments due to commas
        assert len(segments) >= 1
        assert all(s.phonemes for s in segments)

    def test_long_text_recursive_splitting(self, tokenizer):
        """Test that long text gets split recursively to meet phoneme limit."""
        # Create a very long text that will exceed phoneme limit
        long_text = " ".join(["word"] * 200)
        segments = split_and_phonemize_text(
            long_text, tokenizer, max_chars=100, split_mode="sentence"
        )

        # Should create multiple segments
        assert len(segments) > 1
        # All segments should have phonemes within limit
        for seg in segments:
            assert len(seg.phonemes) <= 510

    def test_empty_text(self, tokenizer):
        """Test handling of empty text."""
        segments = split_and_phonemize_text("", tokenizer)
        assert segments == []

    def test_whitespace_only(self, tokenizer):
        """Test handling of whitespace-only text."""
        segments = split_and_phonemize_text("   \n\n   ", tokenizer)
        assert segments == []

    def test_paragraph_and_sentence_metadata(self, tokenizer):
        """Test that paragraph and sentence indices are set."""
        text = "Sentence one. Sentence two."
        segments = split_and_phonemize_text(text, tokenizer, split_mode="sentence")

        # All segments should have paragraph index
        assert all(isinstance(s.paragraph, int) for s in segments)
        # Some should have sentence index
        assert any(s.sentence is not None for s in segments)

    def test_lang_parameter(self, tokenizer):
        """Test that language parameter is passed through."""
        text = "Hello"
        segments = split_and_phonemize_text(text, tokenizer, lang="en-gb")

        assert len(segments) >= 1
        assert all(s.lang == "en-gb" for s in segments)

    def test_warning_callback_on_truncation(self, tokenizer):
        """Test that warning callback is called for very long phonemes."""
        warnings = []

        def warn_callback(msg: str):
            warnings.append(msg)

        # Create text that will be very long (force truncation by setting low max_chars)
        long_text = "supercalifragilisticexpialidocious" * 50
        segments = split_and_phonemize_text(
            long_text,
            tokenizer,
            max_chars=10,  # Very small to force truncation
            warn_callback=warn_callback,
        )

        # Should have created segments
        assert len(segments) >= 1


class TestPhonemeSegmentPauseAfter:
    """Tests for pause_after field in PhonemeSegment."""

    def test_default_pause_after(self):
        """Test that pause_after defaults to 0.0."""
        segment = PhonemeSegment(
            text="Hello",
            phonemes="həˈloʊ",
            tokens=[1, 2, 3],
        )
        assert segment.pause_after == 0.0

    def test_custom_pause_after(self):
        """Test setting custom pause_after value."""
        segment = PhonemeSegment(
            text="Hello",
            phonemes="həˈloʊ",
            tokens=[1, 2, 3],
            pause_after=1.5,
        )
        assert segment.pause_after == 1.5

    def test_pause_after_serialization(self):
        """Test that pause_after is included in to_dict()."""
        segment = PhonemeSegment(
            text="Hello",
            phonemes="həˈloʊ",
            tokens=[1, 2, 3],
            pause_after=0.5,
        )
        data = segment.to_dict()
        assert "pause_after" in data
        assert data["pause_after"] == 0.5

    def test_pause_after_deserialization(self):
        """Test that pause_after is loaded from from_dict()."""
        data = {
            "text": "Hello",
            "phonemes": "həˈloʊ",
            "tokens": [1, 2, 3],
            "lang": "en-us",
            "paragraph": 0,
            "sentence": None,
            "pause_after": 1.2,
        }
        segment = PhonemeSegment.from_dict(data)
        assert segment.pause_after == 1.2

    def test_pause_after_backward_compatibility(self):
        """Test that old format without pause_after still works."""
        data = {
            "text": "Hello",
            "phonemes": "həˈloʊ",
            "tokens": [1, 2, 3],
            "lang": "en-us",
            "paragraph": 0,
            "sentence": None,
        }
        segment = PhonemeSegment.from_dict(data)
        assert segment.pause_after == 0.0  # Default value

    def test_pause_after_round_trip(self):
        """Test serialization and deserialization preserve pause_after."""
        original = PhonemeSegment(
            text="Test",
            phonemes="tɛst",
            tokens=[4, 5, 6],
            lang="en-us",
            paragraph=1,
            sentence=2,
            pause_after=0.75,
        )
        data = original.to_dict()
        restored = PhonemeSegment.from_dict(data)

        assert restored.text == original.text
        assert restored.phonemes == original.phonemes
        assert restored.tokens == original.tokens
        assert restored.lang == original.lang
        assert restored.paragraph == original.paragraph
        assert restored.sentence == original.sentence
        assert restored.pause_after == original.pause_after
