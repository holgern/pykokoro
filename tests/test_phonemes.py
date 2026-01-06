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
        """Test that long text gets split with cascading modes to meet phoneme limit."""
        # Create a very long text that will exceed phoneme limit
        long_text = " ".join(["word"] * 200)
        segments = split_and_phonemize_text(long_text, tokenizer, split_mode="sentence")

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
        """Test warning callback for very long phonemes that can't be split."""
        warnings = []

        def warn_callback(msg: str):
            warnings.append(msg)

        # Create a single very long word that can't be split further
        # Even at word level, this will be too long and must be truncated
        long_word = "supercalifragilisticexpialidocious" * 50
        segments = split_and_phonemize_text(
            long_word,
            tokenizer,
            split_mode="word",  # Start at word level
            warn_callback=warn_callback,
        )

        # Should have created segments
        assert len(segments) >= 1
        # Should have warned about truncation
        # (since a single word that's too long can't be split further)
        if len(segments[0].phonemes) >= 510:
            # If phonemes were at/near the limit, a warning should have been issued
            assert len(warnings) > 0


class TestCascadingSplitModes:
    """Tests for cascading split mode behavior when phonemes are too long."""

    @pytest.fixture
    def tokenizer(self):
        """Create a tokenizer instance."""
        return create_tokenizer()

    def test_cascade_paragraph_to_sentence(self, tokenizer):
        """Test that paragraph mode cascades to sentence when segment is too long."""
        # Create a paragraph with multiple sentences, where the paragraph exceeds limit
        # but individual sentences do not
        long_paragraph = (
            "This is the first sentence that has some reasonable length. "
            "This is the second sentence with more words. "
            "This is the third sentence. "
        ) * 10  # Repeat to make it long

        segments = split_and_phonemize_text(
            long_paragraph,
            tokenizer,
            split_mode="paragraph",
            max_phoneme_length=100,  # Low limit to force cascade
        )

        # Should have created multiple segments (cascaded to sentence)
        assert len(segments) > 1
        # All segments should be within limit
        for seg in segments:
            assert len(seg.phonemes) <= 100

    def test_cascade_sentence_to_clause(self, tokenizer):
        """Test that sentence mode cascades to clause when sentence is too long."""
        # Create a long sentence with commas
        long_sentence = (
            "This is a sentence with many clauses, separated by commas, "
            "and each clause has some words, to make it longer, "
            "so it exceeds the limit, and needs to be split."
        )

        segments = split_and_phonemize_text(
            long_sentence,
            tokenizer,
            split_mode="sentence",
            max_phoneme_length=80,  # Low limit to force cascade
        )

        # Should have created multiple segments (cascaded to clause)
        assert len(segments) > 1
        # All segments should be within limit
        for seg in segments:
            assert len(seg.phonemes) <= 80

    def test_cascade_clause_to_word(self, tokenizer):
        """Test that clause mode cascades to word when clause is too long."""
        # Create a clause without commas (just words)
        long_clause = " ".join(["word"] * 50)

        segments = split_and_phonemize_text(
            long_clause,
            tokenizer,
            split_mode="clause",
            max_phoneme_length=50,  # Low limit to force cascade to word
        )

        # Should have created multiple segments (cascaded to word)
        assert len(segments) > 1
        # All segments should be within limit
        for seg in segments:
            assert len(seg.phonemes) <= 50

    def test_word_level_truncation_last_resort(self, tokenizer):
        """Test that word level truncates when a single word exceeds limit."""
        warnings = []

        def warn_callback(msg: str):
            warnings.append(msg)

        # Single very long word
        very_long_word = "a" * 1000

        segments = split_and_phonemize_text(
            very_long_word,
            tokenizer,
            split_mode="word",
            max_phoneme_length=50,
            warn_callback=warn_callback,
        )

        # Should have created one segment (truncated)
        assert len(segments) == 1
        # Should be at limit
        assert len(segments[0].phonemes) <= 50
        # Should have warned
        assert len(warnings) > 0
        assert "truncat" in warnings[0].lower()

    def test_no_unnecessary_cascade(self, tokenizer):
        """Test that text within limit doesn't cascade to finer modes."""
        # Short text that fits comfortably
        short_text = "Hello world. This is short."

        segments = split_and_phonemize_text(
            short_text,
            tokenizer,
            split_mode="paragraph",
            max_phoneme_length=510,
        )

        # Should create minimal segments (not over-split)
        # With paragraph mode, this should be 1 segment
        assert len(segments) == 1
        assert len(segments[0].phonemes) <= 510

    def test_metadata_preserved_during_cascade(self, tokenizer):
        """Test that paragraph/sentence indices are preserved when cascading."""
        # Two paragraphs, first one needs to cascade
        text = (
            "First paragraph sentence one. "
            "First paragraph sentence two.\n\nSecond paragraph."
        )

        segments = split_and_phonemize_text(
            text,
            tokenizer,
            split_mode="paragraph",
            max_phoneme_length=80,  # Force first paragraph to cascade
        )

        # Should have multiple segments
        assert len(segments) > 1

        # Check that paragraph indices are preserved
        para_indices = [seg.paragraph for seg in segments]
        # Should have segments from both paragraph 0 and paragraph 1
        assert 0 in para_indices
        assert 1 in para_indices

    def test_cascade_with_mixed_lengths(self, tokenizer):
        """Test cascade behavior with mix of short and long segments."""
        # Create text with one very long sentence and one short sentence
        text = (
            "This is a very long sentence with many many words "
            "that will exceed the phoneme limit " * 5 + "Short."
        )

        segments = split_and_phonemize_text(
            text,
            tokenizer,
            split_mode="sentence",
            max_phoneme_length=100,
        )

        # Should have multiple segments
        assert len(segments) > 1
        # All should be within limit
        for seg in segments:
            assert len(seg.phonemes) <= 100
        # At least one segment should be the short sentence
        assert any(len(seg.phonemes) < 20 for seg in segments)


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
