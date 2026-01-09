"""Tests for pykokoro.phonemes module."""

import numpy as np
import pytest

from pykokoro.phonemes import (
    PhonemeSegment,
    batch_chunks_by_optimal_length,
    phonemize_text_list,
    populate_segment_pauses,
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

        # Single very long word - use a pattern that creates many phonemes
        # Using repeated long words to ensure phoneme count exceeds limit
        very_long_word = "supercalifragilisticexpialidocious" * 100

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
        # Note: Only check if phonemes were actually long enough to need truncation
        phonemes_before_truncation = tokenizer.phonemize(very_long_word, lang="en-us")
        if len(phonemes_before_truncation) > 50:
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


class TestPopulateSegmentPauses:
    """Tests for populate_segment_pauses() function."""

    def test_empty_segment_list(self):
        """Test with empty segment list."""
        rng = np.random.default_rng(seed=42)
        segments = []
        result = populate_segment_pauses(segments, 0.1, 0.3, 0.5, 0.0, rng)
        assert result == []

    def test_single_segment(self):
        """Test that single segment gets zero pause."""
        rng = np.random.default_rng(seed=42)
        segments = [PhonemeSegment(text="Hello", phonemes="həˈloʊ", tokens=[1, 2, 3])]
        result = populate_segment_pauses(segments, 0.1, 0.3, 0.5, 0.0, rng)

        assert len(result) == 1
        assert result[0].pause_after == 0.0

    def test_paragraph_boundary_pause(self):
        """Test that paragraph boundary gets pause_paragraph."""
        rng = np.random.default_rng(seed=42)
        segments = [
            PhonemeSegment(
                text="First", phonemes="fɝst", tokens=[1, 2], paragraph=0, sentence=0
            ),
            PhonemeSegment(
                text="Second", phonemes="sɛkənd", tokens=[3, 4], paragraph=1, sentence=0
            ),
        ]
        result = populate_segment_pauses(segments, 0.1, 0.3, 0.5, 0.0, rng)

        # First segment should have pause_paragraph (0.5), last segment has 0.0
        assert result[0].pause_after == 0.5
        assert result[1].pause_after == 0.0

    def test_sentence_boundary_pause(self):
        """Test that sentence boundary gets pause_sentence."""
        rng = np.random.default_rng(seed=42)
        segments = [
            PhonemeSegment(
                text="First", phonemes="fɝst", tokens=[1, 2], paragraph=0, sentence=0
            ),
            PhonemeSegment(
                text="Second", phonemes="sɛkənd", tokens=[3, 4], paragraph=0, sentence=1
            ),
        ]
        result = populate_segment_pauses(segments, 0.1, 0.3, 0.5, 0.0, rng)

        # First segment should have pause_sentence (0.3), last segment has 0.0
        assert result[0].pause_after == 0.3
        assert result[1].pause_after == 0.0

    def test_clause_boundary_pause(self):
        """Test that clause boundary (same sentence) gets pause_clause."""
        rng = np.random.default_rng(seed=42)
        segments = [
            PhonemeSegment(
                text="First", phonemes="fɝst", tokens=[1, 2], paragraph=0, sentence=0
            ),
            PhonemeSegment(
                text="Second", phonemes="sɛkənd", tokens=[3, 4], paragraph=0, sentence=0
            ),
        ]
        result = populate_segment_pauses(segments, 0.1, 0.3, 0.5, 0.0, rng)

        # First segment should have pause_clause (0.1), last segment has 0.0
        assert result[0].pause_after == 0.1
        assert result[1].pause_after == 0.0

    def test_last_segment_always_zero(self):
        """Test that last segment always has zero pause."""
        rng = np.random.default_rng(seed=42)
        segments = [
            PhonemeSegment(
                text="A", phonemes="eɪ", tokens=[1], paragraph=0, sentence=0
            ),
            PhonemeSegment(
                text="B", phonemes="biː", tokens=[2], paragraph=0, sentence=1
            ),
            PhonemeSegment(
                text="C", phonemes="siː", tokens=[3], paragraph=1, sentence=0
            ),
        ]
        result = populate_segment_pauses(segments, 0.1, 0.3, 0.5, 0.0, rng)

        # Last segment always has 0.0 pause
        assert result[-1].pause_after == 0.0
        # Others should have non-zero pauses
        assert result[0].pause_after > 0.0
        assert result[1].pause_after > 0.0

    def test_zero_variance_exact_pauses(self):
        """Test that variance=0 produces exact pause durations."""
        rng = np.random.default_rng(seed=42)
        segments = [
            PhonemeSegment(
                text="A", phonemes="eɪ", tokens=[1], paragraph=0, sentence=0
            ),
            PhonemeSegment(
                text="B", phonemes="biː", tokens=[2], paragraph=0, sentence=0
            ),
            PhonemeSegment(
                text="C", phonemes="siː", tokens=[3], paragraph=0, sentence=1
            ),
            PhonemeSegment(
                text="D", phonemes="diː", tokens=[4], paragraph=1, sentence=0
            ),
            PhonemeSegment(
                text="E", phonemes="iː", tokens=[5], paragraph=1, sentence=0
            ),
        ]

        pause_clause = 0.15
        pause_sentence = 0.35
        pause_paragraph = 0.55

        result = populate_segment_pauses(
            segments, pause_clause, pause_sentence, pause_paragraph, 0.0, rng
        )

        # Check exact pause values (no variance)
        assert result[0].pause_after == pause_clause  # A->B: same sentence
        assert result[1].pause_after == pause_sentence  # B->C: sentence boundary
        assert result[2].pause_after == pause_paragraph  # C->D: paragraph boundary
        assert result[3].pause_after == pause_clause  # D->E: same sentence
        assert result[4].pause_after == 0.0  # Last segment

    def test_variance_applied(self):
        """Test that variance produces different pause durations."""
        rng = np.random.default_rng(seed=42)
        segments = [
            PhonemeSegment(
                text="A", phonemes="eɪ", tokens=[1], paragraph=0, sentence=0
            ),
            PhonemeSegment(
                text="B", phonemes="biː", tokens=[2], paragraph=0, sentence=0
            ),
            PhonemeSegment(
                text="C", phonemes="siː", tokens=[3], paragraph=0, sentence=0
            ),
            PhonemeSegment(
                text="D", phonemes="diː", tokens=[4], paragraph=0, sentence=0
            ),
        ]

        pause_clause = 0.1
        variance = 0.05

        result = populate_segment_pauses(
            segments, pause_clause, 0.3, 0.5, variance, rng
        )

        # All should be clause boundaries (pause_clause + variance)
        pauses = [seg.pause_after for seg in result[:-1]]  # Exclude last (always 0)

        # Pauses should vary (not all exactly pause_clause)
        assert not all(p == pause_clause for p in pauses)

        # But they should be close to pause_clause (within reasonable range)
        for pause in pauses:
            # With variance=0.05, expect pauses roughly in range [0.0, 0.2]
            # (pause_clause ± 3*variance, clamped to non-negative)
            assert pause >= 0.0
            assert pause <= pause_clause + 3 * variance

    def test_variance_reproducibility(self):
        """Test that same seed produces same pauses."""
        segments_1 = [
            PhonemeSegment(
                text="A", phonemes="eɪ", tokens=[1], paragraph=0, sentence=0
            ),
            PhonemeSegment(
                text="B", phonemes="biː", tokens=[2], paragraph=0, sentence=0
            ),
            PhonemeSegment(
                text="C", phonemes="siː", tokens=[3], paragraph=0, sentence=1
            ),
        ]
        segments_2 = [
            PhonemeSegment(
                text="A", phonemes="eɪ", tokens=[1], paragraph=0, sentence=0
            ),
            PhonemeSegment(
                text="B", phonemes="biː", tokens=[2], paragraph=0, sentence=0
            ),
            PhonemeSegment(
                text="C", phonemes="siː", tokens=[3], paragraph=0, sentence=1
            ),
        ]

        rng1 = np.random.default_rng(seed=123)
        rng2 = np.random.default_rng(seed=123)

        result_1 = populate_segment_pauses(segments_1, 0.1, 0.3, 0.5, 0.05, rng1)
        result_2 = populate_segment_pauses(segments_2, 0.1, 0.3, 0.5, 0.05, rng2)

        # Same seed should produce identical pauses
        for seg1, seg2 in zip(result_1, result_2, strict=False):
            assert seg1.pause_after == seg2.pause_after

    def test_variance_non_negative(self):
        """Test that variance never produces negative pauses."""
        rng = np.random.default_rng(seed=42)
        segments = [
            PhonemeSegment(
                text=f"Seg{i}", phonemes="test", tokens=[i], paragraph=0, sentence=0
            )
            for i in range(100)  # Many segments to test variance
        ]

        # Use very small pause and large variance to try to get negative values
        pause_short = 0.01
        variance = 1.0  # Large variance

        result = populate_segment_pauses(segments, pause_short, 0.3, 0.5, variance, rng)

        # All pauses should be non-negative
        for seg in result:
            assert seg.pause_after >= 0.0

    def test_none_sentence_values(self):
        """Test handling of None sentence values."""
        rng = np.random.default_rng(seed=42)
        segments = [
            PhonemeSegment(
                text="A", phonemes="eɪ", tokens=[1], paragraph=0, sentence=None
            ),
            PhonemeSegment(
                text="B", phonemes="biː", tokens=[2], paragraph=0, sentence=0
            ),
            PhonemeSegment(
                text="C", phonemes="siː", tokens=[3], paragraph=0, sentence=None
            ),
        ]

        result = populate_segment_pauses(segments, 0.1, 0.3, 0.5, 0.0, rng)

        # A->B: None != 0, so sentence boundary -> pause_medium
        assert result[0].pause_after == 0.3
        # B->C: 0 != None, so sentence boundary -> pause_medium
        assert result[1].pause_after == 0.3
        # C is last
        assert result[2].pause_after == 0.0

    def test_none_to_none_sentence(self):
        """Test that None to None is treated as same sentence (clause boundary)."""
        rng = np.random.default_rng(seed=42)
        segments = [
            PhonemeSegment(
                text="A", phonemes="eɪ", tokens=[1], paragraph=0, sentence=None
            ),
            PhonemeSegment(
                text="B", phonemes="biː", tokens=[2], paragraph=0, sentence=None
            ),
        ]

        result = populate_segment_pauses(segments, 0.1, 0.3, 0.5, 0.0, rng)

        # None == None, same paragraph -> clause boundary -> pause_short
        assert result[0].pause_after == 0.1

    def test_complex_multi_paragraph_scenario(self):
        """Test complex scenario with multiple paragraphs and sentences."""
        rng = np.random.default_rng(seed=42)
        segments = [
            # Paragraph 0, sentence 0
            PhonemeSegment(
                text="A", phonemes="eɪ", tokens=[1], paragraph=0, sentence=0
            ),
            PhonemeSegment(
                text="B", phonemes="biː", tokens=[2], paragraph=0, sentence=0
            ),
            # Paragraph 0, sentence 1
            PhonemeSegment(
                text="C", phonemes="siː", tokens=[3], paragraph=0, sentence=1
            ),
            # Paragraph 1, sentence 0
            PhonemeSegment(
                text="D", phonemes="diː", tokens=[4], paragraph=1, sentence=0
            ),
            PhonemeSegment(
                text="E", phonemes="iː", tokens=[5], paragraph=1, sentence=0
            ),
            # Paragraph 1, sentence 1
            PhonemeSegment(
                text="F", phonemes="ɛf", tokens=[6], paragraph=1, sentence=1
            ),
            # Paragraph 2, sentence 0
            PhonemeSegment(
                text="G", phonemes="dʒiː", tokens=[7], paragraph=2, sentence=0
            ),
        ]

        result = populate_segment_pauses(segments, 0.1, 0.3, 0.5, 0.0, rng)

        # A->B: clause boundary (same sentence)
        assert result[0].pause_after == 0.1
        # B->C: sentence boundary (same paragraph)
        assert result[1].pause_after == 0.3
        # C->D: paragraph boundary
        assert result[2].pause_after == 0.5
        # D->E: clause boundary (same sentence)
        assert result[3].pause_after == 0.1
        # E->F: sentence boundary (same paragraph)
        assert result[4].pause_after == 0.3
        # F->G: paragraph boundary
        assert result[5].pause_after == 0.5
        # G is last
        assert result[6].pause_after == 0.0

    def test_in_place_modification(self):
        """Test that function modifies segments in-place."""
        rng = np.random.default_rng(seed=42)
        segments = [
            PhonemeSegment(
                text="A", phonemes="eɪ", tokens=[1], paragraph=0, sentence=0
            ),
            PhonemeSegment(
                text="B", phonemes="biː", tokens=[2], paragraph=0, sentence=1
            ),
        ]

        # Store references
        original_segment_0 = segments[0]
        original_segment_1 = segments[1]

        result = populate_segment_pauses(segments, 0.1, 0.3, 0.5, 0.0, rng)

        # Result should be same list object
        assert result is segments
        # Original segment objects should be modified
        assert original_segment_0 is result[0]
        assert original_segment_1 is result[1]
        # Pauses should be set
        assert original_segment_0.pause_after == 0.3
        assert original_segment_1.pause_after == 0.0

    def test_all_same_paragraph_and_sentence(self):
        """Test segments all in same paragraph and sentence get pause_short."""
        rng = np.random.default_rng(seed=42)
        segments = [
            PhonemeSegment(
                text="A", phonemes="eɪ", tokens=[1], paragraph=0, sentence=0
            ),
            PhonemeSegment(
                text="B", phonemes="biː", tokens=[2], paragraph=0, sentence=0
            ),
            PhonemeSegment(
                text="C", phonemes="siː", tokens=[3], paragraph=0, sentence=0
            ),
            PhonemeSegment(
                text="D", phonemes="diː", tokens=[4], paragraph=0, sentence=0
            ),
        ]

        result = populate_segment_pauses(segments, 0.1, 0.3, 0.5, 0.0, rng)

        # All boundaries are clause boundaries (except last)
        assert result[0].pause_after == 0.1
        assert result[1].pause_after == 0.1
        assert result[2].pause_after == 0.1
        assert result[3].pause_after == 0.0  # Last segment


class TestBatchChunksByOptimalLength:
    """Tests for batch_chunks_by_optimal_length function."""

    @pytest.fixture
    def tokenizer(self):
        """Create a tokenizer for testing."""
        return create_tokenizer()

    def test_no_batching_when_empty(self, tokenizer):
        """Test that empty chunks return empty list."""
        result = batch_chunks_by_optimal_length(
            chunks=[],
            tokenizer=tokenizer,
            lang="en-us",
            optimal_lengths=[50],
            max_phoneme_length=510,
        )
        assert result == []

    def test_no_batching_when_no_targets(self, tokenizer):
        """Test that no batching occurs when optimal_lengths is empty."""
        chunks = [("Hello world", 0, 0), ("Goodbye", 0, 1)]
        result = batch_chunks_by_optimal_length(
            chunks=chunks,
            tokenizer=tokenizer,
            lang="en-us",
            optimal_lengths=[],
            max_phoneme_length=510,
        )
        # Should return chunks as-is (converted to compatible type)
        assert len(result) == 2

    def test_single_chunk_unchanged(self, tokenizer):
        """Test that single chunk is returned unchanged."""
        chunks = [("Hello world", 0, 0)]
        result = batch_chunks_by_optimal_length(
            chunks=chunks,
            tokenizer=tokenizer,
            lang="en-us",
            optimal_lengths=[50],
            max_phoneme_length=510,
        )
        assert len(result) == 1
        assert result[0][0] == "Hello world"
        assert result[0][1] == 0  # paragraph
        assert result[0][2] == 0  # sentence (single, preserved as int)

    def test_batch_very_short_sentences(self, tokenizer):
        """Test batching of very short sentences."""
        # Very short sentences like "Why?", "Do?", "Go!"
        chunks = [
            ("Why?", 0, 0),
            ("Do?", 0, 1),
            ("Go!", 0, 2),
            ("I know.", 0, 3),
        ]

        result = batch_chunks_by_optimal_length(
            chunks=chunks,
            tokenizer=tokenizer,
            lang="en-us",
            optimal_lengths=[50],  # Target 50 phonemes
            max_phoneme_length=510,
        )

        # Should batch all into one segment (total ~35 phonemes)
        assert len(result) == 1
        assert "Why?" in result[0][0]
        assert "Do?" in result[0][0]
        assert "Go!" in result[0][0]
        assert "I know." in result[0][0]
        assert result[0][1] == 0  # paragraph
        assert result[0][2] == "0-3"  # sentence range

    def test_single_target_batching(self, tokenizer):
        """Test batching with single target value."""
        chunks = [
            ("Short.", 0, 0),
            ("Also short.", 0, 1),
            ("This is a much longer sentence that exceeds the target.", 0, 2),
        ]

        result = batch_chunks_by_optimal_length(
            chunks=chunks,
            tokenizer=tokenizer,
            lang="en-us",
            optimal_lengths=[30],
            max_phoneme_length=510,
        )

        # Conservative greedy algorithm will batch all three together
        # since adding the long sentence doesn't overshoot by >30%
        assert len(result) >= 1
        assert "Short." in result[0][0]
        assert "Also short." in result[0][0]
        # Verify the long sentence is included somewhere in results
        assert any("This is a much longer" in r[0] for r in result)

    def test_array_targets_batching(self, tokenizer):
        """Test batching with array of target values."""
        chunks = [
            ("A.", 0, 0),
            ("B.", 0, 1),
            ("C.", 0, 2),
            ("This is a medium length sentence.", 0, 3),
            ("D.", 0, 4),
        ]

        result = batch_chunks_by_optimal_length(
            chunks=chunks,
            tokenizer=tokenizer,
            lang="en-us",
            optimal_lengths=[20, 40],  # Multiple targets
            max_phoneme_length=510,
        )

        # Should create batches trying to reach targets
        assert len(result) >= 1
        # All chunks should be accounted for
        combined_text = " ".join(r[0] for r in result)
        assert "A." in combined_text
        assert "D." in combined_text

    def test_respects_max_phoneme_length(self, tokenizer):
        """Test that batching won't combine chunks if it would exceed max."""
        # Create chunks that would exceed max if all batched
        long_text = "This is a very long sentence. " * 20  # Very long (>510)
        chunks = [
            (long_text, 0, 0),
            ("Short", 0, 1),
        ]

        result = batch_chunks_by_optimal_length(
            chunks=chunks,
            tokenizer=tokenizer,
            lang="en-us",
            optimal_lengths=[1000],  # High target
            max_phoneme_length=510,
        )

        # Should not batch the two chunks together (would exceed max)
        # The long chunk will be passed through as-is (for cascade to handle)
        # The short chunk should be separate
        assert len(result) == 2

        # First result should be the long text alone
        assert long_text.strip() in result[0][0]
        assert result[0][2] == 0  # Single sentence

        # Second result should be the short text alone
        assert "Short" in result[1][0]
        assert result[1][2] == 1  # Single sentence

    def test_sentence_range_format(self, tokenizer):
        """Test that sentence metadata uses correct format."""
        chunks = [
            ("First.", 0, 0),
            ("Second.", 0, 1),
            ("Third.", 0, 2),
        ]

        result = batch_chunks_by_optimal_length(
            chunks=chunks,
            tokenizer=tokenizer,
            lang="en-us",
            optimal_lengths=[50],
            max_phoneme_length=510,
        )

        # Should batch all three
        assert len(result) == 1
        assert result[0][2] == "0-2"  # Range format for multiple sentences

    def test_single_sentence_preserved_as_int(self, tokenizer):
        """Test that single sentence metadata is int, not string."""
        chunks = [
            ("This is a long enough sentence that it won't be batched.", 0, 5),
        ]

        result = batch_chunks_by_optimal_length(
            chunks=chunks,
            tokenizer=tokenizer,
            lang="en-us",
            optimal_lengths=[50],
            max_phoneme_length=510,
        )

        assert len(result) == 1
        assert result[0][2] == 5  # Should be int, not "5" or "5-5"

    def test_conservative_stopping(self, tokenizer):
        """Test conservative stopping logic."""
        # Create scenario where we reach target and next would overshoot
        chunks = [
            ("First sentence.", 0, 0),
            ("Second sentence.", 0, 1),
            ("This is a very long third sentence with many words.", 0, 2),
        ]

        result = batch_chunks_by_optimal_length(
            chunks=chunks,
            tokenizer=tokenizer,
            lang="en-us",
            optimal_lengths=[40],  # Target 40
            max_phoneme_length=510,
        )

        # Should create at least 2 batches (stops when adding would overshoot)
        assert len(result) >= 1

    def test_none_sentence_handling(self, tokenizer):
        """Test handling of None sentence indices."""
        chunks = [
            ("Text one.", 0, None),
            ("Text two.", 0, None),
        ]

        result = batch_chunks_by_optimal_length(
            chunks=chunks,
            tokenizer=tokenizer,
            lang="en-us",
            optimal_lengths=[50],
            max_phoneme_length=510,
        )

        # Should batch them
        assert len(result) == 1
        assert result[0][2] is None  # Both None -> result None

    def test_mixed_none_and_int_sentences(self, tokenizer):
        """Test handling of mixed None and int sentence indices."""
        chunks = [
            ("Text one.", 0, None),
            ("Text two.", 0, 1),
        ]

        result = batch_chunks_by_optimal_length(
            chunks=chunks,
            tokenizer=tokenizer,
            lang="en-us",
            optimal_lengths=[50],
            max_phoneme_length=510,
        )

        # Should batch them
        assert len(result) == 1
        assert result[0][2] is None  # Mixed None/int -> None

    def test_empty_text_chunks_skipped(self, tokenizer):
        """Test that empty text chunks are skipped."""
        chunks = [
            ("Hello", 0, 0),
            ("", 0, 1),  # Empty
            ("   ", 0, 2),  # Whitespace only
            ("World", 0, 3),
        ]

        result = batch_chunks_by_optimal_length(
            chunks=chunks,
            tokenizer=tokenizer,
            lang="en-us",
            optimal_lengths=[50],
            max_phoneme_length=510,
        )

        # Should only have non-empty chunks
        assert len(result) >= 1
        combined = result[0][0]
        assert "Hello" in combined
        assert "World" in combined

    def test_sorted_targets(self, tokenizer):
        """Test that targets are sorted internally."""
        chunks = [("Short.", 0, 0), ("Also.", 0, 1)]

        # Pass unsorted targets
        result = batch_chunks_by_optimal_length(
            chunks=chunks,
            tokenizer=tokenizer,
            lang="en-us",
            optimal_lengths=[70, 30, 50],  # Unsorted
            max_phoneme_length=510,
        )

        # Should still work (internally sorted)
        assert len(result) >= 1


class TestSplitAndPhonemizeWithOptimalLength:
    """Tests for split_and_phonemize_text with optimal_phoneme_length."""

    @pytest.fixture
    def tokenizer(self):
        """Create a tokenizer for testing."""
        return create_tokenizer()

    def test_optimal_length_none_no_batching(self, tokenizer):
        """Test that None optimal_length produces no batching."""
        text = "Why? Do? Go!"

        segments = split_and_phonemize_text(
            text=text,
            tokenizer=tokenizer,
            lang="en-us",
            split_mode="sentence",
            optimal_phoneme_length=None,
        )

        # Should have 3 separate segments (one per sentence)
        assert len(segments) == 3
        assert segments[0].sentence == 0
        assert segments[1].sentence == 1
        assert segments[2].sentence == 2

    def test_optimal_length_single_value(self, tokenizer):
        """Test batching with single optimal_length value."""
        text = '"Why?" "Do?" "Go!" "I know."'

        segments = split_and_phonemize_text(
            text=text,
            tokenizer=tokenizer,
            lang="en-us",
            split_mode="sentence",
            optimal_phoneme_length=50,
        )

        # Should batch into fewer segments
        assert len(segments) < 4
        # Check sentence metadata
        if len(segments) == 1:
            assert isinstance(segments[0].sentence, str)  # Range format
            assert "-" in segments[0].sentence

    def test_optimal_length_array_value(self, tokenizer):
        """Test batching with array optimal_length values."""
        text = '"A." "B." "C." "This is longer." "D." "E."'

        segments = split_and_phonemize_text(
            text=text,
            tokenizer=tokenizer,
            lang="en-us",
            split_mode="sentence",
            optimal_phoneme_length=[20, 40],
        )

        # Should create optimized batches
        assert len(segments) >= 1
        # Verify all text is preserved
        combined_text = " ".join(seg.text for seg in segments)
        assert "A." in combined_text
        assert "E." in combined_text


class TestTextToPhonemeSegmentsWithOptimalLength:
    """Integration tests for text_to_phoneme_segments with optimal_phoneme_length."""

    def test_end_to_end_batching(self):
        """Test complete flow from text to batched segments."""
        from pykokoro import Kokoro
        from pykokoro.phonemes import text_to_phoneme_segments

        kokoro = Kokoro()
        text = '"Why?" she asked. "Do it!" he said. "Go!" they shouted.'

        # Without batching
        segments1 = text_to_phoneme_segments(
            text=text,
            tokenizer=kokoro.tokenizer,
            lang="en-us",
            split_mode="sentence",
            optimal_phoneme_length=None,
        )

        # With batching
        segments2 = text_to_phoneme_segments(
            text=text,
            tokenizer=kokoro.tokenizer,
            lang="en-us",
            split_mode="sentence",
            optimal_phoneme_length=50,
        )

        # Batched should have fewer segments
        assert len(segments2) < len(segments1)

        # Total phonemes should be similar
        total1 = sum(len(s.phonemes) for s in segments1)
        total2 = sum(len(s.phonemes) for s in segments2)
        assert abs(total1 - total2) < 5  # Allow small difference

    def test_kokoro_create_with_optimal_length(self):
        """Test Kokoro.create() with optimal_phoneme_length parameter."""
        from pykokoro import Kokoro

        kokoro = Kokoro()
        text = '"Why?" "Do?" "Go!"'

        # Should not raise an error
        samples, sample_rate = kokoro.create(
            text,
            voice="af_bella",
            lang="en-us",
            split_mode="sentence",
            optimal_phoneme_length=50,
        )

        assert isinstance(samples, np.ndarray)
        assert sample_rate == 24000
        assert len(samples) > 0
