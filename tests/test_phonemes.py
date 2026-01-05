"""Tests for pykokoro.phonemes module."""

import tempfile
from pathlib import Path

import pytest

from pykokoro.phonemes import (
    FORMAT_VERSION,
    PhonemeBook,
    PhonemeChapter,
    PhonemeSegment,
    create_phoneme_book_from_chapters,
    phonemize_text_list,
)
from pykokoro.tokenizer import Tokenizer


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


class TestPhonemeChapter:
    """Tests for PhonemeChapter dataclass."""

    def test_create_basic(self):
        """Test basic chapter creation."""
        chapter = PhonemeChapter(title="Chapter 1")
        assert chapter.title == "Chapter 1"
        assert chapter.segments == []
        assert chapter.chapter_index == 0

    def test_add_segment(self):
        """Test adding a segment."""
        chapter = PhonemeChapter(title="Chapter 1")
        segment = PhonemeSegment(text="hello", phonemes="həˈloʊ", tokens=[1, 2, 3])
        chapter.add_segment(segment)
        assert len(chapter.segments) == 1
        assert chapter.segments[0] is segment

    def test_total_tokens(self):
        """Test total token count."""
        chapter = PhonemeChapter(title="Chapter 1")
        chapter.add_segment(PhonemeSegment(text="a", phonemes="a", tokens=[1, 2, 3]))
        chapter.add_segment(PhonemeSegment(text="b", phonemes="b", tokens=[4, 5]))
        assert chapter.total_tokens == 5

    def test_total_phonemes(self):
        """Test total phoneme count."""
        chapter = PhonemeChapter(title="Chapter 1")
        chapter.add_segment(PhonemeSegment(text="a", phonemes="abc", tokens=[1]))
        chapter.add_segment(PhonemeSegment(text="b", phonemes="de", tokens=[2]))
        assert chapter.total_phonemes == 5

    def test_to_dict(self):
        """Test conversion to dictionary."""
        chapter = PhonemeChapter(title="Chapter 1", chapter_index=5)
        chapter.add_segment(PhonemeSegment(text="hello", phonemes="həˈloʊ", tokens=[1]))
        d = chapter.to_dict()
        assert d["title"] == "Chapter 1"
        assert d["chapter_index"] == 5
        assert len(d["segments"]) == 1

    def test_from_dict(self):
        """Test creation from dictionary."""
        d = {
            "title": "Chapter 1",
            "chapter_index": 5,
            "segments": [
                {"text": "hello", "phonemes": "həˈloʊ", "tokens": [1], "lang": "en-us"}
            ],
        }
        chapter = PhonemeChapter.from_dict(d)
        assert chapter.title == "Chapter 1"
        assert chapter.chapter_index == 5
        assert len(chapter.segments) == 1

    def test_format_readable(self):
        """Test human-readable formatting."""
        chapter = PhonemeChapter(title="Chapter 1")
        chapter.add_segment(PhonemeSegment(text="hello", phonemes="həˈloʊ", tokens=[1]))
        readable = chapter.format_readable()
        assert "# Chapter 1" in readable
        assert "hello [həˈloʊ]" in readable

    def test_iter_segments(self):
        """Test segment iteration."""
        chapter = PhonemeChapter(title="Chapter 1")
        s1 = PhonemeSegment(text="a", phonemes="a", tokens=[1])
        s2 = PhonemeSegment(text="b", phonemes="b", tokens=[2])
        chapter.add_segment(s1)
        chapter.add_segment(s2)
        segments = list(chapter.iter_segments())
        assert segments == [s1, s2]


class TestPhonemeBook:
    """Tests for PhonemeBook dataclass."""

    def test_create_basic(self):
        """Test basic book creation."""
        book = PhonemeBook(title="Test Book")
        assert book.title == "Test Book"
        assert book.chapters == []
        assert book.vocab_version == "v1.0"
        assert book.lang == "en-us"

    def test_add_chapter(self):
        """Test adding a chapter."""
        book = PhonemeBook(title="Test Book")
        chapter = PhonemeChapter(title="Chapter 1")
        book.add_chapter(chapter)
        assert len(book.chapters) == 1
        assert book.chapters[0] is chapter
        assert chapter.chapter_index == 0

    def test_create_chapter(self):
        """Test creating a chapter."""
        book = PhonemeBook(title="Test Book")
        chapter = book.create_chapter("Chapter 1")
        assert len(book.chapters) == 1
        assert chapter.title == "Chapter 1"
        assert chapter.chapter_index == 0

    def test_chapter_index_auto_increment(self):
        """Test chapter indices auto-increment."""
        book = PhonemeBook(title="Test Book")
        c1 = book.create_chapter("Chapter 1")
        c2 = book.create_chapter("Chapter 2")
        assert c1.chapter_index == 0
        assert c2.chapter_index == 1

    def test_total_tokens(self):
        """Test total token count across chapters."""
        book = PhonemeBook(title="Test Book")
        c1 = book.create_chapter("Chapter 1")
        c1.add_segment(PhonemeSegment(text="a", phonemes="a", tokens=[1, 2, 3]))
        c2 = book.create_chapter("Chapter 2")
        c2.add_segment(PhonemeSegment(text="b", phonemes="b", tokens=[4, 5]))
        assert book.total_tokens == 5

    def test_total_segments(self):
        """Test total segment count across chapters."""
        book = PhonemeBook(title="Test Book")
        c1 = book.create_chapter("Chapter 1")
        c1.add_segment(PhonemeSegment(text="a", phonemes="a", tokens=[1]))
        c1.add_segment(PhonemeSegment(text="b", phonemes="b", tokens=[2]))
        c2 = book.create_chapter("Chapter 2")
        c2.add_segment(PhonemeSegment(text="c", phonemes="c", tokens=[3]))
        assert book.total_segments == 3

    def test_to_dict(self):
        """Test conversion to dictionary."""
        book = PhonemeBook(title="Test Book", metadata={"author": "Test Author"})
        c = book.create_chapter("Chapter 1")
        c.add_segment(PhonemeSegment(text="hello", phonemes="həˈloʊ", tokens=[1]))
        d = book.to_dict()
        assert d["format_version"] == FORMAT_VERSION
        assert d["title"] == "Test Book"
        assert d["vocab_version"] == "v1.0"
        assert d["lang"] == "en-us"
        assert d["metadata"]["author"] == "Test Author"
        assert d["stats"]["total_chapters"] == 1
        assert d["stats"]["total_segments"] == 1
        assert len(d["chapters"]) == 1

    def test_from_dict(self):
        """Test creation from dictionary."""
        d = {
            "title": "Test Book",
            "vocab_version": "v1.0",
            "lang": "en-gb",
            "metadata": {"author": "Test"},
            "chapters": [
                {
                    "title": "Chapter 1",
                    "chapter_index": 0,
                    "segments": [
                        {
                            "text": "hello",
                            "phonemes": "həˈloʊ",
                            "tokens": [1],
                            "lang": "en-gb",
                        }
                    ],
                }
            ],
        }
        book = PhonemeBook.from_dict(d)
        assert book.title == "Test Book"
        assert book.lang == "en-gb"
        assert len(book.chapters) == 1

    def test_save_and_load_json(self):
        """Test saving and loading JSON."""
        book = PhonemeBook(title="Test Book")
        c = book.create_chapter("Chapter 1")
        c.add_segment(PhonemeSegment(text="hello", phonemes="həˈloʊ", tokens=[1, 2, 3]))

        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "test.json"
            book.save(path)

            # Verify file exists
            assert path.exists()

            # Load and verify
            loaded = PhonemeBook.load(path)
            assert loaded.title == "Test Book"
            assert len(loaded.chapters) == 1
            assert len(loaded.chapters[0].segments) == 1
            assert loaded.chapters[0].segments[0].text == "hello"

    def test_save_readable(self):
        """Test saving human-readable format."""
        book = PhonemeBook(title="Test Book")
        c = book.create_chapter("Chapter 1")
        c.add_segment(PhonemeSegment(text="hello", phonemes="həˈloʊ", tokens=[1]))
        c.add_segment(PhonemeSegment(text="world", phonemes="wɜːld", tokens=[2]))

        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "test.txt"
            book.save_readable(path)

            # Verify file exists
            assert path.exists()

            # Read and verify content
            content = path.read_text()
            assert "# Test Book" in content
            assert "hello [həˈloʊ]" in content
            assert "world [wɜːld]" in content

    def test_iter_segments(self):
        """Test iterating over all segments with chapter index."""
        book = PhonemeBook(title="Test Book")
        c1 = book.create_chapter("Chapter 1")
        c1.add_segment(PhonemeSegment(text="a", phonemes="a", tokens=[1]))
        c2 = book.create_chapter("Chapter 2")
        c2.add_segment(PhonemeSegment(text="b", phonemes="b", tokens=[2]))

        segments = list(book.iter_segments())
        assert len(segments) == 2
        assert segments[0][0] == 0  # Chapter index
        assert segments[0][1].text == "a"
        assert segments[1][0] == 1
        assert segments[1][1].text == "b"

    def test_iter_chapters(self):
        """Test iterating over chapters."""
        book = PhonemeBook(title="Test Book")
        c1 = book.create_chapter("Chapter 1")
        c2 = book.create_chapter("Chapter 2")
        chapters = list(book.iter_chapters())
        assert chapters == [c1, c2]

    def test_get_info(self):
        """Test getting book info."""
        book = PhonemeBook(title="Test Book", metadata={"author": "Test"})
        c = book.create_chapter("Chapter 1")
        c.add_segment(PhonemeSegment(text="hello", phonemes="həˈloʊ", tokens=[1, 2, 3]))

        info = book.get_info()
        assert info["title"] == "Test Book"
        assert info["chapters"] == 1
        assert info["segments"] == 1
        assert info["tokens"] == 3
        assert info["metadata"]["author"] == "Test"


class TestPhonemeChapterWithTokenizer:
    """Tests for PhonemeChapter with real tokenizer."""

    @pytest.fixture
    def tokenizer(self):
        """Create a tokenizer instance."""
        return Tokenizer()

    def test_add_text(self, tokenizer):
        """Test adding text with tokenizer."""
        chapter = PhonemeChapter(title="Chapter 1")
        segments = chapter.add_text("hello world", tokenizer)

        assert len(segments) == 1
        segment = segments[0]
        assert segment.text == "hello world"
        assert len(segment.phonemes) > 0
        assert len(segment.tokens) > 0
        assert segment.lang == "en-us"

    def test_add_text_with_lang(self, tokenizer):
        """Test adding text with custom language."""
        chapter = PhonemeChapter(title="Chapter 1")
        segments = chapter.add_text("hello", tokenizer, lang="en-gb")
        assert len(segments) == 1
        assert segments[0].lang == "en-gb"

    def test_add_text_warns_on_long_phonemes(self, tokenizer):
        """Test that add_text warns when phonemes exceed max length."""
        chapter = PhonemeChapter(title="Chapter 1")
        warnings = []

        def warn_callback(msg):
            warnings.append(msg)

        # Create a very long text without any natural split points
        # A single long word repeated will be hard to split
        long_text = "Supercalifragilisticexpialidocious " * 5
        segments = chapter.add_text(
            long_text,
            tokenizer,
            max_chars=30,  # Small enough to trigger splitting
            max_phoneme_length=20,  # Very small limit - single words exceed this
            warn_callback=warn_callback,
        )

        # Should have created segments
        assert len(segments) > 0
        # Should have issued warnings about truncation (single words can't be split)
        assert len(warnings) > 0
        assert "truncating" in warnings[0].lower()

    def test_add_text_splits_to_avoid_long_phonemes(self, tokenizer):
        """Test that add_text splits text to avoid exceeding phoneme limit."""
        chapter = PhonemeChapter(title="Chapter 1")

        # Normal text with reasonable limits
        text = "Hello world. This is a test. How are you today?"
        segments = chapter.add_text(
            text,
            tokenizer,
            split_mode="sentence",
            max_chars=300,
        )

        # All segments should have phonemes within limit
        for seg in segments:
            assert len(seg.phonemes) <= 510


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

    def test_create_phoneme_book_from_chapters(self, tokenizer):
        """Test creating a book from chapter data."""
        chapters = [
            ("Chapter 1", ["Hello world.", "How are you?"]),
            ("Chapter 2", ["Goodbye."]),
        ]

        book = create_phoneme_book_from_chapters(
            title="Test Book",
            chapters=chapters,
            tokenizer=tokenizer,
        )

        assert book.title == "Test Book"
        assert len(book.chapters) == 2
        assert book.chapters[0].title == "Chapter 1"
        assert len(book.chapters[0].segments) == 2
        assert book.chapters[1].title == "Chapter 2"
        assert len(book.chapters[1].segments) == 1
