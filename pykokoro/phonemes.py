"""Phoneme data structures for pykokoro.

This module provides data structures for storing and manipulating
pre-tokenized book content (phonemes and tokens).
"""

from __future__ import annotations

import json
from collections.abc import Callable, Iterator
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from .tokenizer import Tokenizer

# Version of the phoneme export format
FORMAT_VERSION = "1.0"


@dataclass
class PhonemeSegment:
    """A segment of text with its phoneme representation.

    Attributes:
        text: Original text
        phonemes: IPA phoneme string
        tokens: Token IDs
        lang: Language code used for phonemization
        paragraph: Paragraph index (0-based) for pause calculation
        sentence: Sentence index within paragraph (0-based, or None)
    """

    text: str
    phonemes: str
    tokens: list[int]
    lang: str = "en-us"
    paragraph: int = 0
    sentence: int | None = None

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            "text": self.text,
            "phonemes": self.phonemes,
            "tokens": self.tokens,
            "lang": self.lang,
            "paragraph": self.paragraph,
            "sentence": self.sentence,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> PhonemeSegment:
        """Create from dictionary."""
        return cls(
            text=data["text"],
            phonemes=data["phonemes"],
            tokens=data["tokens"],
            lang=data.get("lang", "en-us"),
            paragraph=data.get("paragraph", 0),
            sentence=data.get("sentence"),
        )

    def format_readable(self) -> str:
        """Format as human-readable string: text [phonemes]."""
        return f"{self.text} [{self.phonemes}]"


@dataclass
class PhonemeChapter:
    """A chapter containing phoneme segments.

    Attributes:
        title: Chapter title
        segments: List of phoneme segments
        chapter_index: Chapter number (0-based)
    """

    title: str
    segments: list[PhonemeSegment] = field(default_factory=list)
    chapter_index: int = 0

    def add_segment(self, segment: PhonemeSegment) -> None:
        """Add a segment to the chapter."""
        self.segments.append(segment)

    def add_text(
        self,
        text: str,
        tokenizer: Tokenizer,
        lang: str = "en-us",
        split_mode: str = "sentence",
        max_chars: int = 300,
        language_model: str = "en_core_web_sm",
        max_phoneme_length: int = 510,
        warn_callback: Callable[[str], None] | None = None,
    ) -> list[PhonemeSegment]:
        """Add text by phonemizing it.

        Text is split according to split_mode before phonemization to create
        natural segment boundaries and avoid exceeding the tokenizer's maximum
        phoneme length.

        Args:
            text: Text to add
            tokenizer: Tokenizer instance
            lang: Language code for phonemization
            split_mode: How to split the text:
                - "paragraph": Split on double newlines only
                - "sentence": Split on sentence boundaries (using spaCy)
                - "clause": Split on sentences + commas for finer segments
            max_chars: Maximum characters per segment (default 300, used for
                       further splitting if segments are too long)
            language_model: spaCy language model for sentence/clause splitting
            max_phoneme_length: Maximum phoneme length (default 510, Kokoro limit)
            warn_callback: Optional callback for warnings (receives warning message)

        Returns:
            List of created PhonemeSegments
        """
        import re

        from phrasplit import split_long_lines, split_text

        # Safety filter: Remove <<CHAPTER: ...>> markers that epub2text might add
        # This provides defense-in-depth even if callers forget to filter
        text = re.sub(
            r"^\s*<<CHAPTER:[^>]*>>\s*\n*", "", text, count=1, flags=re.MULTILINE
        )

        def warn(msg: str) -> None:
            """Issue a warning."""
            if warn_callback:
                warn_callback(msg)

        def phonemize_with_split(
            chunk: str,
            current_max_chars: int,
            paragraph_idx: int = 0,
            sentence_idx: int | None = None,
        ) -> list[PhonemeSegment]:
            """Phonemize a chunk, splitting further if phonemes exceed limit."""
            chunk = chunk.strip()
            if not chunk:
                return []

            phonemes = tokenizer.phonemize(chunk, lang=lang)

            # Check if phonemes exceed limit
            if len(phonemes) > max_phoneme_length:
                # Try splitting further if we have room
                if current_max_chars > 50:
                    # Reduce max_chars and retry
                    new_max_chars = current_max_chars // 2
                    sub_chunks = split_long_lines(chunk, new_max_chars, language_model)
                    results = []
                    for sub in sub_chunks:
                        results.extend(
                            phonemize_with_split(
                                sub, new_max_chars, paragraph_idx, sentence_idx
                            )
                        )
                    return results
                else:
                    # Can't split further - warn and truncate
                    warn(
                        f"Segment phonemes too long ({len(phonemes)} > "
                        f"{max_phoneme_length}), truncating. Text: '{chunk[:50]}...'"
                    )
                    # Truncate phonemes to limit
                    phonemes = phonemes[:max_phoneme_length]

            tokens = tokenizer.tokenize(phonemes)
            return [
                PhonemeSegment(
                    text=chunk,
                    phonemes=phonemes,
                    tokens=tokens,
                    lang=lang,
                    paragraph=paragraph_idx,
                    sentence=sentence_idx,
                )
            ]

        # Use the new unified split_text function
        if split_mode in ["paragraph", "sentence", "clause"]:
            phrasplit_segments = split_text(
                text,
                mode=split_mode,
                language_model=language_model,
                apply_corrections=True,
                split_on_colon=True,
            )
        else:
            # Default: treat as single chunk with paragraph 0
            from phrasplit import Segment

            phrasplit_segments = (
                [Segment(text=text, paragraph=0, sentence=0)] if text.strip() else []
            )

        segments = []

        for phrasplit_seg in phrasplit_segments:
            chunk = phrasplit_seg.text.strip()
            if not chunk:
                continue

            # If chunk is still too long, split it further
            if len(chunk) > max_chars:
                sub_chunks = split_long_lines(chunk, max_chars, language_model)
            else:
                sub_chunks = [chunk]

            for sub_chunk in sub_chunks:
                new_segments = phonemize_with_split(
                    sub_chunk,
                    max_chars,
                    paragraph_idx=phrasplit_seg.paragraph,
                    sentence_idx=phrasplit_seg.sentence,
                )
                for seg in new_segments:
                    self.segments.append(seg)
                    segments.append(seg)

        return segments

    @property
    def total_tokens(self) -> int:
        """Total number of tokens in this chapter."""
        return sum(len(s.tokens) for s in self.segments)

    @property
    def total_phonemes(self) -> int:
        """Total number of phoneme characters in this chapter."""
        return sum(len(s.phonemes) for s in self.segments)

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            "title": self.title,
            "chapter_index": self.chapter_index,
            "segments": [s.to_dict() for s in self.segments],
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> PhonemeChapter:
        """Create from dictionary."""
        chapter = cls(
            title=data["title"],
            chapter_index=data.get("chapter_index", 0),
        )
        for seg_data in data.get("segments", []):
            chapter.segments.append(PhonemeSegment.from_dict(seg_data))
        return chapter

    def format_readable(self) -> str:
        """Format as human-readable string."""
        lines = [f"# {self.title}", ""]
        for segment in self.segments:
            lines.append(segment.format_readable())
        return "\n".join(lines)

    def iter_segments(self) -> Iterator[PhonemeSegment]:
        """Iterate over segments."""
        yield from self.segments


@dataclass
class PhonemeBook:
    """A book containing multiple chapters of phoneme data.

    Attributes:
        title: Book title
        chapters: List of chapters
        metadata: Additional metadata
        vocab_version: Vocabulary version used for tokenization
        lang: Default language code
    """

    title: str
    chapters: list[PhonemeChapter] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)
    vocab_version: str = "v1.0"
    lang: str = "en-us"

    def add_chapter(self, chapter: PhonemeChapter) -> None:
        """Add a chapter to the book."""
        chapter.chapter_index = len(self.chapters)
        self.chapters.append(chapter)

    def create_chapter(self, title: str) -> PhonemeChapter:
        """Create and add a new chapter.

        Args:
            title: Chapter title

        Returns:
            The created PhonemeChapter
        """
        chapter = PhonemeChapter(title=title, chapter_index=len(self.chapters))
        self.chapters.append(chapter)
        return chapter

    @property
    def total_tokens(self) -> int:
        """Total number of tokens in the book."""
        return sum(c.total_tokens for c in self.chapters)

    @property
    def total_phonemes(self) -> int:
        """Total number of phoneme characters in the book."""
        return sum(c.total_phonemes for c in self.chapters)

    @property
    def total_segments(self) -> int:
        """Total number of segments in the book."""
        return sum(len(c.segments) for c in self.chapters)

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            "format_version": FORMAT_VERSION,
            "title": self.title,
            "vocab_version": self.vocab_version,
            "lang": self.lang,
            "metadata": self.metadata,
            "stats": {
                "total_chapters": len(self.chapters),
                "total_segments": self.total_segments,
                "total_tokens": self.total_tokens,
                "total_phonemes": self.total_phonemes,
            },
            "chapters": [c.to_dict() for c in self.chapters],
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> PhonemeBook:
        """Create from dictionary."""
        book = cls(
            title=data["title"],
            vocab_version=data.get("vocab_version", "v1.0"),
            lang=data.get("lang", "en-us"),
            metadata=data.get("metadata", {}),
        )
        for ch_data in data.get("chapters", []):
            book.chapters.append(PhonemeChapter.from_dict(ch_data))
        return book

    def save(self, path: str | Path, indent: int = 2) -> None:
        """Save to JSON file.

        Args:
            path: Output file path
            indent: JSON indentation (use None for compact output)
        """
        path = Path(path)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(self.to_dict(), f, indent=indent, ensure_ascii=False)

    @classmethod
    def load(cls, path: str | Path) -> PhonemeBook:
        """Load from JSON file.

        Args:
            path: Input file path

        Returns:
            PhonemeBook instance
        """
        path = Path(path)
        with open(path, encoding="utf-8") as f:
            data = json.load(f)
        return cls.from_dict(data)

    def save_readable(self, path: str | Path) -> None:
        """Save as human-readable text file.

        Format: text [phonemes]

        Args:
            path: Output file path
        """
        path = Path(path)
        with open(path, "w", encoding="utf-8") as f:
            f.write(f"# {self.title}\n")
            f.write(f"# Vocabulary: {self.vocab_version}\n")
            f.write(f"# Language: {self.lang}\n\n")

            for chapter in self.chapters:
                f.write(chapter.format_readable())
                f.write("\n\n")

    def iter_segments(self) -> Iterator[tuple[int, PhonemeSegment]]:
        """Iterate over all segments with chapter index.

        Yields:
            Tuples of (chapter_index, segment)
        """
        for chapter in self.chapters:
            for segment in chapter.segments:
                yield chapter.chapter_index, segment

    def iter_chapters(self) -> Iterator[PhonemeChapter]:
        """Iterate over chapters."""
        yield from self.chapters

    def get_info(self) -> dict[str, Any]:
        """Get summary information about the book.

        Returns:
            Dictionary with book statistics
        """
        return {
            "title": self.title,
            "vocab_version": self.vocab_version,
            "lang": self.lang,
            "chapters": len(self.chapters),
            "segments": self.total_segments,
            "tokens": self.total_tokens,
            "phonemes": self.total_phonemes,
            "metadata": self.metadata,
        }


def phonemize_text_list(
    texts: list[str],
    tokenizer: Tokenizer,
    lang: str = "en-us",
) -> list[PhonemeSegment]:
    """Phonemize a list of texts.

    Args:
        texts: List of text strings
        tokenizer: Tokenizer instance
        lang: Language code

    Returns:
        List of PhonemeSegment instances
    """
    segments = []
    for text in texts:
        phonemes = tokenizer.phonemize(text, lang=lang)
        tokens = tokenizer.tokenize(phonemes)
        segments.append(
            PhonemeSegment(
                text=text,
                phonemes=phonemes,
                tokens=tokens,
                lang=lang,
            )
        )
    return segments


def create_phoneme_book_from_chapters(
    title: str,
    chapters: list[tuple[str, list[str]]],
    tokenizer: Tokenizer,
    lang: str = "en-us",
    vocab_version: str = "v1.0",
) -> PhonemeBook:
    """Create a PhonemeBook from chapter data.

    Args:
        title: Book title
        chapters: List of (chapter_title, paragraphs) tuples
        tokenizer: Tokenizer instance
        lang: Language code
        vocab_version: Vocabulary version

    Returns:
        PhonemeBook instance
    """
    book = PhonemeBook(
        title=title,
        vocab_version=vocab_version,
        lang=lang,
    )

    for chapter_title, paragraphs in chapters:
        chapter = book.create_chapter(chapter_title)
        for para in paragraphs:
            chapter.add_text(para, tokenizer, lang=lang)

    return book
