"""Phoneme data structures for pykokoro.

This module provides data structures for storing and manipulating
phoneme segments for TTS generation.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from .tokenizer import Tokenizer


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
        pause_after: Duration of pause after this segment in seconds
    """

    text: str
    phonemes: str
    tokens: list[int]
    lang: str = "en-us"
    paragraph: int = 0
    sentence: int | None = None
    pause_after: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            "text": self.text,
            "phonemes": self.phonemes,
            "tokens": self.tokens,
            "lang": self.lang,
            "paragraph": self.paragraph,
            "sentence": self.sentence,
            "pause_after": self.pause_after,
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
            pause_after=data.get("pause_after", 0.0),
        )

    def format_readable(self) -> str:
        """Format as human-readable string: text [phonemes]."""
        return f"{self.text} [{self.phonemes}]"


def split_and_phonemize_text(
    text: str,
    tokenizer: Tokenizer,
    lang: str = "en-us",
    split_mode: str = "sentence",
    max_chars: int = 300,
    language_model: str = "en_core_web_sm",
    max_phoneme_length: int = 510,
    warn_callback: Callable[[str], None] | None = None,
) -> list[PhonemeSegment]:
    """Split text and convert to phoneme segments.

    This is a standalone function that handles intelligent text splitting
    and phoneme generation for TTS processing. Text is split according to
    split_mode before phonemization to create natural segment boundaries
    and avoid exceeding the tokenizer's maximum phoneme length.

    Args:
        text: Input text to process
        tokenizer: Tokenizer instance for phonemization
        lang: Language code (e.g., "en-us")
        split_mode: How to split the text. Options: "paragraph" (double newlines),
            "sentence" (sentence boundaries with spaCy), "clause" (sentences + commas with spaCy)
        max_chars: Maximum characters per segment (default 300, used for
                   further splitting if segments are too long)
        language_model: spaCy language model for sentence/clause splitting
        max_phoneme_length: Maximum phoneme length (default 510, Kokoro limit)
        warn_callback: Optional callback for warnings (receives warning message)

    Returns:
        List of PhonemeSegments ready for audio generation
    """
    import re

    from phrasplit import split_long_lines, split_text

    # Safety filter: Remove <<CHAPTER: ...>> markers that epub2text might add
    # This provides defense-in-depth even if callers forget to filter
    text = re.sub(r"^\s*<<CHAPTER:[^>]*>>\s*\n*", "", text, count=1, flags=re.MULTILINE)

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
            segments.extend(new_segments)

    return segments


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
