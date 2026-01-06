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


def _get_next_split_mode(current_mode: str) -> str | None:
    """Get the next finer split mode for cascading when phonemes are too long.

    Cascade order: paragraph → sentence → clause → word → None (truncate)

    Args:
        current_mode: Current split mode

    Returns:
        Next finer split mode, or None if already at finest level
    """
    cascade = {
        "paragraph": "sentence",
        "sentence": "clause",
        "clause": "word",
        "word": None,  # Can't split finer than word level
    }
    return cascade.get(current_mode)


def _split_text_with_mode(
    text: str,
    mode: str,
    language_model: str,
    paragraph_idx: int = 0,
    sentence_idx: int | None = None,
) -> list[tuple[str, int, int | None]]:
    """Split text using specified mode.

    Args:
        text: Text to split
        mode: Split mode (paragraph, sentence, clause, word)
        language_model: spaCy model name for sentence/clause modes
        paragraph_idx: Paragraph index to assign to segments
        sentence_idx: Sentence index to assign to segments

    Returns:
        List of tuples: (text_chunk, paragraph_idx, sentence_idx)
    """
    if mode == "word":
        # Word-level splitting: split on whitespace
        words = text.split()
        return [(word, paragraph_idx, sentence_idx) for word in words]
    else:
        # Use phrasplit for paragraph/sentence/clause
        from phrasplit import split_text

        segments = split_text(
            text,
            mode=mode,
            language_model=language_model,
            apply_corrections=True,
            split_on_colon=True,
        )

        # Convert phrasplit.Segment to our tuple format
        return [
            (seg.text.strip(), seg.paragraph, seg.sentence)
            for seg in segments
            if seg.text.strip()
        ]


def split_and_phonemize_text(
    text: str,
    tokenizer: Tokenizer,
    lang: str = "en-us",
    split_mode: str = "sentence",
    language_model: str = "en_core_web_sm",
    max_phoneme_length: int = 510,
    warn_callback: Callable[[str], None] | None = None,
) -> list[PhonemeSegment]:
    """Split text and convert to phoneme segments.

    This function intelligently splits text to ensure all phoneme segments
    stay within max_phoneme_length. It uses a cascading approach:

    1. Split text using the specified split_mode
    2. Phonemize each chunk
    3. If phonemes exceed limit, automatically cascade to finer split mode:
       - paragraph → sentence → clause → word
    4. Only truncates as last resort (when even individual words are too long)

    Args:
        text: Input text to process
        tokenizer: Tokenizer instance for phonemization
        lang: Language code (e.g., "en-us")
        split_mode: Initial splitting strategy. Options:
            - "paragraph": Split on double newlines
            - "sentence": Split on sentence boundaries (requires spaCy)
            - "clause": Split on sentences + commas (requires spaCy)
        language_model: spaCy model name for sentence/clause splitting
        max_phoneme_length: Maximum phoneme length (default 510, Kokoro limit).
            Segments exceeding this will be automatically re-split.
        warn_callback: Optional callback for warnings (receives warning message)

    Returns:
        List of PhonemeSegments, each guaranteed to have phonemes <= max_phoneme_length
    """

    def warn(msg: str) -> None:
        """Issue a warning."""
        if warn_callback:
            warn_callback(msg)

    def process_chunk_with_cascade(
        chunk_text: str,
        current_mode: str,
        paragraph_idx: int,
        sentence_idx: int | None,
    ) -> list[PhonemeSegment]:
        """Process a text chunk, cascading to finer split modes if needed.

        This is the core recursive function that:
        1. Phonemizes the chunk
        2. Checks if phonemes fit within max_phoneme_length
        3. If too long, cascades to next finer split mode
        4. If already at word level, truncates and warns
        """
        chunk_text = chunk_text.strip()
        if not chunk_text:
            return []

        # Phonemize this chunk
        phonemes = tokenizer.phonemize(chunk_text, lang=lang)

        # Check if phonemes fit within limit
        if len(phonemes) <= max_phoneme_length:
            # Success! Create the segment
            tokens = tokenizer.tokenize(phonemes)
            return [
                PhonemeSegment(
                    text=chunk_text,
                    phonemes=phonemes,
                    tokens=tokens,
                    lang=lang,
                    paragraph=paragraph_idx,
                    sentence=sentence_idx,
                )
            ]

        # Phonemes are too long - need to cascade to finer split mode
        next_mode = _get_next_split_mode(current_mode)

        if next_mode is None:
            # Already at word level (or finer), can't split more
            # Last resort: truncate and warn
            warn(
                f"Segment phonemes ({len(phonemes)}) exceed max ({max_phoneme_length}) "
                f"even at word level. Truncating. Text: '{chunk_text[:50]}...'"
            )
            phonemes = phonemes[:max_phoneme_length]
            tokens = tokenizer.tokenize(phonemes)
            return [
                PhonemeSegment(
                    text=chunk_text,
                    phonemes=phonemes,
                    tokens=tokens,
                    lang=lang,
                    paragraph=paragraph_idx,
                    sentence=sentence_idx,
                )
            ]

        # Cascade to next finer split mode
        try:
            sub_chunks = _split_text_with_mode(
                chunk_text,
                next_mode,
                language_model,
                paragraph_idx,
                sentence_idx,
            )
        except ImportError:
            # spaCy not installed - can only do word splitting
            if next_mode in ["sentence", "clause"]:
                warn(
                    f"spaCy required for '{next_mode}' mode but not installed. "
                    f"Falling back to word-level splitting."
                )
                sub_chunks = _split_text_with_mode(
                    chunk_text,
                    "word",
                    language_model,
                    paragraph_idx,
                    sentence_idx,
                )
            else:
                raise

        # Recursively process each sub-chunk with the finer mode
        results = []
        for sub_text, sub_para, sub_sent in sub_chunks:
            sub_segments = process_chunk_with_cascade(
                sub_text,
                next_mode,  # Sub-chunks use the finer mode as their "current mode"
                sub_para,
                sub_sent,
            )
            results.extend(sub_segments)

        return results

    # Initial text splitting using the requested split_mode
    if split_mode in ["paragraph", "sentence", "clause"]:
        initial_chunks = _split_text_with_mode(
            text,
            split_mode,
            language_model,
        )
    else:
        # Default: treat as single chunk with paragraph 0
        initial_chunks = [(text, 0, 0)] if text.strip() else []

    # Process each initial chunk (with cascading if needed)
    segments = []
    for chunk_text, paragraph_idx, sentence_idx in initial_chunks:
        chunk_segments = process_chunk_with_cascade(
            chunk_text,
            split_mode,
            paragraph_idx,
            sentence_idx,
        )
        segments.extend(chunk_segments)

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
