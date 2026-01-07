"""Phoneme data structures for pykokoro.

This module provides data structures for storing and manipulating
phoneme segments for TTS generation.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import numpy as np

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


def populate_segment_pauses(
    segments: list[PhonemeSegment],
    pause_clause: float,
    pause_sentence: float,
    pause_paragraph: float,
    pause_variance: float,
    rng: np.random.Generator,
) -> list[PhonemeSegment]:
    """Populate pause_after for each PhonemeSegment based on text boundaries.

    Assigns natural pause durations between segments based on the type of boundary:
    - Paragraph boundary (different paragraph): pause_paragraph
    - Sentence boundary (same paragraph, different sentence): pause_sentence
    - Clause boundary (same sentence): pause_clause
    - Last segment: 0.0 (no pause after)

    Gaussian variance is applied to pause durations for naturalness using
    apply_pause_variance(). The function modifies segments in-place.

    Note: When sentence is None, it is treated as a distinct value for comparison,
    so segments with sentence=None in the same paragraph will be considered as
    having a sentence boundary between them if one has sentence=None and the other
    has sentence=0 (or any other integer value).

    Args:
        segments: List of PhonemeSegment instances to populate with pauses
        pause_clause: Base pause duration for clause boundaries (seconds)
        pause_sentence: Base pause duration for sentence boundaries (seconds)
        pause_paragraph: Base pause duration for paragraph boundaries (seconds)
        pause_variance: Standard deviation for Gaussian variance (seconds)
        rng: NumPy random generator for reproducible variance

    Returns:
        The same list of segments with pause_after field populated (modified in-place)
    """
    for i, segment in enumerate(segments):
        if i < len(segments) - 1:  # Not the last segment
            next_segment = segments[i + 1]

            # Determine pause type based on boundary
            if next_segment.paragraph != segment.paragraph:
                # Paragraph boundary
                base_pause = pause_paragraph
            elif next_segment.sentence != segment.sentence:
                # Sentence boundary (within same paragraph)
                base_pause = pause_sentence
            else:
                # Clause boundary (within same sentence)
                base_pause = pause_clause

            # Apply variance
            segment.pause_after = apply_pause_variance(base_pause, pause_variance, rng)
    return segments


def apply_pause_variance(
    pause_duration: float,
    variance_std: float,
    rng: np.random.Generator,
) -> float:
    """Apply Gaussian variance to pause duration.

    Args:
        pause_duration: Base pause duration in seconds
        variance_std: Standard deviation for Gaussian distribution
        rng: NumPy random generator for reproducibility

    Returns:
        Pause duration with variance applied (never negative)
    """
    if variance_std <= 0:
        return pause_duration

    variance = rng.normal(0, variance_std)
    return max(0.0, pause_duration + variance)


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


def convert_pause_segments_to_phoneme_segments(
    pause_segments: list[tuple[str, float]],
    initial_pause: float,
    tokenizer: Tokenizer,
    lang: str = "en-us",
) -> list[PhonemeSegment]:
    """Convert split_with_pauses output to PhonemeSegment list.

    Args:
        pause_segments: List of (text, pause_after) tuples from split_with_pauses
        initial_pause: Initial pause duration before first segment
        tokenizer: Tokenizer instance for phonemization
        lang: Language code

    Returns:
        List of PhonemeSegment instances with pause_after populated.
        If initial_pause > 0, the first segment will be an empty phoneme segment
        with that pause duration.
    """
    segments = []

    # Add initial pause as empty segment if present
    if initial_pause > 0:
        segments.append(
            PhonemeSegment(
                text="",
                phonemes="",
                tokens=[],
                lang=lang,
                pause_after=initial_pause,
            )
        )

    # Convert each text segment to PhonemeSegment
    for text, pause_after in pause_segments:
        # Phonemize the text (may be empty string)
        if text.strip():
            phonemes = tokenizer.phonemize(text, lang=lang)
            tokens = tokenizer.tokenize(phonemes)
        else:
            phonemes = ""
            tokens = []

        segments.append(
            PhonemeSegment(
                text=text,
                phonemes=phonemes,
                tokens=tokens,
                lang=lang,
                pause_after=pause_after,
            )
        )

    return segments


def has_pause_markers(text: str) -> bool:
    """Check if text contains pause markers: (.), (..), (...).

    Args:
        text: Input text to check

    Returns:
        True if text contains any pause markers, False otherwise

    Example:
        >>> has_pause_markers("Hello (.) World")
        True
        >>> has_pause_markers("Hello World")
        False
    """
    import re

    pattern = r"\(\.\.\.\)|\(\.\.\)|\(\.\)"
    return bool(re.search(pattern, text))


def split_text_with_pauses(
    text: str,
    pause_short: float = 0.3,
    pause_medium: float = 0.6,
    pause_long: float = 1.0,
) -> tuple[float, list[tuple[str, float]]]:
    """Split text at pause markers and return segments with pause durations.

    Detects and splits on pause markers: (.), (..), (...)
    Removes the markers from the text segments.
    Consecutive pause markers have their durations added together.

    Args:
        text: Input text with optional pause markers
        pause_short: Duration for (.) in seconds
        pause_medium: Duration for (..) in seconds
        pause_long: Duration for (...) in seconds

    Returns:
        Tuple of (initial_pause_duration, segments_list) where segments_list
        is a list of (text_segment, pause_after_seconds) tuples

    Example:
        >>> split_text_with_pauses("Hello (.) World (...) Foo")
        (0.0, [("Hello", 0.3), ("World", 1.0), ("Foo", 0.0)])

        >>> split_text_with_pauses("Start (...) (..) End")
        (0.0, [("Start", 1.6), ("End", 0.0)])  # 1.0 + 0.6 = 1.6

        >>> split_text_with_pauses("(...) Hello")
        (1.0, [("Hello", 0.0)])  # Leading pause
    """
    import re

    # Pattern to match pause markers: (.), (..), (...)
    # Use lookbehind and lookahead to split while preserving markers
    pattern = r"\(\.\.\.\)|\(\.\.\)|\(\.\)"

    # Split text and capture markers
    parts = re.split(f"({pattern})", text)

    # Process parts into segments with pauses
    segments = []
    current_text = ""
    accumulated_pause = 0.0
    initial_pause = 0.0
    found_first_text = False

    for part in parts:
        if not part:
            continue

        # Check if this is a pause marker
        if re.match(pattern, part):
            # Determine pause duration
            if part == "(...)":
                pause_duration = pause_long
            elif part == "(..)":
                pause_duration = pause_medium
            else:  # '(.)'
                pause_duration = pause_short

            # If we haven't found any text yet, this is initial pause
            if not found_first_text and not current_text.strip():
                initial_pause += pause_duration
            else:
                # Accumulate pause for current segment
                accumulated_pause += pause_duration
        else:
            # This is text
            stripped = part.strip()
            if stripped:
                found_first_text = True

                # If we have accumulated text, save it as a segment
                if current_text.strip():
                    segments.append((current_text.strip(), accumulated_pause))
                    accumulated_pause = 0.0
                    current_text = ""

                current_text = stripped
            elif current_text.strip():
                # Whitespace part after we have text, keep accumulating
                current_text += part

    # Add final segment if any
    if current_text.strip():
        segments.append((current_text.strip(), accumulated_pause))

    return initial_pause, segments


def text_to_phoneme_segments(
    text: str,
    tokenizer: Tokenizer,
    lang: str = "en-us",
    split_mode: str | None = None,
    pause_short: float = 0.3,
    pause_medium: float = 0.6,
    pause_long: float = 1.0,
    pause_clause: float = 0.3,
    pause_sentence: float = 0.6,
    pause_paragraph: float = 1.0,
    pause_variance: float = 0.05,
    trim_silence: bool = False,
    rng: np.random.Generator | None = None,
) -> list[PhonemeSegment]:
    """Convert text to list of PhonemeSegment with pauses populated.

    Unified function that handles all text-to-segment conversion:
    - Manual pause markers (automatically detected: (.), (..), (...))
    - Automatic pauses (split_mode with trim_silence)
    - Combination of both

    Pause markers in text are automatically detected and processed. If your text
    contains (.), (..), or (...), they will be removed and converted to silence
    after the preceding segment.

    Args:
        text: Input text (pause markers automatically detected)
        tokenizer: Tokenizer instance for phonemization
        lang: Language code
        split_mode: Optional split mode ('paragraph', 'sentence', 'clause')
        pause_short: Duration for short pauses  (.)
        pause_medium: Duration for medium pauses  (..)
        pause_long: Duration for long pauses ...)
        pause_clause: Duration for short pauses (clause boundaries)
        pause_sentence: Duration for medium pauses (sentence boundaries)
        pause_paragraph: Duration for long pauses (paragraph boundaries)
        pause_variance: Standard deviation for Gaussian pause variance
        trim_silence: Whether automatic pauses should be added with split_mode
        rng: NumPy random generator for reproducibility

    Returns:
        List of PhonemeSegment instances with pause_after populated

    Raises:
        ImportError: If spaCy is required but not installed

    Example:
        Basic usage with pause markers (automatically detected):

        >>> from pykokoro import Tokenizer, text_to_phoneme_segments
        >>> tokenizer = Tokenizer()
        >>> segments = text_to_phoneme_segments(
        ...     "Hello (.) World (...) End",
        ...     tokenizer=tokenizer
        ... )
        >>> # Automatically detects markers
        >>> # Returns segments with pause_after: [0.3, 1.0, 0.0]

        Automatic pauses with sentence splitting:

        >>> segments = text_to_phoneme_segments(
        ...     "First sentence. Second sentence.",
        ...     tokenizer=tokenizer,
        ...     split_mode="sentence",
        ...     trim_silence=True
        ... )
        >>> # Automatically adds pauses between sentences

        Combination of manual and automatic pauses:

        >>> segments = text_to_phoneme_segments(
        ...     "First part (...) Second sentence. Third sentence.",
        ...     tokenizer=tokenizer,
        ...     split_mode="sentence",
        ...     trim_silence=True
        ... )
        >>> # Manual pause after "First part", automatic pauses between sentences
    """
    # Create RNG if not provided
    if rng is None:
        rng = np.random.default_rng()

    # Validate spaCy requirement for sentence/clause modes
    if split_mode in ["sentence", "clause"]:
        try:
            import spacy  # noqa: F401
        except ImportError as err:
            raise ImportError(
                f"spaCy is required for split_mode='{split_mode}'. "
                "Install with: pip install spacy && "
                "python -m spacy download en_core_web_sm"
            ) from err

    # Auto-detect pause markers in text
    has_pauses = has_pause_markers(text)

    # Case 1: Text contains pause markers
    if has_pauses:
        # Parse pause markers from text
        initial_pause, pause_segments = split_text_with_pauses(
            text, pause_short, pause_medium, pause_long
        )

        # If split_mode is also enabled, process each pause segment with splitting
        if split_mode is not None:
            all_segments: list[PhonemeSegment] = []

            # Add initial pause as empty segment if present
            if initial_pause > 0:
                all_segments.append(
                    PhonemeSegment(
                        text="",
                        phonemes="",
                        tokens=[],
                        lang=lang,
                        pause_after=initial_pause,
                    )
                )

            # Process each pause-delimited segment with split_mode
            for segment_text, pause_after in pause_segments:
                if segment_text.strip():
                    # Split this segment using split_mode
                    sub_segments = split_and_phonemize_text(
                        segment_text,
                        tokenizer=tokenizer,
                        lang=lang,
                        split_mode=split_mode,
                    )

                    # Add automatic pauses between sub-segments if trim_silence
                    if trim_silence and len(sub_segments) > 0:
                        populate_segment_pauses(
                            sub_segments,
                            pause_clause,
                            pause_sentence,
                            pause_paragraph,
                            pause_variance,
                            rng,
                        )

                    # Override last sub-segment's pause with explicit pause_after
                    if sub_segments:
                        sub_segments[-1].pause_after += pause_after

                    all_segments.extend(sub_segments)
                else:
                    # Empty text segment, just add pause
                    if pause_after > 0:
                        all_segments.append(
                            PhonemeSegment(
                                text="",
                                phonemes="",
                                tokens=[],
                                lang=lang,
                                pause_after=pause_after,
                            )
                        )

            return all_segments
        else:
            # No split_mode, just convert pause segments directly
            return convert_pause_segments_to_phoneme_segments(
                pause_segments, initial_pause, tokenizer, lang
            )

    # Case 2: Automatic splitting with split_mode
    elif split_mode is not None:
        # Split text into segments
        segments = split_and_phonemize_text(
            text,
            tokenizer=tokenizer,
            lang=lang,
            split_mode=split_mode,
        )

        # Add automatic pauses if trim_silence enabled
        if trim_silence:
            populate_segment_pauses(
                segments,
                pause_clause,
                pause_sentence,
                pause_paragraph,
                pause_variance,
                rng,
            )

        return segments

    # Case 3: No pauses, no splitting - simple phonemization
    else:
        # Just phonemize the entire text as a single segment
        phonemes = tokenizer.phonemize(text, lang=lang)
        tokens = tokenizer.tokenize(phonemes)
        return [
            PhonemeSegment(
                text=text,
                phonemes=phonemes,
                tokens=tokens,
                lang=lang,
                pause_after=0.0,
            )
        ]
