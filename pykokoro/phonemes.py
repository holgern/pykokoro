"""Phoneme data structures for pykokoro.

This module provides data structures for storing and manipulating
phoneme segments for TTS generation.
"""

from __future__ import annotations

import logging
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Literal

import numpy as np

if TYPE_CHECKING:
    from .tokenizer import Tokenizer

logger = logging.getLogger(__name__)


@dataclass
class PhonemeSegment:
    """A segment of text with its phoneme representation.

    Attributes:
        text: Original text
        phonemes: IPA phoneme string
        tokens: Token IDs
        lang: Language code used for phonemization
        paragraph: Paragraph index (0-based) for pause calculation
        sentence: Sentence index (int), sentence range ("0-2"), or None
        pause_after: Duration of pause after this segment in seconds
        ssmd_metadata: Optional SSMD metadata (emphasis, prosody, markers, etc.)
    """

    text: str
    phonemes: str
    tokens: list[int]
    lang: str = "en-us"
    paragraph: int = 0
    sentence: int | str | None = None
    pause_after: float = 0.0
    ssmd_metadata: dict[str, Any] | None = field(default=None, repr=False)

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        result = {
            "text": self.text,
            "phonemes": self.phonemes,
            "tokens": self.tokens,
            "lang": self.lang,
            "paragraph": self.paragraph,
            "sentence": self.sentence,
            "pause_after": self.pause_after,
        }
        if self.ssmd_metadata is not None:
            result["ssmd_metadata"] = self.ssmd_metadata
        return result

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
            ssmd_metadata=data.get("ssmd_metadata"),
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


def batch_chunks_by_optimal_length(
    chunks: list[tuple[str, int, int | None]],
    tokenizer: Tokenizer,
    lang: str,
    optimal_lengths: list[int],
    max_phoneme_length: int,
) -> list[tuple[str, int, int | str | None]]:
    """Batch small text chunks together to reach optimal phoneme length.

    Combines consecutive chunks to achieve target phoneme lengths, improving
    audio quality for very short segments. Uses a greedy conservative approach:
    - Merges chunks toward highest target in optimal_lengths
    - Stops when reaching any target AND adding next would significantly overshoot
    - Respects max_phoneme_length limit (won't batch if it would exceed 510)
    - Single chunks exceeding max are passed through (handled by cascade later)

    The sentence metadata for batched chunks uses range format: "0-2" indicates
    sentences 0, 1, and 2 were merged. Single sentences keep int format.

    Args:
        chunks: List of (text, paragraph_idx, sentence_idx) tuples from splitting
        tokenizer: Tokenizer instance for phonemization
        lang: Language code for phonemization
        optimal_lengths: Sorted list of target lengths (e.g., [30, 50, 70])
            Higher values are preferred targets
        max_phoneme_length: Hard maximum limit (510 for Kokoro)

    Returns:
        List of batched chunks: (text, paragraph_idx, sentence_range)
        where sentence_range is "start-end" string for batched segments or None

    Example:
        Input chunks (after sentence splitting):
        [("Why?", 0, 0), ("Do?", 0, 1), ("Go!", 0, 2), ("I know.", 0, 3)]

        With optimal_lengths=[50]:
        Output: [("Why? Do? Go! I know.", 0, "0-3")]

        All four short sentences batched together to reach ~50 phonemes
    """
    if not chunks or not optimal_lengths:
        # Return with compatible type annotation
        result: list[tuple[str, int, int | str | None]] = [
            (text, para, sent) for text, para, sent in chunks
        ]
        return result

    # Ensure optimal_lengths is sorted ascending
    targets = sorted(optimal_lengths)
    highest_target = targets[-1]
    overshoot_tolerance = 0.3  # 30% tolerance for overshooting

    batched_chunks = []
    current_batch_texts = []
    current_batch_phoneme_length = 0
    first_chunk_paragraph = None
    first_chunk_sentence = None
    last_chunk_sentence = None

    def flush_batch() -> None:
        """Flush current batch to output."""
        nonlocal current_batch_texts, current_batch_phoneme_length
        nonlocal first_chunk_paragraph, first_chunk_sentence, last_chunk_sentence

        if not current_batch_texts:
            return

        combined_text = " ".join(current_batch_texts)

        # Determine sentence metadata
        if first_chunk_sentence is None and last_chunk_sentence is None:
            # Both None - preserve None
            sentence_metadata: int | str | None = None
        elif first_chunk_sentence is None or last_chunk_sentence is None:
            # Mixed None and int - use None
            sentence_metadata = None
        elif first_chunk_sentence == last_chunk_sentence:
            # Single sentence - preserve as int
            sentence_metadata = first_chunk_sentence
        else:
            # Multiple sentences - use range format "start-end"
            sentence_metadata = f"{first_chunk_sentence}-{last_chunk_sentence}"

        batched_chunks.append((combined_text, first_chunk_paragraph, sentence_metadata))

        # Reset batch
        current_batch_texts = []
        current_batch_phoneme_length = 0
        first_chunk_paragraph = None
        first_chunk_sentence = None
        last_chunk_sentence = None

    for chunk_text, para_idx, sent_idx in chunks:
        chunk_text = chunk_text.strip()
        if not chunk_text:
            continue

        # Calculate phonemes for this chunk
        chunk_phonemes = tokenizer.phonemize(chunk_text, lang=lang)
        chunk_length = len(chunk_phonemes)

        # If this single chunk already exceeds max, add it as-is
        # (it will be handled by cascade logic later)
        if chunk_length > max_phoneme_length:
            # Flush any current batch first
            if current_batch_texts:
                flush_batch()
            # Add oversized chunk alone
            batched_chunks.append((chunk_text, para_idx, sent_idx))
            continue

        # If this is the first chunk in batch, start accumulating
        if not current_batch_texts:
            current_batch_texts.append(chunk_text)
            current_batch_phoneme_length = chunk_length
            first_chunk_paragraph = para_idx
            first_chunk_sentence = sent_idx
            last_chunk_sentence = sent_idx
            continue

        # Calculate what length we'd have if we add this chunk
        potential_length = current_batch_phoneme_length + chunk_length

        # Check if adding would exceed max
        if potential_length > max_phoneme_length:
            # Flush current batch and start new one
            flush_batch()
            current_batch_texts.append(chunk_text)
            current_batch_phoneme_length = chunk_length
            first_chunk_paragraph = para_idx
            first_chunk_sentence = sent_idx
            last_chunk_sentence = sent_idx
            continue

        # Conservative stopping logic:
        # If current batch is at any target AND adding next would overshoot
        # the next higher target significantly, stop merging
        current_at_target = any(current_batch_phoneme_length >= t for t in targets)

        if current_at_target:
            # Find next higher target
            next_higher_target = None
            for t in targets:
                if t > current_batch_phoneme_length:
                    next_higher_target = t
                    break

            # If no higher target, we're already above highest
            # Use highest as reference
            if next_higher_target is None:
                next_higher_target = highest_target

            # Check if adding would overshoot by more than tolerance
            overshoot_amount = potential_length - next_higher_target
            overshoot_ratio = (
                overshoot_amount / next_higher_target if next_higher_target > 0 else 0
            )

            if overshoot_ratio > overshoot_tolerance:
                # Stop merging - flush current and start new batch
                flush_batch()
                current_batch_texts.append(chunk_text)
                current_batch_phoneme_length = chunk_length
                first_chunk_paragraph = para_idx
                first_chunk_sentence = sent_idx
                last_chunk_sentence = sent_idx
                continue

        # Otherwise, add to current batch (greedy merging toward highest target)
        current_batch_texts.append(chunk_text)
        current_batch_phoneme_length = potential_length
        last_chunk_sentence = sent_idx

    # Flush final batch
    flush_batch()

    return batched_chunks


def batch_phoneme_segments(
    segments: list[PhonemeSegment],
    tokenizer: Tokenizer,
    optimal_lengths: list[int],
    max_phoneme_length: int,
) -> list[PhonemeSegment]:
    """Batch PhonemeSegment instances to reach optimal phoneme length.

    Uses conservative batching strategy with overshoot tolerance.
    Preserves SSMD metadata and pause_after from last segment in each batch.

    This function combines consecutive short segments to achieve target
    phoneme lengths, improving audio quality for very short segments.
    It uses a greedy conservative approach:
    - Merges segments toward highest target in optimal_lengths
    - Stops when reaching any target AND adding next would significantly overshoot
    - Respects max_phoneme_length limit (won't batch if it would exceed 510)
    - Single segments exceeding max are passed through (handled by cascade later)

    Args:
        segments: List of PhonemeSegment instances to batch
        tokenizer: Tokenizer instance (for re-phonemizing batched text)
        optimal_lengths: Target phoneme lengths (e.g., [30, 50, 70])
        max_phoneme_length: Maximum allowed phonemes (510 for Kokoro)

    Returns:
        List of batched PhonemeSegment instances with preserved metadata
    """
    if not segments or not optimal_lengths:
        return segments

    # Ensure optimal_lengths is sorted ascending
    targets = sorted(optimal_lengths)
    highest_target = targets[-1]
    overshoot_tolerance = 0.3  # 30% tolerance for overshooting

    batched_segments: list[PhonemeSegment] = []
    current_batch_segments: list[PhonemeSegment] = []
    current_batch_phoneme_length = 0

    def flush_batch() -> None:
        """Flush current batch to output."""
        nonlocal current_batch_segments, current_batch_phoneme_length

        if not current_batch_segments:
            return

        # Combine text from all segments
        combined_text = " ".join(seg.text for seg in current_batch_segments)

        # Re-phonemize combined text (use language from first segment)
        lang = current_batch_segments[0].lang
        combined_phonemes = tokenizer.phonemize(combined_text, lang=lang)
        combined_tokens = tokenizer.tokenize(combined_phonemes)

        # Determine paragraph/sentence metadata
        first_seg = current_batch_segments[0]
        last_seg = current_batch_segments[-1]

        if first_seg.sentence is None or last_seg.sentence is None:
            sentence_metadata: int | str | None = None
        elif first_seg.sentence == last_seg.sentence:
            sentence_metadata = first_seg.sentence
        else:
            # Multiple sentences - use range format
            sentence_metadata = f"{first_seg.sentence}-{last_seg.sentence}"

        # Create batched segment
        # Preserve pause_after and ssmd_metadata from LAST segment
        batched_seg = PhonemeSegment(
            text=combined_text,
            phonemes=combined_phonemes,
            tokens=combined_tokens,
            lang=lang,
            paragraph=first_seg.paragraph,
            sentence=sentence_metadata,
            pause_after=last_seg.pause_after,  # Preserve from last
            ssmd_metadata=last_seg.ssmd_metadata,  # Preserve from last
        )

        batched_segments.append(batched_seg)

        # Reset batch
        current_batch_segments = []
        current_batch_phoneme_length = 0

    for segment in segments:
        # Skip empty segments (but preserve them as-is)
        if not segment.text.strip():
            # Flush current batch first
            if current_batch_segments:
                flush_batch()
            # Add empty segment as-is (preserves pause-only segments)
            batched_segments.append(segment)
            continue

        segment_length = len(segment.phonemes)

        # If this single segment already exceeds max, add it as-is
        if segment_length > max_phoneme_length:
            # Flush any current batch first
            if current_batch_segments:
                flush_batch()
            # Add oversized segment alone (will be cascaded later)
            batched_segments.append(segment)
            continue

        # If this is the first segment in batch, start accumulating
        if not current_batch_segments:
            current_batch_segments.append(segment)
            current_batch_phoneme_length = segment_length
            continue

        # Calculate potential length if we add this segment
        potential_length = current_batch_phoneme_length + segment_length

        # Check if adding would exceed max
        if potential_length > max_phoneme_length:
            # Flush current batch and start new one
            flush_batch()
            current_batch_segments.append(segment)
            current_batch_phoneme_length = segment_length
            continue

        # Conservative stopping logic (same as batch_chunks_by_optimal_length)
        current_at_target = any(current_batch_phoneme_length >= t for t in targets)

        if current_at_target:
            # Find next higher target
            next_higher_target = None
            for t in targets:
                if t > current_batch_phoneme_length:
                    next_higher_target = t
                    break

            if next_higher_target is None:
                next_higher_target = highest_target

            # Check if adding would overshoot by more than tolerance
            overshoot_amount = potential_length - next_higher_target
            overshoot_ratio = (
                overshoot_amount / next_higher_target if next_higher_target > 0 else 0
            )

            if overshoot_ratio > overshoot_tolerance:
                # Stop merging - flush current and start new batch
                flush_batch()
                current_batch_segments.append(segment)
                current_batch_phoneme_length = segment_length
                continue

        # Otherwise, add to current batch (greedy merging toward highest target)
        current_batch_segments.append(segment)
        current_batch_phoneme_length = potential_length

    # Flush final batch
    flush_batch()

    return batched_segments


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
        mode: Split mode ('paragraph', 'sentence', 'clause', or 'word')
        language_model: spaCy model name (used for sentence/clause mode)
        paragraph_idx: Paragraph index to assign to segments
        sentence_idx: Sentence index to assign to segments

    Returns:
        List of tuples: (text_chunk, paragraph_idx, sentence_idx)

    Raises:
        ValueError: If an unsupported mode is provided
    """
    if mode == "word":
        # Word-level splitting: split on whitespace
        words = text.split()
        return [(word, paragraph_idx, sentence_idx) for word in words]
    elif mode == "paragraph":
        # Split on double newlines
        paragraphs = text.split("\n\n")
        return [(p.strip(), i, 0) for i, p in enumerate(paragraphs) if p.strip()]
    elif mode == "sentence":
        # Use phrasplit for sentence splitting
        from phrasplit import split_text

        segments = split_text(
            text,
            mode="sentence",
            language_model=language_model,
            apply_corrections=True,
        )

        # Convert phrasplit.Segment to our tuple format
        return [
            (seg.text.strip(), seg.paragraph, seg.sentence)
            for seg in segments
            if seg.text.strip()
        ]
    elif mode == "clause":
        # Use phrasplit for clause splitting (commas, semicolons)
        from phrasplit import split_text

        segments = split_text(
            text,
            mode="clause",
            language_model=language_model,
            apply_corrections=True,
            split_on_colon=True,
        )

        # Convert phrasplit.Segment to our tuple format
        # Keep paragraph_idx from caller, use sentence from phrasplit
        return [
            (seg.text.strip(), paragraph_idx, sentence_idx)
            for seg in segments
            if seg.text.strip()
        ]
    else:
        # Unsupported mode
        raise ValueError(
            f"Unsupported split mode: {mode}. Expected 'paragraph', 'sentence', 'clause', or 'word'."
        )


def split_and_phonemize_text(
    text: str,
    tokenizer: Tokenizer,
    lang: str = "en-us",
    split_mode: str = "sentence",
    language_model: str = "en_core_web_sm",
    max_phoneme_length: int = 510,
    optimal_phoneme_length: int | list[int] | None = None,
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
        optimal_phoneme_length: Optional target phoneme length(s) for batching.
            Merges short segments to reach optimal length for better audio quality.
            - None (default): No batching
            - int: Single target (e.g., 50)
            - list[int]: Multiple targets (e.g., [30, 50, 70])
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
        sentence_idx: int | str | None,
    ) -> list[PhonemeSegment]:
        """Process a text chunk, cascading to finer split modes if needed.

        This is the core recursive function that:
        1. Phonemizes the chunk
        2. Checks if phonemes fit within max_phoneme_length
        3. If too long, cascades to next finer split mode
        4. If already at word level, truncates and warns

        Args:
            chunk_text: Text to process
            current_mode: Current split mode
            paragraph_idx: Paragraph index
            sentence_idx: Sentence index (int), range string ("0-2"), or None
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
            # When cascading, pass sentence_idx only if it's an int
            # For range strings, use None since we're re-splitting
            cascade_sentence_idx = (
                sentence_idx if isinstance(sentence_idx, int) else None
            )

            sub_chunks = _split_text_with_mode(
                chunk_text,
                next_mode,
                language_model,
                paragraph_idx,
                cascade_sentence_idx,
            )
        except ImportError:
            # spaCy not installed - can only do word splitting
            if next_mode in ["sentence", "clause"]:
                warn(
                    f"spaCy required for '{next_mode}' mode but not installed. "
                    f"Falling back to word-level splitting."
                )
                cascade_sentence_idx = (
                    sentence_idx if isinstance(sentence_idx, int) else None
                )
                sub_chunks = _split_text_with_mode(
                    chunk_text,
                    "word",
                    language_model,
                    paragraph_idx,
                    cascade_sentence_idx,
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

        # Apply optimal length batching if specified
        if optimal_phoneme_length is not None:
            # Normalize to list
            if isinstance(optimal_phoneme_length, int):
                optimal_lengths = [optimal_phoneme_length]
            else:
                optimal_lengths = optimal_phoneme_length

            # Batch small chunks together
            initial_chunks = batch_chunks_by_optimal_length(
                initial_chunks,
                tokenizer,
                lang,
                optimal_lengths,
                max_phoneme_length,
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


# SSMD Integration: Import break parsing from ssmd_parser module
# Old pause markers (.), (..), (...) have been removed in favor of SSMD syntax
from .ssmd_parser import (  # noqa: E402
    has_ssmd_markup,
    parse_ssmd_to_segments,
    ssmd_segments_to_phoneme_segments,
)


def text_to_phoneme_segments(
    text: str,
    tokenizer: Tokenizer,
    lang: str = "en-us",
    pause_mode: Literal["tts", "manual"] = "tts",
    pause_clause: float = 0.3,
    pause_sentence: float = 0.6,
    pause_paragraph: float = 1.0,
    pause_variance: float = 0.05,
    optimal_phoneme_length: int | list[int] | None = None,
    rng: np.random.Generator | None = None,
) -> list[PhonemeSegment]:
    """Convert text to list of PhonemeSegment with pauses populated.

    Simplified unified function that:
    1. Always parses text through SSMD (with sentence detection)
    2. Optionally batches short segments to reach optimal phoneme length
    3. Handles phoneme overflow via cascade (sentence → clause → word)
    4. Applies pause handling based on pause_mode

    SSMD markup in text is automatically detected and processed. Supported features:
    - Breaks: ...n, ...w, ...c, ...s, ...p, ...500ms, ...2s
    - Emphasis: *text* (moderate), **text** (strong)
    - Prosody: +loud+, >fast>, ^high^ (stored for future processing)
    - Language: [Bonjour](fr) switches language for that segment
    - Phonemes: [tomato](ph: təˈmeɪtoʊ) uses explicit phonemes
    - Substitution: [H2O](sub: water) replaces text before phonemization
    - Markers: @name (stored in metadata)

    Args:
        text: Input text (SSMD markup automatically detected and processed)
        tokenizer: Tokenizer instance for phonemization
        lang: Default language code (can be overridden per-segment with SSMD)
        pause_mode: Pause handling mode:
            - "tts" (default): TTS generates pauses naturally at sentence boundaries.
              SSMD pauses are preserved. trim_silence=False during audio generation.
            - "manual": PyKokoro controls pauses with precision. Automatic pauses
              are added between segments. trim_silence=True during audio generation.
        pause_clause: Duration for SSMD ...c and automatic clause boundary pauses
        pause_sentence: Duration for SSMD ...s and automatic sentence boundary pauses
        pause_paragraph: Duration for SSMD ...p and automatic paragraph boundary pauses
        pause_variance: Standard deviation for Gaussian pause variance (only used
            when pause_mode="manual")
        optimal_phoneme_length: Optional target length(s) for batching segments.
            Merges short segments to reach optimal length for better audio quality.
            - None (default): No batching
            - int: Single target (e.g., 50 merges until ~50 phonemes)
            - list[int]: Multiple targets (e.g., [30, 50, 70] tries to reach 70,
              falls back to 50 or 30 as needed)
        rng: NumPy random generator for reproducibility

    Returns:
        List of PhonemeSegment instances with pause_after populated

    Example:
        Basic usage (TTS handles pauses):

        >>> from pykokoro import Tokenizer, text_to_phoneme_segments
        >>> tokenizer = Tokenizer()
        >>> segments = text_to_phoneme_segments(
        ...     "Hello. World.",
        ...     tokenizer=tokenizer
        ... )
        >>> # Sentences split, TTS generates natural pauses

        With SSMD breaks:

        >>> segments = text_to_phoneme_segments(
        ...     "Hello ...500ms World.",
        ...     tokenizer=tokenizer
        ... )
        >>> # SSMD pause preserved in segment

        Manual pause control:

        >>> segments = text_to_phoneme_segments(
        ...     "First sentence. Second sentence.",
        ...     tokenizer=tokenizer,
        ...     pause_mode="manual"
        ... )
        >>> # Automatic pauses added between sentences (with trim_silence)

        With optimal batching:

        >>> segments = text_to_phoneme_segments(
        ...     '"Why?" "Do?" "Go!"',
        ...     tokenizer=tokenizer,
        ...     optimal_phoneme_length=50
        ... )
        >>> # Short sentences batched together for better prosody
    """
    from .tokenizer import MAX_PHONEME_LENGTH

    # Create RNG if not provided
    if rng is None:
        rng = np.random.default_rng()

    # Step 1: ALWAYS parse through SSMD (with sentence detection)
    initial_pause, ssmd_segments = parse_ssmd_to_segments(
        text,
        tokenizer,
        lang=lang,
        pause_none=0.0,
        pause_weak=0.15,
        pause_clause=pause_clause,
        pause_sentence=pause_sentence,
        pause_paragraph=pause_paragraph,
    )

    # Step 2: Convert SSMD segments to phoneme segments
    phoneme_segments = ssmd_segments_to_phoneme_segments(
        ssmd_segments,
        initial_pause,
        tokenizer,
        default_lang=lang,
        paragraph=0,
        sentence_start=0,
    )

    # Step 3: PHONEME BATCHING (before cascade!)
    if optimal_phoneme_length is not None:
        # Normalize to list
        if isinstance(optimal_phoneme_length, int):
            optimal_lengths = [optimal_phoneme_length]
        else:
            optimal_lengths = list(optimal_phoneme_length)

        # Batch segments using conservative strategy
        phoneme_segments = batch_phoneme_segments(
            phoneme_segments,
            tokenizer,
            optimal_lengths,
            MAX_PHONEME_LENGTH,
        )

    # Step 4: CASCADE (handle overflow for segments exceeding max_phoneme_length)
    final_segments: list[PhonemeSegment] = []
    for segment in phoneme_segments:
        if not segment.text.strip():
            # Empty segment (just pause), preserve as-is
            final_segments.append(segment)
            continue

        if len(segment.phonemes) <= MAX_PHONEME_LENGTH:
            # Segment fits, no cascade needed
            final_segments.append(segment)
            continue

        # Segment too long - cascade from sentence level
        sub_segments = _cascade_split_segment(
            segment,
            tokenizer,
            MAX_PHONEME_LENGTH,
        )
        final_segments.extend(sub_segments)

    # Step 5: PAUSE HANDLING (based on pause_mode)
    if pause_mode == "manual":
        # Add automatic pauses between segments
        # Note: This ADDS to existing SSMD pauses in pause_after
        populate_segment_pauses(
            final_segments,
            pause_clause,
            pause_sentence,
            pause_paragraph,
            pause_variance,
            rng,
        )
    # else: pause_mode == "tts", segments keep SSMD pauses, TTS handles rest

    return final_segments


def _cascade_split_segment(
    segment: PhonemeSegment,
    tokenizer: Tokenizer,
    max_phoneme_length: int,
    language_model: str = "en_core_web_sm",
) -> list[PhonemeSegment]:
    """Split a segment using cascade logic (sentence → clause → word).

    Preserves SSMD metadata and pause on the last sub-segment.

    Args:
        segment: PhonemeSegment that exceeds max_phoneme_length
        tokenizer: Tokenizer instance
        max_phoneme_length: Maximum phonemes per segment
        language_model: spaCy model name for clause splitting

    Returns:
        List of PhonemeSegment instances, each within max_phoneme_length
    """

    def process_chunk_with_cascade(
        chunk_text: str,
        current_mode: str,
        paragraph_idx: int,
        sentence_idx: int | str | None,
    ) -> list[PhonemeSegment]:
        """Process a text chunk, cascading to finer split modes if needed."""
        chunk_text = chunk_text.strip()
        if not chunk_text:
            return []

        # Phonemize this chunk
        phonemes = tokenizer.phonemize(chunk_text, lang=segment.lang)

        # Check if phonemes fit within limit
        if len(phonemes) <= max_phoneme_length:
            # Success! Create the segment
            tokens = tokenizer.tokenize(phonemes)
            return [
                PhonemeSegment(
                    text=chunk_text,
                    phonemes=phonemes,
                    tokens=tokens,
                    lang=segment.lang,
                    paragraph=paragraph_idx,
                    sentence=sentence_idx,
                )
            ]

        # Phonemes are too long - need to cascade to finer split mode
        next_mode = _get_next_split_mode(current_mode)

        if next_mode is None:
            # Already at word level, can't split more - truncate and warn
            logger.warning(
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
                    lang=segment.lang,
                    paragraph=paragraph_idx,
                    sentence=sentence_idx,
                )
            ]

        # Cascade to next finer split mode
        try:
            cascade_sentence_idx = (
                sentence_idx if isinstance(sentence_idx, int) else None
            )

            sub_chunks = _split_text_with_mode(
                chunk_text,
                next_mode,
                language_model,
                paragraph_idx,
                cascade_sentence_idx,
            )
        except ImportError:
            # spaCy not installed - fall back to word splitting
            if next_mode == "clause":
                logger.warning(
                    f"spaCy required for clause splitting but not installed. "
                    f"Falling back to word-level splitting."
                )
                cascade_sentence_idx = (
                    sentence_idx if isinstance(sentence_idx, int) else None
                )
                sub_chunks = _split_text_with_mode(
                    chunk_text,
                    "word",
                    language_model,
                    paragraph_idx,
                    cascade_sentence_idx,
                )
            else:
                raise

        # Recursively process each sub-chunk
        results = []
        for sub_text, sub_para, sub_sent in sub_chunks:
            sub_segments = process_chunk_with_cascade(
                sub_text,
                next_mode,
                sub_para,
                sub_sent,
            )
            results.extend(sub_segments)

        return results

    # Start cascade from "sentence" level (since input comes from SSMD parsing)
    sub_segments = process_chunk_with_cascade(
        segment.text,
        "sentence",
        segment.paragraph,
        segment.sentence,
    )

    # Preserve SSMD metadata and pause on last sub-segment
    if sub_segments:
        sub_segments[-1].pause_after += segment.pause_after
        if segment.ssmd_metadata:
            sub_segments[-1].ssmd_metadata = segment.ssmd_metadata

    return sub_segments
