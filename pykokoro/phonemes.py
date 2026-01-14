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
        pause_before: Duration of pause before this segment in seconds
        pause_after: Duration of pause after this segment in seconds
        ssmd_metadata: Optional SSMD metadata (emphasis, prosody, markers, etc.)
    """

    text: str
    phonemes: str
    tokens: list[int]
    lang: str = "en-us"
    paragraph: int = 0
    sentence: int | str | None = None
    pause_before: float = 0.0
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
            "pause_before": self.pause_before,
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
            pause_before=data.get("pause_before", 0.0),
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
            f"Unsupported split mode: {mode}. "
            f"Expected 'paragraph', 'sentence', 'clause', or 'word'."
        )


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

    Note: Short sentences are handled automatically during audio generation using
    the repeat-and-cut technique (see short_sentence_handler.py), which produces
    higher quality audio than segment batching.

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
    rng: np.random.Generator | None = None,
) -> list[PhonemeSegment]:
    """Convert text to list of PhonemeSegment with pauses populated.

    Simplified unified function that:
    1. Always parses text through SSMD (with sentence detection)
    2. Handles phoneme overflow via cascade (sentence → clause → word)
    3. Applies pause handling based on pause_mode:
       - "tts": Merges sentences into paragraphs for natural TTS pauses
       - "manual": Keeps sentence-level segmentation for precise control

    Short sentences are handled automatically during audio generation using
    the repeat-and-cut technique (see short_sentence_handler.py), which
    produces higher quality audio than segment batching.

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
            - "tts" (default): Segments are merged into paragraphs so TTS can
              generate natural pauses. SSMD explicit pauses are preserved as
              segment boundaries. Best for natural speech.
            - "manual": Keeps sentence-level segmentation. PyKokoro controls
              pauses with precision. Automatic pauses are added between segments.
              trim_silence=True during audio generation. Best for precise timing.
        pause_clause: Duration for SSMD ...c and automatic clause boundary pauses
        pause_sentence: Duration for SSMD ...s and automatic sentence boundary pauses
        pause_paragraph: Duration for SSMD ...p and automatic paragraph boundary pauses
        pause_variance: Standard deviation for Gaussian pause variance (only used
            when pause_mode="manual")
        rng: NumPy random generator for reproducibility

    Returns:
        List of PhonemeSegment instances with pause_after populated

    Example:
        Basic usage (TTS handles pauses - paragraphs given to TTS):

        >>> from pykokoro import Tokenizer, text_to_phoneme_segments
        >>> tokenizer = Tokenizer()
        >>> segments = text_to_phoneme_segments(
        ...     "Hello. World. How are you?",
        ...     tokenizer=tokenizer
        ... )
        >>> # All sentences merged into one segment, TTS generates natural pauses

        With SSMD breaks (creates segment boundaries):

        >>> segments = text_to_phoneme_segments(
        ...     "Hello ...500ms World.",
        ...     tokenizer=tokenizer
        ... )
        >>> # SSMD pause creates a segment boundary

        Manual pause control (sentence-level segmentation):

        >>> segments = text_to_phoneme_segments(
        ...     "First sentence. Second sentence.",
        ...     tokenizer=tokenizer,
        ...     pause_mode="manual"
        ... )
        >>> # Each sentence is a separate segment with automatic pauses

        Short sentences (like "Why?" or "Go!") are automatically handled
        by the repeat-and-cut technique during audio generation for
        improved prosody.
    """
    from .constants import MAX_PHONEME_LENGTH

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

    # Step 3: For TTS mode, merge segments into larger chunks (paragraphs)
    # This allows the TTS model to generate natural pauses between sentences
    if pause_mode == "tts":
        phoneme_segments = _merge_segments_for_tts(
            phoneme_segments,
            tokenizer,
            MAX_PHONEME_LENGTH,
        )

    # Step 4: CASCADE (handle overflow for segments exceeding max_phoneme_length)
    # Note: Short sentences are NOT batched here - they are handled during audio
    # generation by the repeat-and-cut technique in AudioGenerator, which produces
    # higher quality audio.
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


def _merge_segments_for_tts(  # noqa: C901
    segments: list[PhonemeSegment],
    tokenizer: Tokenizer,
    max_phoneme_length: int,
) -> list[PhonemeSegment]:
    """Merge consecutive segments into larger chunks for TTS mode.

    This allows the TTS model to receive paragraph-level text, enabling it
    to generate natural pauses between sentences. Segments are merged up to
    max_phoneme_length, respecting:
    - Paragraph boundaries (different paragraph index)
    - Explicit SSMD pauses (segments with pause_after > 0)
    - Language changes (different lang attribute)
    - SSMD metadata that shouldn't be merged (prosody, voice changes)

    Args:
        segments: List of PhonemeSegment instances (sentence-level)
        tokenizer: Tokenizer for re-phonemizing merged text
        max_phoneme_length: Maximum phoneme length per merged segment

    Returns:
        List of merged PhonemeSegment instances
    """
    if not segments:
        return segments

    merged_segments: list[PhonemeSegment] = []
    current_batch: list[PhonemeSegment] = []
    current_phoneme_length = 0

    def flush_batch() -> None:
        """Flush current batch to merged_segments."""
        nonlocal current_batch, current_phoneme_length

        if not current_batch:
            return

        if len(current_batch) == 1:
            # Single segment - check if it needs cascading
            segment = current_batch[0]
            if len(segment.phonemes) > max_phoneme_length:
                # Segment too long, cascade split it
                split_segments = _cascade_split_segment(
                    segment, tokenizer, max_phoneme_length
                )
                # Preserve pause_after on the last sub-segment
                if split_segments and segment.pause_after > 0:
                    split_segments[-1].pause_after = segment.pause_after
                merged_segments.extend(split_segments)
            else:
                merged_segments.append(segment)
        else:
            # Merge multiple segments
            merged_text = " ".join(
                seg.text for seg in current_batch if seg.text.strip()
            )
            lang = current_batch[0].lang
            paragraph = current_batch[0].paragraph

            # Check if any segment has phoneme overrides via SSMD
            has_phoneme_overrides = any(
                seg.ssmd_metadata and seg.ssmd_metadata.get("phonemes")
                for seg in current_batch
            )

            if has_phoneme_overrides:
                # Preserve phoneme overrides - concatenate phonemes directly
                # This prevents re-phonemization which would lose the overrides
                merged_phonemes = " ".join(
                    seg.phonemes for seg in current_batch if seg.phonemes.strip()
                )
            else:
                # Re-phonemize the merged text for accurate phonemes
                # This gives better results when there are no explicit overrides
                merged_phonemes = tokenizer.phonemize(merged_text, lang=lang)

            # Check if merged result exceeds limit
            if len(merged_phonemes) > max_phoneme_length:
                # Merged result too long - need to cascade split
                # Create a temporary segment for cascading
                first_sent = current_batch[0].sentence
                last_sent = current_batch[-1].sentence
                if first_sent is None or last_sent is None:
                    sentence_meta: int | str | None = None
                elif first_sent == last_sent:
                    sentence_meta = first_sent
                else:
                    sentence_meta = f"{first_sent}-{last_sent}"

                temp_segment = PhonemeSegment(
                    text=merged_text,
                    phonemes=merged_phonemes,
                    tokens=[],  # Will be regenerated in cascade
                    lang=lang,
                    paragraph=paragraph,
                    sentence=sentence_meta,
                    pause_after=current_batch[-1].pause_after,
                    ssmd_metadata=current_batch[-1].ssmd_metadata,
                )

                # Cascade split the merged segment
                split_segments = _cascade_split_segment(
                    temp_segment, tokenizer, max_phoneme_length
                )
                # Preserve pause_after on the last sub-segment
                if split_segments and temp_segment.pause_after > 0:
                    split_segments[-1].pause_after = temp_segment.pause_after
                merged_segments.extend(split_segments)
            else:
                # Merged result fits within limit
                merged_tokens = tokenizer.tokenize(merged_phonemes)

                # Create sentence range metadata
                first_sent = current_batch[0].sentence
                last_sent = current_batch[-1].sentence
                if first_sent is None or last_sent is None:
                    sentence_meta = None
                elif first_sent == last_sent:
                    sentence_meta = first_sent
                else:
                    sentence_meta = f"{first_sent}-{last_sent}"

                # Preserve pause_after and ssmd_metadata from last segment
                merged_segment = PhonemeSegment(
                    text=merged_text,
                    phonemes=merged_phonemes,
                    tokens=merged_tokens,
                    lang=lang,
                    paragraph=paragraph,
                    sentence=sentence_meta,
                    pause_after=current_batch[-1].pause_after,
                    ssmd_metadata=current_batch[-1].ssmd_metadata,
                )
                merged_segments.append(merged_segment)

        current_batch = []
        current_phoneme_length = 0

    def should_break_batch(segment: PhonemeSegment) -> bool:
        """Check if we should start a new batch before this segment."""
        if not current_batch:
            return False

        last_seg = current_batch[-1]

        # Break on paragraph boundary
        if segment.paragraph != last_seg.paragraph:
            return True

        # Break on language change
        if segment.lang != last_seg.lang:
            return True

        # Break if last segment has explicit SSMD pause
        # (user wants a pause here, so we should respect that boundary)
        if last_seg.pause_after > 0:
            return True

        # Break if segment has special SSMD metadata that shouldn't be merged
        if segment.ssmd_metadata:
            # Voice changes should create boundaries
            if segment.ssmd_metadata.get("voice_name"):
                return True
            # Prosody changes should create boundaries
            if any(
                segment.ssmd_metadata.get(k)
                for k in ["prosody_volume", "prosody_pitch", "prosody_rate"]
            ):
                return True
            # Phoneme overrides should NOT automatically create boundaries
            # They can be merged with other phoneme-override segments

        return False

    for segment in segments:
        # Skip empty segments but preserve them (they may have pause_after)
        if not segment.text.strip():
            flush_batch()
            merged_segments.append(segment)
            continue

        # Check if we need to start a new batch
        if should_break_batch(segment):
            flush_batch()

        segment_length = len(segment.phonemes)

        # Check if adding this segment would exceed limit
        if (
            current_batch
            and current_phoneme_length + segment_length > max_phoneme_length
        ):
            flush_batch()

        # Add segment to current batch
        current_batch.append(segment)
        current_phoneme_length += segment_length

    # Flush remaining batch
    flush_batch()

    return merged_segments


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
                    "spaCy required for clause splitting but not installed. "
                    "Falling back to word-level splitting."
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
