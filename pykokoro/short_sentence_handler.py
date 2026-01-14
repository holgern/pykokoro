"""Short sentence handling for pykokoro using single-word context approach.

This module provides functionality to improve audio quality for short, single-word
sentences by using a "context-prepending" technique:

Only activates for short (<5 phonemes) AND single-word sentences (no spaces)

This approach produces better prosody and intonation compared to generating
very short sentences directly, as neural TTS models typically need more context
to produce natural-sounding speech.

Multi-word or sentences with internal breaks will NOT use this handler, as they
already have sufficient context for natural prosody.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from .audio_generator import AudioGenerator
    from .phonemes import PhonemeSegment
    from .tokenizer import Tokenizer

logger = logging.getLogger(__name__)
# Enable debug logging for this module
logger.setLevel(logging.DEBUG)


def is_single_word(text: str) -> bool:
    """Check if text is a single word (contains no spaces).

    Args:
        text: Text to check

    Returns:
        True if text contains no spaces (single word), False otherwise

    Examples:
        >>> is_single_word("Hi!")
        True
        >>> is_single_word("Hi there!")
        False
        >>> is_single_word("Don't!")
        True
        >>> is_single_word("Oh, really?")
        False
    """
    return " " not in text.strip()


@dataclass
class ShortSentenceConfig:
    """Configuration for short sentence handling using single-word context.

    Short, single-word sentences (< 10 phonemes, no spaces) often sound robotic
    when generated alone. This module improves quality by:
    1. Checking sentence is both short AND single-word (no spaces)
    2. Adding phoneme around word

    Multi-word sentences or sentences with breaks will NOT use this handler.

    Attributes:
        min_phoneme_length: Threshold below which sentences are considered "short"
            and will use context extraction. Default: 10 phonemes.
        phoneme_pretext: Phoneme(s) to add before and after the target word
            when generating combined audio for context. Default: "—".
        enabled: Whether short sentence handling is enabled. Default: True.

    """

    min_phoneme_length: int = 5
    phoneme_pretext: str = "—"
    enabled: bool = True

    def should_use_pause_surrounding(self, phoneme_length: int, text: str) -> bool:
        """Check if segment should use pause surrounding.

        Args:
            phoneme_length: Number of phonemes in the segment
            text: The text content to check for single-word status

        Returns:
            True if pause-surrounding should be applied
            (sentence is short AND single-word)
        """
        return (
            self.enabled
            and phoneme_length < self.min_phoneme_length
            and is_single_word(text)
        )

    def contains_only_punctuation(self, phoneme: str) -> bool:
        """Check if segment contains only pounctions.

        Args:
            phoneme_length: Number of phonemes in the segment
            text: The text content to check for single-word status

        Returns:
            True if segment skipping should be applied
            (sentence is short AND single-word)
        """
        contains_only = ';:,.!?—…"()“” '

        return (
            self.enabled
            and len(phoneme) < self.min_phoneme_length
            and all(char in contains_only for char in phoneme)
        )


def generate_short_sentence_audio(
    segment: PhonemeSegment,
    audio_generator: AudioGenerator,
    voice_style: np.ndarray,
    speed: float,
    config: ShortSentenceConfig | None = None,
    tokenizer: Tokenizer | None = None,
) -> np.ndarray:
    """Generate high-quality audio for short, single-word sentences using context.

    This function duplicates the word with a pause and finds a midpoint boundary:
    1. Only activates for short (<10 phonemes) AND single-word sentences (no spaces)
    2. Add phoneme(s) around the word for contex
    3. Generate combined audio

    Multi-word sentences will NOT use this handler and generate normally.

    Args:
        segment: PhonemeSegment containing the sentence
        audio_generator: AudioGenerator instance for TTS
        voice_style: Voice style vector
        speed: Speech speed multiplier
        config: Short sentence configuration (uses defaults if None)
        tokenizer: Tokenizer for phonemizing combined text (uses audio_generator's
            tokenizer if None)

    Returns:
        High-quality audio for the sentence (with context extraction if applicable)

    Note:
        This function makes 1 TTS call: generates context + target together
        (or target alone if multi-word)
    """
    if config is None:
        config = ShortSentenceConfig()

    if tokenizer is None:
        tokenizer = audio_generator._tokenizer

    phoneme_length = len(segment.phonemes)

    # Check if should use context prepending (short AND single-word)
    if not config.should_use_pause_surrounding(phoneme_length, segment.text):
        # Multi-word or long sentence - generate normally
        audio, _ = audio_generator.generate_from_phonemes(
            segment.phonemes, voice_style, speed
        )

        if not is_single_word(segment.text):
            logger.debug(
                f"Skipping short handler for multi-word sentence: '{segment.text[:50]}'"
            )

        return audio

    logger.debug(
        f"Using context extraction for short single-word: '{segment.text[:50]}' "
        f"({phoneme_length} phonemes)"
    )

    phonemes = tokenizer.phonemize(segment.text, lang=segment.lang)
    combined_phonemes = config.phoneme_pretext + phonemes + config.phoneme_pretext
    combined_audio, _ = audio_generator.generate_from_phonemes(
        combined_phonemes, voice_style, speed
    )

    return combined_audio


def is_segment_empty(
    segment: PhonemeSegment,
    config: ShortSentenceConfig | None = None,
) -> bool:
    """Check if segment contains only .

    Checks if segment is BOTH short (<10 phonemes) AND contains only pounctions.

    Args:
        segment: PhonemeSegment to check
        config: Configuration (uses defaults if None)

    Returns:
        True if segment should be skipped
    """
    if config is None:
        config = ShortSentenceConfig()

    # Skip empty segments
    if not segment.phonemes.strip():
        return False
    return config.contains_only_punctuation(segment.phonemes)


def is_segment_short(
    segment: PhonemeSegment,
    config: ShortSentenceConfig | None = None,
) -> bool:
    """Check if segment should use context-prepending.

    Checks if segment is BOTH short (<10 phonemes) AND single-word (no spaces).

    Args:
        segment: PhonemeSegment to check
        config: Configuration (uses defaults if None)

    Returns
        True if segment should use pause-surrounding (short AND single-word)
    """
    if config is None:
        config = ShortSentenceConfig()

    # Skip empty segments
    if not segment.phonemes.strip():
        return False

    return config.should_use_pause_surrounding(len(segment.phonemes), segment.text)
