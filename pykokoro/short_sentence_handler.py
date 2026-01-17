"""Short sentence handling for pykokoro using single-word context approach.

This module provides functionality to improve audio quality for short, single-word
sentences by applying a "context-prepending" technique during phoneme creation.

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

if TYPE_CHECKING:
    from .phonemes import PhonemeSegment

logger = logging.getLogger(__name__)
# Enable debug logging for this module
logger.setLevel(logging.DEBUG)


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
        return self.enabled and phoneme_length < self.min_phoneme_length

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
