"""SSMD (Speech Synthesis Markdown) parser for pykokoro.

This module provides integration with the SSMD library to support
rich markup syntax for TTS generation including:
- Breaks/Pauses: ...c (comma), ...s (sentence), ...p (paragraph), ...500ms
- Emphasis: *text* (moderate), **text** (strong)
- Prosody: +loud+, >fast>, ^high^, etc.
- Language switching: [Bonjour](fr)
- Phonetic pronunciation: [tomato](ph: təˈmeɪtoʊ)
- Substitution: [H2O](sub: water)
- Voice markers: @voice: name
- Markers: @marker_name

This module uses SSMD's parse_sentences() API to extract structured data
and maps it to PyKokoro's internal segment representation.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

from ssmd import parse_sentences

if TYPE_CHECKING:
    from ssmd import SSMDSegment as SSMDParserSegment

    from .phonemes import PhonemeSegment
    from .tokenizer import Tokenizer


@dataclass
class SSMDMetadata:
    """Metadata extracted from SSMD markup for a text segment.

    Attributes:
        emphasis: Emphasis level ("moderate", "strong", or None)
        prosody_volume: Volume level (0-5 scale or relative like "+6dB")
        prosody_rate: Rate/speed level (1-5 scale or relative like "+20%")
        prosody_pitch: Pitch level (1-5 scale or relative like "+15%")
        language: Language code override for this segment
        phonemes: Explicit phoneme string (bypasses G2P)
        substitution: Substitution text (replaces original before G2P)
        markers: List of marker names in this segment
        voice_name: Voice name for this segment (e.g., "af_sarah", "Joanna")
        voice_language: Voice language attribute (e.g., "en-US", "fr-FR")
        voice_gender: Voice gender attribute ("male", "female", "neutral")
        voice_variant: Voice variant number for multi-variant voices
    """

    emphasis: str | None = None
    prosody_volume: str | None = None
    prosody_rate: str | None = None
    prosody_pitch: str | None = None
    language: str | None = None
    phonemes: str | None = None
    substitution: str | None = None
    markers: list[str] = field(default_factory=list)
    voice_name: str | None = None
    voice_language: str | None = None
    voice_gender: str | None = None
    voice_variant: str | None = None

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            "emphasis": self.emphasis,
            "prosody_volume": self.prosody_volume,
            "prosody_rate": self.prosody_rate,
            "prosody_pitch": self.prosody_pitch,
            "language": self.language,
            "phonemes": self.phonemes,
            "substitution": self.substitution,
            "markers": self.markers,
            "voice_name": self.voice_name,
            "voice_language": self.voice_language,
            "voice_gender": self.voice_gender,
            "voice_variant": self.voice_variant,
        }


@dataclass
class SSMDSegment:
    """A parsed segment from SSMD markup.

    Attributes:
        text: Processed text (after substitutions, stripped of markup)
        pause_after: Pause duration after this segment in seconds
        metadata: SSMD metadata (emphasis, prosody, etc.)
    """

    text: str
    pause_after: float = 0.0
    metadata: SSMDMetadata = field(default_factory=SSMDMetadata)


def has_ssmd_markup(text: str) -> bool:
    """Check if text contains SSMD markup.

    Args:
        text: Input text to check

    Returns:
        True if text contains any SSMD markup patterns
    """
    # Break markers
    if re.search(r"\.\.\.[nwcsp]|\.\.\.(\d+\.?\d*)(ms|s)", text):
        return True

    # Emphasis (must have word character adjacent to asterisk)
    if re.search(r"\*\w[^*]*\*|\*[^*]*\w\*", text):
        return True

    # Prosody
    if re.search(r"[+><^][\w\s]+[+><^]", text):
        return True

    # Annotations: [text](annotation)
    if re.search(r"\[[^\]]+\]\([^)]+\)", text):
        return True

    # Markers: @name
    if re.search(r"(?:^|\s)@\w+", text):
        return True

    # Voice markers: @voice: name
    if re.search(r"@voice:\s*\w+", text, re.IGNORECASE):
        return True

    return False


def _convert_break_strength_to_duration(
    strength: str | None,
    time: str | None,
    pause_none: float = 0.0,
    pause_weak: float = 0.15,
    pause_clause: float = 0.3,
    pause_sentence: float = 0.6,
    pause_paragraph: float = 1.0,
) -> float:
    """Convert SSMD BreakAttrs to pause duration in seconds.

    Args:
        strength: Break strength ('none', 'x-weak', 'weak', 'medium',
            'strong', 'x-strong')
        time: Break time ('500ms', '2s')
        pause_none: Duration for 'none' strength
        pause_weak: Duration for 'x-weak'/'weak' strength
        pause_clause: Duration for 'medium' strength
        pause_sentence: Duration for 'strong' strength
        pause_paragraph: Duration for 'x-strong' strength

    Returns:
        Pause duration in seconds
    """
    # If explicit time is provided, use it
    if time:
        if time.endswith("ms"):
            return float(time[:-2]) / 1000.0
        elif time.endswith("s"):
            return float(time[:-1])

    # Otherwise use strength mapping
    if strength:
        strength_map = {
            "none": pause_none,
            "x-weak": pause_weak,
            "weak": pause_weak,
            "medium": pause_clause,
            "strong": pause_sentence,
            "x-strong": pause_paragraph,
        }
        return strength_map.get(strength, 0.0)

    return 0.0


def _map_ssmd_segment_to_metadata(
    ssmd_seg: SSMDParserSegment,
) -> tuple[str, SSMDMetadata]:
    """Map SSMD parser segment to PyKokoro metadata.

    Args:
        ssmd_seg: SSMDSegment from SSMD parser

    Returns:
        Tuple of (text, metadata)
    """
    metadata = SSMDMetadata()

    # Handle text transformations (substitution > phoneme > original)
    text = ssmd_seg.text
    if ssmd_seg.substitution:
        text = ssmd_seg.substitution
        metadata.substitution = ssmd_seg.substitution
    elif ssmd_seg.phoneme:
        metadata.phonemes = ssmd_seg.phoneme
        # Keep original text for display, phoneme will override during synthesis

    # Emphasis
    if ssmd_seg.emphasis:
        metadata.emphasis = "moderate"  # SSMD uses boolean, map to PyKokoro's string

    # Language
    if ssmd_seg.language:
        metadata.language = ssmd_seg.language

    # Prosody
    if ssmd_seg.prosody:
        metadata.prosody_volume = ssmd_seg.prosody.volume
        metadata.prosody_rate = ssmd_seg.prosody.rate
        metadata.prosody_pitch = ssmd_seg.prosody.pitch

    # Markers (from marks_before and marks_after)
    markers = []
    if ssmd_seg.marks_before:
        markers.extend(ssmd_seg.marks_before)
    if ssmd_seg.marks_after:
        markers.extend(ssmd_seg.marks_after)
    if markers:
        metadata.markers = markers

    return text, metadata


def parse_ssmd_to_segments(
    text: str,
    tokenizer: Tokenizer,
    lang: str = "en-us",
    pause_none: float = 0.0,
    pause_weak: float = 0.15,
    pause_clause: float = 0.3,
    pause_sentence: float = 0.6,
    pause_paragraph: float = 1.0,
) -> tuple[float, list[SSMDSegment]]:
    """Parse SSMD markup and convert to segments with metadata.

    This function uses SSMD's parse_sentences() API to extract structured data
    and maps it to PyKokoro's internal segment representation.

    Features supported:
    - Text segments with substitutions applied
    - Pause durations from break markers (...c, ...s, ...p, ...500ms)
    - Metadata (emphasis, prosody, language, phonemes, voice, etc.)
    - Voice markers (@voice: name) with proper propagation
    - Text transformations (say-as, substitution, phoneme)
    - Markers (@marker_name)

    Args:
        text: Input text with SSMD markup
        tokenizer: Tokenizer instance (for future use with inline phonemes)
        lang: Default language code
        pause_none: Duration for 'none' break strength in seconds
        pause_weak: Duration for 'weak' break strength in seconds
        pause_clause: Duration for 'medium' break strength in seconds
        pause_sentence: Duration for 'strong' break strength in seconds
        pause_paragraph: Duration for 'x-strong' break strength in seconds

    Returns:
        Tuple of (initial_pause, segments) where segments is a list of SSMDSegment

    Example:
        >>> segments = parse_ssmd_to_segments(
        ...     "Hello ...c *important* ...s [Bonjour](fr)",
        ...     tokenizer
        ... )
        >>> segments = parse_ssmd_to_segments(
        ...     "@voice: sarah\\nHello!\\n\\n@voice: michael\\nWorld!",
        ...     tokenizer
        ... )
    """
    # Use SSMD's parse_sentences to get structured data
    sentences = parse_sentences(
        text,
        sentence_detection=True,
        include_default_voice=True,
        language=lang,
    )

    if not sentences:
        return 0.0, []

    pykokoro_segments = []
    initial_pause = 0.0

    for sentence in sentences:
        # Extract voice context for this sentence
        voice_metadata = SSMDMetadata()
        if sentence.voice:
            voice_metadata.voice_name = sentence.voice.name
            voice_metadata.voice_language = sentence.voice.language
            voice_metadata.voice_gender = sentence.voice.gender
            voice_metadata.voice_variant = (
                str(sentence.voice.variant) if sentence.voice.variant else None
            )

        # Process each segment in the sentence
        for seg_idx, ssmd_seg in enumerate(sentence.segments):
            # Map SSMD segment to PyKokoro metadata
            seg_text, metadata = _map_ssmd_segment_to_metadata(ssmd_seg)

            # Apply voice context if segment doesn't have its own voice
            if not metadata.voice_name and voice_metadata.voice_name:
                metadata.voice_name = voice_metadata.voice_name
                metadata.voice_language = voice_metadata.voice_language
                metadata.voice_gender = voice_metadata.voice_gender
                metadata.voice_variant = voice_metadata.voice_variant

            # Calculate pause after this segment
            pause_after = 0.0

            # Check for breaks after this segment
            if ssmd_seg.breaks_after:
                # Use the last break if multiple
                last_break = ssmd_seg.breaks_after[-1]
                pause_after = _convert_break_strength_to_duration(
                    last_break.strength,
                    last_break.time,
                    pause_none=pause_none,
                    pause_weak=pause_weak,
                    pause_clause=pause_clause,
                    pause_sentence=pause_sentence,
                    pause_paragraph=pause_paragraph,
                )

            # If this is the last segment in the sentence, check sentence-level breaks
            if seg_idx == len(sentence.segments) - 1 and sentence.breaks_after:
                last_break = sentence.breaks_after[-1]
                sentence_pause = _convert_break_strength_to_duration(
                    last_break.strength,
                    last_break.time,
                    pause_none=pause_none,
                    pause_weak=pause_weak,
                    pause_clause=pause_clause,
                    pause_sentence=pause_sentence,
                    pause_paragraph=pause_paragraph,
                )
                pause_after = max(pause_after, sentence_pause)

            # Create PyKokoro SSMDSegment
            pykokoro_segments.append(
                SSMDSegment(text=seg_text, pause_after=pause_after, metadata=metadata)
            )

    return initial_pause, pykokoro_segments


def ssmd_segments_to_phoneme_segments(
    ssmd_segments: list[SSMDSegment],
    initial_pause: float,
    tokenizer: Tokenizer,
    default_lang: str = "en-us",
    paragraph: int = 0,
    sentence_start: int = 0,
) -> list[PhonemeSegment]:
    """Convert SSMDSegment list to PhonemeSegment list.

    Args:
        ssmd_segments: List of parsed SSMD segments
        initial_pause: Initial pause before first segment
        tokenizer: Tokenizer for phonemization
        default_lang: Default language code
        paragraph: Paragraph index
        sentence_start: Starting sentence index

    Returns:
        List of PhonemeSegment instances
    """
    from .phonemes import PhonemeSegment

    segments = []

    # Add initial pause as empty segment if present
    if initial_pause > 0:
        segments.append(
            PhonemeSegment(
                text="",
                phonemes="",
                tokens=[],
                lang=default_lang,
                paragraph=paragraph,
                sentence=sentence_start,
                pause_after=initial_pause,
            )
        )

    # Convert each SSMD segment to PhonemeSegment
    for i, ssmd_seg in enumerate(ssmd_segments):
        # Determine language
        lang = ssmd_seg.metadata.language or default_lang

        # Get phonemes - use explicit if provided, otherwise phonemize
        if ssmd_seg.metadata.phonemes:
            # Use explicit phoneme override
            phonemes = ssmd_seg.metadata.phonemes
            tokens = tokenizer.text_to_tokens(phonemes, lang=lang)
        else:
            # Phonemize the text
            phonemes = tokenizer.phonemize(ssmd_seg.text, lang=lang)
            tokens = tokenizer.text_to_tokens(phonemes, lang=lang)

        # Create phoneme segment with voice metadata
        seg = PhonemeSegment(
            text=ssmd_seg.text,
            phonemes=phonemes,
            tokens=tokens,
            lang=lang,
            paragraph=paragraph,
            sentence=sentence_start + i,
            pause_after=ssmd_seg.pause_after,
        )

        # Attach voice metadata for voice switching
        seg.voice_name = ssmd_seg.metadata.voice_name
        seg.voice_language = ssmd_seg.metadata.voice_language
        seg.voice_gender = ssmd_seg.metadata.voice_gender
        seg.voice_variant = ssmd_seg.metadata.voice_variant

        # Store full metadata as dict for backward compatibility
        seg.ssmd_metadata = ssmd_seg.metadata.to_dict()

        segments.append(seg)

    return segments
