"""SSMD (Speech Synthesis Markdown) parser for pykokoro.

This module provides integration with the SSMD library to support
rich markup syntax for TTS generation including:
- Breaks/Pauses: ...c (comma), ...s (sentence), ...p (paragraph), ...500ms
- Emphasis: *text* (moderate), **text** (strong)
- Prosody: +loud+, >fast>, ^high^, etc.
- Language switching: [Bonjour](fr)
- Phonetic pronunciation: [tomato](ph: təˈmeɪtoʊ)
- Substitution: [H2O](sub: water)
- Say-as: [123](as: cardinal), [3rd](as: ordinal), [+1-555-0123](as: telephone)
- Voice markers: @voice: name
- Markers: @marker_name

This module uses SSMD's parse_sentences() API to extract structured data
and maps it to PyKokoro's internal segment representation.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

from ssmd import TTSCapabilities, parse_sentences

if TYPE_CHECKING:
    from ssmd import SSMDSegment as SSMDParserSegment

    from .tokenizer import Tokenizer

ANNOTATION_RE = re.compile(
    r"""
    \[
        [^\]]+                      # [text]
    \]
    \{
        \s*\w+\s*=\s*
        (?:'[^']+'|"[^"]+")         # 'value' OR "value"
        (?:\s+\w+\s*=\s*(?:'[^']+'|"[^"]+"))*
        \s*
    \}
    """,
    re.VERBOSE,
)

LANG_SHORTHAND_RE = re.compile(r"\[([^\]]+)\]\(([a-zA-Z]{2,3}(?:-[a-zA-Z]{2})?)\)")


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
        say_as_interpret: Say-as interpretation type (e.g., "telephone", "date")
        say_as_format: Say-as format attribute (e.g., "mdy" for dates)
        say_as_detail: Say-as detail attribute (e.g., "2" for cardinal detail)
        markers: List of marker names in this segment
        voice_name: Voice name for this segment (e.g., "af_sarah", "Joanna")
        voice_language: Voice language attribute (e.g., "en-US", "fr-FR")
        voice_gender: Voice gender attribute ("male", "female", "neutral")
        voice_variant: Voice variant number for multi-variant voices
        audio_src: Audio file source URL or path (for audio segments)
        audio_alt_text: Fallback text if audio cannot be played
    """

    emphasis: str | None = None
    prosody_volume: str | None = None
    prosody_rate: str | None = None
    prosody_pitch: str | None = None
    language: str | None = None
    phonemes: str | None = None
    substitution: str | None = None
    say_as_interpret: str | None = None
    say_as_format: str | None = None
    say_as_detail: str | None = None
    markers: list[str] = field(default_factory=list)
    voice_name: str | None = None
    voice_language: str | None = None
    voice_gender: str | None = None
    voice_variant: str | None = None
    audio_src: str | None = None
    audio_alt_text: str | None = None

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
            "say_as_interpret": self.say_as_interpret,
            "say_as_format": self.say_as_format,
            "say_as_detail": self.say_as_detail,
            "markers": self.markers,
            "voice_name": self.voice_name,
            "voice_language": self.voice_language,
            "voice_gender": self.voice_gender,
            "voice_variant": self.voice_variant,
            "audio_src": self.audio_src,
            "audio_alt_text": self.audio_alt_text,
        }


@dataclass
class SSMDSegment:
    """A parsed segment from SSMD markup.

    Attributes:
        text: Processed text (after substitutions, stripped of markup)
        pause_before: Pause duration before this segment in seconds (e.g., for headings)
        pause_after: Pause duration after this segment in seconds
        metadata: SSMD metadata (emphasis, prosody, etc.)
        paragraph: Paragraph index this segment belongs to
    """

    text: str
    pause_before: float = 0.0
    pause_after: float = 0.0
    metadata: SSMDMetadata = field(default_factory=SSMDMetadata)
    paragraph: int = 0


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

    # Annotations: [text]{annotation)
    if bool(ANNOTATION_RE.search(text)):
        return True

    # Markers: @name
    if re.search(r"(?:^|\s)@\w+", text):
        return True

    # Voice markers: <div></div>
    if bool(
        re.search(r"<div\b[^>]*=.*?>.*?</div\s*>", text, re.IGNORECASE | re.DOTALL)
    ):
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
    lang: str = "en-us",
) -> tuple[str, SSMDMetadata]:
    """Map SSMD parser segment to PyKokoro metadata.

    Args:
        ssmd_seg: SSMDSegment from SSMD parser
        lang: Language code for say-as normalization

    Returns:
        Tuple of (text, metadata)
    """
    from .say_as import normalize_say_as

    metadata = SSMDMetadata()

    # Handle text transformations (priority order: audio >
    # say-as > sub > phoneme > text)
    text = ssmd_seg.text

    # Audio segments - play pre-recorded audio
    if ssmd_seg.audio:
        metadata.audio_src = ssmd_seg.audio.src
        metadata.audio_alt_text = ssmd_seg.audio.alt_text
        # Use alt_text for phonemization fallback
        text = ssmd_seg.audio.alt_text or ""
    elif ssmd_seg.say_as:
        # Store say-as metadata
        metadata.say_as_interpret = ssmd_seg.say_as.interpret_as
        metadata.say_as_format = ssmd_seg.say_as.format
        metadata.say_as_detail = ssmd_seg.say_as.detail

        # Normalize text based on interpret-as type
        text = normalize_say_as(
            text,
            interpret_as=ssmd_seg.say_as.interpret_as,
            lang=lang,
            format_str=ssmd_seg.say_as.format,
            detail=ssmd_seg.say_as.detail,
        )
    elif ssmd_seg.substitution:
        text = ssmd_seg.substitution
        metadata.substitution = ssmd_seg.substitution
    elif ssmd_seg.phoneme:
        # Access phoneme string from PhonemeAttrs object
        # ssmd_seg.phoneme has .ph (phoneme string) and .alphabet ("ipa" or "x-sampa")
        metadata.phonemes = ssmd_seg.phoneme.ph
        # Keep original text for display, phoneme will override during synthesis

    # Emphasis - SSMD supports: True, "moderate", "strong", "reduced", "none"
    if ssmd_seg.emphasis:
        if isinstance(ssmd_seg.emphasis, bool):
            metadata.emphasis = "moderate"  # True maps to moderate (default)
        elif isinstance(ssmd_seg.emphasis, str):
            metadata.emphasis = ssmd_seg.emphasis  # Use explicit level

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
    model_size: str | None = None,
    use_spacy: bool | None = None,
    heading_levels: dict | None = None,
    parse_yaml_header: bool = False,
) -> tuple[float, list[SSMDSegment]]:
    """Parse SSMD markup and convert to segments with metadata.

    This function uses SSMD's parse_sentences() API to extract structured data
    and maps it to PyKokoro's internal segment representation.

    Features supported:
    - Text segments with substitutions applied
    - Pause durations from break markers (...c, ...s, ...p, ...500ms)
    - Metadata (emphasis, prosody, language, phonemes, voice, etc.)
    - Voice markers (@voice: name) with proper propagation
    - Text transformations (say-as, substitution, phoneme, audio)
    - Markers (@marker_name)
    - Audio segments ([audio](file.mp3))

    Args:
        text: Input text with SSMD markup
        tokenizer: Tokenizer instance (for future use with inline phonemes)
        lang: Default language code
        pause_none: Duration for 'none' break strength in seconds
        pause_weak: Duration for 'weak' break strength in seconds
        pause_clause: Duration for 'medium' break strength in seconds
        pause_sentence: Duration for 'strong' break strength in seconds
        pause_paragraph: Duration for 'x-strong' break strength in seconds
        model_size: Size of spacy model for sentence detection ('sm', 'md', 'lg')
        use_spacy: Force use of spacy for sentence detection
        heading_levels: Custom heading configurations (overrides default)
        parse_yaml_header: If True, parse YAML front matter and apply config

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
    # Convert shorthand language annotations like [Bonjour](fr)
    text = LANG_SHORTHAND_RE.sub(r'[\1]{lang="\2"}', text)

    # Use SSMD's parse_sentences to get structured data
    # Enable heading detection for markdown-style headings (# ## ###)
    caps = TTSCapabilities()
    caps.heading_emphasis = True

    sentences = parse_sentences(
        ssmd_text=text,
        sentence_detection=True,
        include_default_voice=True,
        language=lang,
        capabilities=caps,
        model_size=model_size,
        use_spacy=use_spacy,
        heading_levels=heading_levels,
        parse_yaml_header=parse_yaml_header,
    )

    if not sentences:
        return 0.0, []

    pykokoro_segments = []
    initial_pause = 0.0

    # Track paragraph index for proper segmentation
    paragraph_idx = 0

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

        # Process each segment in sentence
        for seg_idx, ssmd_seg in enumerate(sentence.segments):
            # Determine language for this segment
            # Priority: segment lang > sentence lang > default
            segment_lang = ssmd_seg.language or lang

            # Map SSMD segment to PyKokoro metadata
            seg_text, metadata = _map_ssmd_segment_to_metadata(ssmd_seg, segment_lang)

            # Apply voice context if segment doesn't have its own voice
            if not metadata.voice_name and voice_metadata.voice_name:
                metadata.voice_name = voice_metadata.voice_name
                metadata.voice_language = voice_metadata.voice_language
                metadata.voice_gender = voice_metadata.voice_gender
                metadata.voice_variant = voice_metadata.voice_variant

            # Calculate pause before this segment (for headings)
            pause_before = 0.0

            # Check for breaks before this segment
            if ssmd_seg.breaks_before:
                # Use the last break if multiple
                last_break = ssmd_seg.breaks_before[-1]
                pause_before = _convert_break_strength_to_duration(
                    last_break.strength,
                    last_break.time,
                    pause_none=pause_none,
                    pause_weak=pause_weak,
                    pause_clause=pause_clause,
                    pause_sentence=pause_sentence,
                    pause_paragraph=pause_paragraph,
                )

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

            # Create PyKokoro SSMDSegment with paragraph tracking
            pykokoro_segments.append(
                SSMDSegment(
                    text=seg_text,
                    pause_before=pause_before,
                    pause_after=pause_after,
                    metadata=metadata,
                    paragraph=paragraph_idx,
                )
            )

        # Increment paragraph index when we hit a paragraph boundary
        if sentence.is_paragraph_end:
            paragraph_idx += 1

    return initial_pause, pykokoro_segments
