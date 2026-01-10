"""SSMD (Speech Synthesis Markdown) parser for pykokoro.

This module provides integration with the SSMD library to support
rich markup syntax for TTS generation including:
- Breaks/Pauses: ...c (comma), ...s (sentence), ...p (paragraph), ...500ms
- Emphasis: *text* (moderate), **text** (strong)
- Prosody: +loud+, >fast>, ^high^, etc.
- Language switching: [Bonjour](fr)
- Phonetic pronunciation: [tomato](ph: təˈmeɪtoʊ)
- Substitution: [H2O](sub: water)
- Markers: @marker_name
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

from ssmd import Document, to_text

if TYPE_CHECKING:
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


def parse_ssmd_breaks(
    text: str,
    pause_none: float = 0.0,
    pause_weak: float = 0.15,
    pause_clause: float = 0.3,
    pause_sentence: float = 0.6,
    pause_paragraph: float = 1.0,
) -> tuple[float, list[tuple[str, float]]]:
    """Parse SSMD break syntax and extract segments with pause durations.

    Detects and splits on SSMD break markers:
    - ...n (none) → pause_none (default 0.0s)
    - ...w (weak/x-weak) → pause_weak (default 0.15s)
    - ...c (comma/medium) → pause_clause (default 0.3s)
    - ...s (sentence/strong) → pause_sentence (default 0.6s)
    - ...p (paragraph/x-strong) → pause_paragraph (default 1.0s)
    - ...500ms → 0.5s
    - ...2s → 2.0s

    Note: Bare ... (ellipsis without modifier) is NOT treated as a pause
    and will be passed to kokorog2p for phonemization.

    Args:
        text: Input text with optional SSMD break markers
        pause_none: Duration for ...n in seconds
        pause_weak: Duration for ...w in seconds
        pause_clause: Duration for ...c in seconds
        pause_sentence: Duration for ...s in seconds
        pause_paragraph: Duration for ...p in seconds

    Returns:
        Tuple of (initial_pause_duration, segments_list) where segments_list
        is a list of (text_segment, pause_after_seconds) tuples

    Example:
        >>> parse_ssmd_breaks("Hello ...c world ...s Foo")
        (0.0, [("Hello", 0.3), ("world", 0.6), ("Foo", 0.0)])

        >>> parse_ssmd_breaks("Start ...500ms pause ...2s End")
        (0.0, [("Start", 0.5), ("pause", 2.0), ("End", 0.0)])

        >>> parse_ssmd_breaks("...p After a long pause")
        (1.0, [("After a long pause", 0.0)])
    """
    # Pattern to match SSMD break markers: ...(n|w|c|s|p) or ...123ms or ...2s
    # Note: Does NOT match bare ... (ellipsis)
    pattern = r"\.\.\.(?:([nwcsp])|(\d+)ms|(\d+(?:\.\d+)?)s)\b"

    # Split text and capture markers
    parts = re.split(f"({pattern})", text)

    # Process parts into segments with pauses
    segments = []
    current_text = ""
    accumulated_pause = 0.0
    initial_pause = 0.0
    found_first_text = False

    strength_map = {
        "n": pause_none,
        "w": pause_weak,
        "c": pause_clause,
        "s": pause_sentence,
        "p": pause_paragraph,
    }

    i = 0
    while i < len(parts):
        part = parts[i]

        # Check if this is a complete break marker match
        if i + 3 < len(parts) and re.match(r"\.\.\.(?:[nwcsp]|\d)", part):
            # Extract the break type from the next parts
            strength_code = parts[i + 1] if parts[i + 1] else None
            milliseconds = parts[i + 2] if parts[i + 2] else None
            seconds = parts[i + 3] if parts[i + 3] else None

            # Determine pause duration
            if strength_code and strength_code in strength_map:
                pause_duration = strength_map[strength_code]
            elif milliseconds:
                pause_duration = int(milliseconds) / 1000.0
            elif seconds:
                pause_duration = float(seconds)
            else:
                # Shouldn't happen due to regex, but be safe
                pause_duration = 0.0

            # If we haven't found any text yet, this is initial pause
            if not found_first_text and not current_text.strip():
                initial_pause += pause_duration
            else:
                # Accumulate pause for current segment
                accumulated_pause += pause_duration

            # Skip the matched groups
            i += 4
            continue

        # This is regular text
        if part and not re.match(r"^[nwcsp]$", part) and part not in ["", None]:
            stripped = part.strip()
            if stripped:
                found_first_text = True

                # If we have accumulated text, save it as a segment
                if current_text.strip():
                    segments.append((current_text.strip(), accumulated_pause))
                    accumulated_pause = 0.0
                    current_text = ""

                current_text = part  # Keep original spacing
            elif current_text.strip():
                # Whitespace part after we have text, keep accumulating
                current_text += part

        i += 1

    # Add final segment if any
    if current_text.strip():
        segments.append((current_text.strip(), accumulated_pause))

    return initial_pause, segments


def has_ssmd_markup(text: str) -> bool:
    """Check if text contains SSMD markup.

    Detects:
    - Break markers: ...c, ...s, ...p, ...500ms, ...2s
    - Emphasis: *text* or **text**
    - Prosody shorthand: +loud+, >fast>, ^high^, etc.
    - Annotations: [text](...)
    - Markers: @name

    Note: Does NOT treat bare ... as markup (ellipsis).

    Args:
        text: Input text to check

    Returns:
        True if text contains SSMD markup, False otherwise

    Example:
        >>> has_ssmd_markup("Hello ...c world")
        True
        >>> has_ssmd_markup("Hello *world*")
        True
        >>> has_ssmd_markup("Hello ... world")  # Bare ellipsis
        False
        >>> has_ssmd_markup("Hello world")
        False
    """
    # Check for SSMD break markers (not bare ...)
    if re.search(r"\.\.\.(?:[nwcsp]|\d+ms|\d+(?:\.\d+)?s)\b", text):
        return True

    # Check for emphasis (* or **) - must not have spaces after opening marker
    if re.search(r"\*+\S[^*]*\*+", text):
        return True

    # Check for prosody shorthand (+, >, <, ^, _, ~, -)
    if re.search(r"[+>^_~<-]{1,2}[^+>^_~<-]+[+>^_~<-]{1,2}", text):
        return True

    # Check for annotations [text](...)
    if re.search(r"\[([^\]]+)\]\(([^)]+)\)", text):
        return True

    # Check for markers @name - must be preceded by whitespace or start of string
    if re.search(r"(?:^|\s)@\w+", text):
        return True

    return False


def parse_voice_annotation(annotation_content: str) -> dict[str, str | None]:
    """Parse voice annotation from SSMD markup.

    Supports SSMD voice syntax:
    - Simple voice name: "voice: Joanna" or "voice: af_sarah"
    - Cloud TTS voice: "voice: en-US-Wavenet-A"
    - Language and gender: "voice: fr-FR, gender: female"
    - All attributes: "voice: en-GB, gender: male, variant: 1"

    Args:
        annotation_content: The content inside parentheses (e.g., "voice: Joanna")

    Returns:
        Dictionary with keys: 'name', 'language', 'gender', 'variant'
        (any of which may be None)

    Example:
        >>> parse_voice_annotation("voice: Joanna")
        {'name': 'Joanna', 'language': None, 'gender': None, 'variant': None}

        >>> parse_voice_annotation("voice: fr-FR, gender: female")
        {'name': None, 'language': 'fr-FR', 'gender': 'female', 'variant': None}
    """
    result: dict[str, str | None] = {
        "name": None,
        "language": None,
        "gender": None,
        "variant": None,
    }

    # Split by comma to handle multiple attributes
    parts = [p.strip() for p in annotation_content.split(",")]

    for part in parts:
        if ":" not in part:
            continue

        key, value = part.split(":", 1)
        key = key.strip().lower()
        value = value.strip()

        if key == "voice":
            # Could be a simple name or a language code
            # If it looks like a language code (has hyphen) and no other language specified
            if "-" in value and result["language"] is None:
                # Could be cloud TTS voice like "en-US-Wavenet-A" or language "fr-FR"
                # If it has more than one hyphen, treat as voice name
                if value.count("-") > 1:
                    result["name"] = value
                else:
                    # Ambiguous - could be voice name or language
                    # Default to name, will be overridden if explicit language follows
                    result["name"] = value
            else:
                result["name"] = value
        elif key == "language":
            result["language"] = value
        elif key == "gender":
            result["gender"] = value
        elif key == "variant":
            result["variant"] = value

    return result


def extract_voice_from_ssml(ssml: str) -> dict[str, str | None]:
    """Extract voice attributes from SSML <voice> tag.

    Args:
        ssml: SSML text that may contain <voice> tags

    Returns:
        Dictionary with voice attributes: name, language, gender, variant

    Example:
        >>> extract_voice_from_ssml('<voice name="sarah">Hello</voice>')
        {'name': 'sarah', 'language': None, 'gender': None, 'variant': None}
    """
    result: dict[str, str | None] = {
        "name": None,
        "language": None,
        "gender": None,
        "variant": None,
    }

    # Look for <voice> tag with attributes
    voice_pattern = r"<voice\s+([^>]+)>"
    voice_match = re.search(voice_pattern, ssml)
    if not voice_match:
        return result

    # Extract all attributes from the voice tag
    attrs_text = voice_match.group(1)
    attr_pattern = r'(\w+)="([^"]+)"'
    for attr_match in re.finditer(attr_pattern, attrs_text):
        key = attr_match.group(1).lower()
        value = attr_match.group(2)

        if key == "name":
            result["name"] = value
        elif key == "language" or key == "lang":
            result["language"] = value
        elif key == "gender":
            result["gender"] = value
        elif key == "variant":
            result["variant"] = value

    return result


def _extract_pause_from_ssml(
    ssml: str,
    pause_none: float = 0.0,
    pause_weak: float = 0.15,
    pause_clause: float = 0.3,
    pause_sentence: float = 0.6,
    pause_paragraph: float = 1.0,
) -> float:
    """Extract pause duration from SSML <break> tag.

    Args:
        ssml: SSML text that may contain <break> tags
        pause_none: Duration for "none" strength
        pause_weak: Duration for "weak" or "x-weak" strength
        pause_clause: Duration for "medium" strength
        pause_sentence: Duration for "strong" strength
        pause_paragraph: Duration for "x-strong" strength

    Returns:
        Pause duration in seconds (uses last break tag found, defaults to 0.0)

    Example:
        >>> _extract_pause_from_ssml('Hello<break strength="medium"/>')
        0.3
    """
    # Look for <break> tags
    break_pattern = r"<break\s+([^>]+)/>"
    breaks = list(re.finditer(break_pattern, ssml))

    if not breaks:
        return 0.0

    # Use the last break in the sentence (typically at the end)
    last_break = breaks[-1]
    attrs_text = last_break.group(1)

    # Extract strength attribute
    strength_pattern = r'strength="([^"]+)"'
    strength_match = re.search(strength_pattern, attrs_text)
    if strength_match:
        strength = strength_match.group(1)
        strength_map = {
            "none": pause_none,
            "x-weak": pause_weak,
            "weak": pause_weak,
            "medium": pause_clause,
            "strong": pause_sentence,
            "x-strong": pause_paragraph,
        }
        return strength_map.get(strength, 0.0)

    # Check for time attribute (e.g., time="500ms")
    time_pattern = r'time="([0-9.]+)(ms|s)"'
    time_match = re.search(time_pattern, attrs_text)
    if time_match:
        value = float(time_match.group(1))
        unit = time_match.group(2)
        if unit == "ms":
            return value / 1000.0
        else:  # seconds
            return value

    return 0.0


def _extract_metadata_from_ssml(ssml: str) -> SSMDMetadata:
    """Extract all metadata from SSML sentence.

    Args:
        ssml: SSML sentence string

    Returns:
        SSMDMetadata with extracted information

    Example:
        >>> _extract_metadata_from_ssml('<voice name="sarah"><emphasis>Hello</emphasis></voice>')
        SSMDMetadata(voice_name='sarah', emphasis='moderate')
    """
    metadata = SSMDMetadata()

    # Extract voice information
    voice_info = extract_voice_from_ssml(ssml)
    if voice_info["name"]:
        metadata.voice_name = voice_info["name"]
        metadata.voice_language = voice_info["language"]
        metadata.voice_gender = voice_info["gender"]
        metadata.voice_variant = voice_info["variant"]

    # Extract emphasis
    if "<emphasis>" in ssml or '<emphasis level="moderate">' in ssml:
        metadata.emphasis = "moderate"
    elif '<emphasis level="strong">' in ssml:
        metadata.emphasis = "strong"

    # Extract language (from <lang> or <voice language="..."> tags)
    lang_pattern = r'<lang xml:lang="([^"]+)"'
    lang_match = re.search(lang_pattern, ssml)
    if lang_match:
        metadata.language = lang_match.group(1)

    # TODO: Extract prosody, phonemes, markers if needed

    return metadata


def _strip_ssml_tags(ssml: str) -> str:
    """Strip all XML/SSML tags from text.

    Args:
        ssml: SSML text with tags

    Returns:
        Plain text with all tags removed

    Example:
        >>> _strip_ssml_tags('<voice name="sarah">Hello</voice>')
        'Hello'
    """
    return re.sub(r"<[^>]+>", "", ssml)


def extract_ssmd_metadata(text: str, ssml: str | None = None) -> SSMDMetadata:
    """Extract SSMD metadata from a text segment.

    Parses SSMD annotations to extract metadata like emphasis, voice,
    language, substitution, etc.

    Supports both voice annotation formats:
    - Inline: [text](voice: name)
    - Marker: @voice: name (via SSML <voice> tags when ssml parameter provided)

    Args:
        text: Text segment potentially containing SSMD markup
        ssml: Optional SSML version of text (for extracting <voice> tags)

    Returns:
        SSMDMetadata instance with extracted metadata

    Example:
        >>> extract_ssmd_metadata("*important*")
        SSMDMetadata(emphasis='moderate', ...)

        >>> extract_ssmd_metadata("[Hello](voice: af_sarah)")
        SSMDMetadata(voice_name='af_sarah', ...)

        >>> extract_ssmd_metadata("Hello", '<voice name="sarah">Hello</voice>')
        SSMDMetadata(voice_name='sarah', ...)
    """
    metadata = SSMDMetadata()

    # First check for voice in SSML (from @voice: markers)
    if ssml:
        voice_info = extract_voice_from_ssml(ssml)
        if voice_info["name"]:
            metadata.voice_name = voice_info["name"]
            metadata.voice_language = voice_info["language"]
            metadata.voice_gender = voice_info["gender"]
            metadata.voice_variant = voice_info["variant"]

    # Check for emphasis
    if re.search(r"\*\*[^*]+\*\*", text):
        metadata.emphasis = "strong"
    elif re.search(r"\*[^*]+\*", text):
        metadata.emphasis = "moderate"

    # Extract annotations [text](annotation)
    annotation_pattern = r"\[([^\]]+)\]\(([^)]+)\)"
    for match in re.finditer(annotation_pattern, text):
        annotation_text = match.group(1)
        annotation_content = match.group(2)

        # Check annotation type
        if annotation_content.startswith("voice:") or ", gender:" in annotation_content:
            # Voice annotation - but only use if no @voice marker from SSML
            # When SSML is provided, voice from SSML takes precedence
            # When multiple inline annotations exist, last one wins
            if not (ssml and metadata.voice_name):
                voice_info = parse_voice_annotation(annotation_content)
                metadata.voice_name = voice_info["name"]
                metadata.voice_language = voice_info["language"]
                metadata.voice_gender = voice_info["gender"]
                metadata.voice_variant = voice_info["variant"]
        elif annotation_content.startswith("sub:"):
            # Substitution
            metadata.substitution = annotation_content[4:].strip()
        elif annotation_content.startswith("ph:"):
            # Phonemes (SSMD syntax)
            metadata.phonemes = annotation_content[3:].strip()
        elif annotation_content.startswith("/") and annotation_content.endswith("/"):
            # Phonemes (kokorog2p syntax)
            metadata.phonemes = annotation_content[1:-1]
        elif re.match(r"^[a-z]{2}(-[A-Z]{2})?$", annotation_content):
            # Language code (e.g., "fr", "en-GB")
            metadata.language = annotation_content

    # Extract markers @name (excluding @voice which is handled above)
    marker_pattern = r"(?:^|\s)@(?!voice:)(\w+)"
    markers = re.findall(marker_pattern, text, re.IGNORECASE)
    if markers:
        metadata.markers = markers

    return metadata


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

    This function processes SSMD markup to extract:
    - Text segments with substitutions applied
    - Pause durations from break markers
    - Metadata (emphasis, prosody, language, phonemes, voice, etc.)

    Voice markers (@voice: name) are handled by SSMD and propagate to all following
    text in the same paragraph until the next @voice: marker or paragraph break.

    Args:
        text: Input text with SSMD markup
        tokenizer: Tokenizer instance (for future use with inline phonemes)
        lang: Default language code
        pause_none: Duration for ...n in seconds
        pause_weak: Duration for ...w in seconds
        pause_clause: Duration for ...c in seconds
        pause_sentence: Duration for ...s in seconds
        pause_paragraph: Duration for ...p in seconds

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
    # Strategy:
    # 1. Split text by paragraph breaks (double newlines)
    # 2. For each paragraph, parse breaks using parse_ssmd_breaks
    # 3. Extract voice metadata from SSML (handles @voice: markers)
    # 4. @voice: marker applies to entire paragraph

    # Split by paragraph breaks but preserve the structure
    paragraphs = re.split(r"\n\s*\n", text)

    all_segments = []
    first_paragraph = True
    initial_pause = 0.0

    for para_text in paragraphs:
        if not para_text.strip():
            continue

        # Parse breaks within this paragraph
        para_initial_pause, para_segments = parse_ssmd_breaks(
            para_text,
            pause_none=pause_none,
            pause_weak=pause_weak,
            pause_clause=pause_clause,
            pause_sentence=pause_sentence,
            pause_paragraph=pause_paragraph,
        )

        # Store initial pause from first paragraph only
        if first_paragraph:
            initial_pause = para_initial_pause
            first_paragraph = False

        # Convert segment to SSML once per paragraph to extract voice
        # (voice marker applies to whole paragraph)
        para_doc = Document(para_text)
        para_ssml = para_doc.to_ssml()
        para_voice_metadata = _extract_metadata_from_ssml(para_ssml)

        # Process each break segment within the paragraph
        for seg_text, pause_after in para_segments:
            # Extract metadata specific to this segment
            seg_doc = Document(seg_text)
            seg_ssml = seg_doc.to_ssml()
            metadata = _extract_metadata_from_ssml(seg_ssml)

            # If paragraph has a voice marker, use it (unless segment has its own)
            if para_voice_metadata.voice_name and not metadata.voice_name:
                metadata.voice_name = para_voice_metadata.voice_name
                metadata.voice_language = para_voice_metadata.voice_language
                metadata.voice_gender = para_voice_metadata.voice_gender
                metadata.voice_variant = para_voice_metadata.voice_variant

            # Strip markup from segment text
            clean_text = _strip_ssml_tags(seg_ssml)

            all_segments.append(
                SSMDSegment(text=clean_text, pause_after=pause_after, metadata=metadata)
            )

    return initial_pause, all_segments


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

    # Process each SSMD segment
    for i, ssmd_seg in enumerate(ssmd_segments):
        # Determine language (use metadata override or default)
        lang = ssmd_seg.metadata.language or default_lang

        # Use explicit phonemes if provided, otherwise phonemize
        if ssmd_seg.metadata.phonemes:
            phonemes = ssmd_seg.metadata.phonemes
        else:
            # Apply substitution if present
            text_to_phonemize = ssmd_seg.metadata.substitution or ssmd_seg.text
            phonemes = tokenizer.phonemize(text_to_phonemize, lang=lang)

        # Tokenize phonemes
        tokens = tokenizer.tokenize(phonemes)

        # Create phoneme segment with SSMD metadata
        segment = PhonemeSegment(
            text=ssmd_seg.text,
            phonemes=phonemes,
            tokens=tokens,
            lang=lang,
            paragraph=paragraph,
            sentence=sentence_start + i,
            pause_after=ssmd_seg.pause_after,
            ssmd_metadata=ssmd_seg.metadata.to_dict(),
        )

        segments.append(segment)

    return segments
