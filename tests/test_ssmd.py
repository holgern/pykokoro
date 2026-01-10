"""Tests for SSMD (Speech Synthesis Markdown) integration in pykokoro."""

import pytest


class TestSSMDBreakParsing:
    """Tests for SSMD break marker parsing."""

    def test_parse_ssmd_breaks_basic(self):
        """Test basic SSMD break markers."""
        from pykokoro.ssmd_parser import parse_ssmd_breaks

        # Test basic strength codes
        initial, segments = parse_ssmd_breaks("Hello ...c world")
        assert initial == 0.0
        assert len(segments) == 2
        assert segments[0] == ("Hello", 0.3)  # clause pause
        assert segments[1] == ("world", 0.0)

    def test_parse_ssmd_breaks_all_strengths(self):
        """Test all SSMD break strength codes."""
        from pykokoro.ssmd_parser import parse_ssmd_breaks

        initial, segments = parse_ssmd_breaks("A ...n B ...w C ...c D ...s E ...p F")
        assert initial == 0.0
        assert len(segments) == 6
        assert segments[0] == ("A", 0.0)  # none
        assert segments[1] == ("B", 0.15)  # weak
        assert segments[2] == ("C", 0.3)  # clause
        assert segments[3] == ("D", 0.6)  # sentence
        assert segments[4] == ("E", 1.0)  # paragraph
        assert segments[5] == ("F", 0.0)

    def test_parse_ssmd_breaks_custom_time_ms(self):
        """Test custom time break in milliseconds."""
        from pykokoro.ssmd_parser import parse_ssmd_breaks

        initial, segments = parse_ssmd_breaks("Wait ...500ms please")
        assert initial == 0.0
        assert len(segments) == 2
        assert segments[0] == ("Wait", 0.5)
        assert segments[1] == ("please", 0.0)

    def test_parse_ssmd_breaks_custom_time_seconds(self):
        """Test custom time break in seconds."""
        from pykokoro.ssmd_parser import parse_ssmd_breaks

        initial, segments = parse_ssmd_breaks("Wait ...2s please")
        assert initial == 0.0
        assert len(segments) == 2
        assert segments[0] == ("Wait", 2.0)
        assert segments[1] == ("please", 0.0)

    def test_parse_ssmd_breaks_custom_durations(self):
        """Test custom pause durations."""
        from pykokoro.ssmd_parser import parse_ssmd_breaks

        initial, segments = parse_ssmd_breaks(
            "A ...c B",
            pause_clause=0.5,
            pause_sentence=1.0,
            pause_paragraph=2.0,
        )
        assert initial == 0.0
        assert segments[0] == ("A", 0.5)  # Custom clause duration

    def test_parse_ssmd_breaks_consecutive(self):
        """Test consecutive breaks add up."""
        from pykokoro.ssmd_parser import parse_ssmd_breaks

        initial, segments = parse_ssmd_breaks("Start ...p ...s End")
        assert initial == 0.0
        assert len(segments) == 2
        assert segments[0] == ("Start", 1.6)  # 1.0 + 0.6
        assert segments[1] == ("End", 0.0)

    def test_parse_ssmd_breaks_leading(self):
        """Test leading pause becomes initial pause."""
        from pykokoro.ssmd_parser import parse_ssmd_breaks

        initial, segments = parse_ssmd_breaks("...p Hello world")
        assert initial == 1.0
        assert len(segments) == 1
        assert segments[0] == ("Hello world", 0.0)

    def test_parse_ssmd_breaks_trailing(self):
        """Test trailing pause."""
        from pykokoro.ssmd_parser import parse_ssmd_breaks

        initial, segments = parse_ssmd_breaks("Goodbye ...p")
        assert initial == 0.0
        assert len(segments) == 1
        assert segments[0] == ("Goodbye", 1.0)

    def test_parse_ssmd_breaks_bare_ellipsis_ignored(self):
        """Test that bare ... (ellipsis) is NOT treated as a pause."""
        from pykokoro.ssmd_parser import parse_ssmd_breaks

        initial, segments = parse_ssmd_breaks("Wait... really?")
        assert initial == 0.0
        assert len(segments) == 1
        assert segments[0] == ("Wait... really?", 0.0)

    def test_parse_ssmd_breaks_decimal_seconds(self):
        """Test decimal seconds in custom time."""
        from pykokoro.ssmd_parser import parse_ssmd_breaks

        initial, segments = parse_ssmd_breaks("A ...1.5s B")
        assert initial == 0.0
        assert segments[0] == ("A", 1.5)
        assert segments[1] == ("B", 0.0)


class TestSSMDDetection:
    """Tests for SSMD markup detection."""

    def test_has_ssmd_markup_breaks(self):
        """Test detection of SSMD break markers."""
        from pykokoro.ssmd_parser import has_ssmd_markup

        assert has_ssmd_markup("Hello ...c world")
        assert has_ssmd_markup("Test ...500ms pause")
        assert has_ssmd_markup("Wait ...2s")
        assert not has_ssmd_markup("Hello... world")  # Bare ellipsis
        assert not has_ssmd_markup("Plain text")

    def test_has_ssmd_markup_emphasis(self):
        """Test detection of emphasis markers."""
        from pykokoro.ssmd_parser import has_ssmd_markup

        assert has_ssmd_markup("This is *important*")
        assert has_ssmd_markup("This is **very important**")
        assert not has_ssmd_markup("This has * asterisks * but not emphasis")

    def test_has_ssmd_markup_prosody(self):
        """Test detection of prosody shorthand."""
        from pykokoro.ssmd_parser import has_ssmd_markup

        assert has_ssmd_markup("Speak +loud+")
        assert has_ssmd_markup("Talk >fast>")
        assert has_ssmd_markup("Say ^high^")
        assert not has_ssmd_markup("Normal text")

    def test_has_ssmd_markup_annotations(self):
        """Test detection of annotations."""
        from pykokoro.ssmd_parser import has_ssmd_markup

        assert has_ssmd_markup("[Bonjour](fr)")
        assert has_ssmd_markup("[word](/phoneme/)")
        assert not has_ssmd_markup("No markup here")

    def test_has_ssmd_markup_markers(self):
        """Test detection of markers."""
        from pykokoro.ssmd_parser import has_ssmd_markup

        assert has_ssmd_markup("Text with @marker")
        assert not has_ssmd_markup("Email@example.com")  # @ in email
        assert not has_ssmd_markup("Plain text")


class TestSSMDSegmentConversion:
    """Tests for SSMD segment parsing and conversion."""

    def test_parse_ssmd_to_segments_basic(self):
        """Test basic SSMD parsing to segments."""
        from pykokoro.ssmd_parser import parse_ssmd_to_segments
        from pykokoro.tokenizer import create_tokenizer

        tokenizer = create_tokenizer()
        initial, segments = parse_ssmd_to_segments(
            "Hello ...c world",
            tokenizer=tokenizer,
        )

        assert initial == 0.0
        assert len(segments) == 2
        assert segments[0].text == "Hello"
        assert segments[0].pause_after == 0.3
        assert segments[1].text == "world"
        assert segments[1].pause_after == 0.0

    def test_parse_ssmd_to_segments_with_markup(self):
        """Test SSMD parsing strips markup from text."""
        from pykokoro.ssmd_parser import parse_ssmd_to_segments
        from pykokoro.tokenizer import create_tokenizer

        tokenizer = create_tokenizer()
        initial, segments = parse_ssmd_to_segments(
            "This is *important* ...s Really!",
            tokenizer=tokenizer,
        )

        assert len(segments) == 2
        # Markup should be stripped from text
        assert "important" in segments[0].text
        assert "*" not in segments[0].text  # Markup removed

    def test_ssmd_segments_to_phoneme_segments(self):
        """Test converting SSMD segments to phoneme segments."""
        from pykokoro.ssmd_parser import (
            SSMDSegment,
            SSMDMetadata,
            ssmd_segments_to_phoneme_segments,
        )
        from pykokoro.tokenizer import create_tokenizer

        tokenizer = create_tokenizer()

        ssmd_segments = [
            SSMDSegment(text="Hello", pause_after=0.5, metadata=SSMDMetadata()),
            SSMDSegment(text="world", pause_after=0.0, metadata=SSMDMetadata()),
        ]

        phoneme_segments = ssmd_segments_to_phoneme_segments(
            ssmd_segments,
            initial_pause=0.0,
            tokenizer=tokenizer,
        )

        assert len(phoneme_segments) == 2
        assert phoneme_segments[0].text == "Hello"
        assert phoneme_segments[0].pause_after == 0.5
        assert len(phoneme_segments[0].phonemes) > 0
        assert len(phoneme_segments[0].tokens) > 0

    def test_ssmd_segments_with_initial_pause(self):
        """Test SSMD segments with initial pause."""
        from pykokoro.ssmd_parser import (
            SSMDSegment,
            SSMDMetadata,
            ssmd_segments_to_phoneme_segments,
        )
        from pykokoro.tokenizer import create_tokenizer

        tokenizer = create_tokenizer()

        ssmd_segments = [
            SSMDSegment(text="Hello", pause_after=0.0, metadata=SSMDMetadata()),
        ]

        phoneme_segments = ssmd_segments_to_phoneme_segments(
            ssmd_segments,
            initial_pause=1.0,
            tokenizer=tokenizer,
        )

        # Should have empty segment for initial pause + text segment
        assert len(phoneme_segments) == 2
        assert phoneme_segments[0].text == ""
        assert phoneme_segments[0].pause_after == 1.0
        assert phoneme_segments[1].text == "Hello"


class TestSSMDMetadata:
    """Tests for SSMD metadata structures."""

    def test_ssmd_metadata_creation(self):
        """Test creating SSMD metadata."""
        from pykokoro.ssmd_parser import SSMDMetadata

        metadata = SSMDMetadata(
            emphasis="strong",
            language="fr",
            phonemes="bɔ̃ʒuʁ",
        )

        assert metadata.emphasis == "strong"
        assert metadata.language == "fr"
        assert metadata.phonemes == "bɔ̃ʒuʁ"

    def test_ssmd_metadata_to_dict(self):
        """Test converting metadata to dictionary."""
        from pykokoro.ssmd_parser import SSMDMetadata

        metadata = SSMDMetadata(emphasis="moderate")
        data = metadata.to_dict()

        assert isinstance(data, dict)
        assert data["emphasis"] == "moderate"
        assert "prosody_volume" in data
        assert "language" in data

    def test_ssmd_segment_creation(self):
        """Test creating SSMD segment."""
        from pykokoro.ssmd_parser import SSMDSegment, SSMDMetadata

        segment = SSMDSegment(
            text="Hello",
            pause_after=0.5,
            metadata=SSMDMetadata(emphasis="strong"),
        )

        assert segment.text == "Hello"
        assert segment.pause_after == 0.5
        assert segment.metadata.emphasis == "strong"


class TestSSMDIntegration:
    """Integration tests for SSMD with text_to_phoneme_segments."""

    def test_text_to_phoneme_segments_with_ssmd(self):
        """Test that text_to_phoneme_segments handles SSMD breaks."""
        from pykokoro.phonemes import text_to_phoneme_segments
        from pykokoro.tokenizer import create_tokenizer

        tokenizer = create_tokenizer()

        # Text with SSMD breaks
        segments = text_to_phoneme_segments(
            text="Hello ...c world ...s End",
            tokenizer=tokenizer,
            lang="en-us",
        )

        # Should have 3 segments with appropriate pauses
        assert len(segments) == 3
        assert segments[0].text == "Hello"
        assert segments[0].pause_after == 0.3  # clause pause
        assert segments[1].text == "world"
        assert segments[1].pause_after == 0.6  # sentence pause
        assert segments[2].text == "End"
        assert segments[2].pause_after == 0.0

    def test_text_to_phoneme_segments_custom_ssmd_durations(self):
        """Test custom SSMD pause durations."""
        from pykokoro.phonemes import text_to_phoneme_segments
        from pykokoro.tokenizer import create_tokenizer

        tokenizer = create_tokenizer()

        segments = text_to_phoneme_segments(
            text="A ...c B",
            tokenizer=tokenizer,
            pause_clause=0.5,
            pause_sentence=1.0,
            pause_paragraph=2.0,
        )

        assert segments[0].pause_after == 0.5  # Custom clause duration

    def test_text_without_ssmd_works_normally(self):
        """Test that text without SSMD still works."""
        from pykokoro.phonemes import text_to_phoneme_segments
        from pykokoro.tokenizer import create_tokenizer

        tokenizer = create_tokenizer()

        segments = text_to_phoneme_segments(
            text="Hello world",
            tokenizer=tokenizer,
        )

        assert len(segments) == 1
        assert segments[0].text == "Hello world"
        assert segments[0].pause_after == 0.0


class TestSSMDVoiceAnnotations:
    """Tests for SSMD voice annotation parsing."""

    def test_parse_voice_annotation_simple_name(self):
        """Test parsing simple voice name."""
        from pykokoro.ssmd_parser import parse_voice_annotation

        result = parse_voice_annotation("voice: af_sarah")
        assert result["name"] == "af_sarah"
        assert result["language"] is None
        assert result["gender"] is None
        assert result["variant"] is None

    def test_parse_voice_annotation_cloud_tts(self):
        """Test parsing cloud TTS voice name."""
        from pykokoro.ssmd_parser import parse_voice_annotation

        result = parse_voice_annotation("voice: en-US-Wavenet-A")
        assert result["name"] == "en-US-Wavenet-A"

    def test_parse_voice_annotation_with_gender(self):
        """Test parsing voice with language and gender."""
        from pykokoro.ssmd_parser import parse_voice_annotation

        result = parse_voice_annotation("voice: fr-FR, gender: female")
        assert result["name"] == "fr-FR"  # Ambiguous, treated as name
        assert result["gender"] == "female"

    def test_parse_voice_annotation_all_attributes(self):
        """Test parsing voice with all attributes."""
        from pykokoro.ssmd_parser import parse_voice_annotation

        result = parse_voice_annotation("voice: en-GB, gender: male, variant: 1")
        assert result["name"] == "en-GB"
        assert result["gender"] == "male"
        assert result["variant"] == "1"

    def test_parse_voice_annotation_language_explicit(self):
        """Test parsing with explicit language attribute."""
        from pykokoro.ssmd_parser import parse_voice_annotation

        result = parse_voice_annotation("language: fr-FR, gender: female")
        assert result["name"] is None
        assert result["language"] == "fr-FR"
        assert result["gender"] == "female"

    def test_extract_ssmd_metadata_voice(self):
        """Test extracting voice metadata from text."""
        from pykokoro.ssmd_parser import extract_ssmd_metadata

        metadata = extract_ssmd_metadata("[Hello](voice: af_sarah)")
        assert metadata.voice_name == "af_sarah"
        assert metadata.voice_language is None
        assert metadata.voice_gender is None
        assert metadata.voice_variant is None

    def test_extract_ssmd_metadata_voice_with_attributes(self):
        """Test extracting voice with multiple attributes."""
        from pykokoro.ssmd_parser import extract_ssmd_metadata

        metadata = extract_ssmd_metadata("[Bonjour](voice: fr-FR, gender: female)")
        assert metadata.voice_name == "fr-FR"
        assert metadata.voice_gender == "female"

    def test_extract_ssmd_metadata_voice_and_emphasis(self):
        """Test extracting both voice and emphasis."""
        from pykokoro.ssmd_parser import extract_ssmd_metadata

        metadata = extract_ssmd_metadata("*[Important](voice: af_nicole)*")
        # Note: Emphasis detection happens on outer level
        assert metadata.voice_name == "af_nicole"

    def test_extract_ssmd_metadata_multiple_annotations(self):
        """Test text with multiple different annotations."""
        from pykokoro.ssmd_parser import extract_ssmd_metadata

        # Only last voice annotation should be kept
        metadata = extract_ssmd_metadata(
            "[Hello](voice: af_sarah) and [Goodbye](voice: am_michael)"
        )
        # Last voice wins
        assert metadata.voice_name == "am_michael"

    def test_ssmd_metadata_to_dict_includes_voice(self):
        """Test that to_dict includes voice fields."""
        from pykokoro.ssmd_parser import SSMDMetadata

        metadata = SSMDMetadata(
            voice_name="af_sarah",
            voice_language="en-US",
            voice_gender="female",
            voice_variant="1",
        )
        data = metadata.to_dict()

        assert data["voice_name"] == "af_sarah"
        assert data["voice_language"] == "en-US"
        assert data["voice_gender"] == "female"
        assert data["voice_variant"] == "1"


class TestSSMDVoiceSwitching:
    """Tests for per-segment voice switching functionality."""

    def test_ssmd_metadata_preserved_in_phoneme_segments(self):
        """Test that voice metadata is preserved in PhonemeSegment."""
        from pykokoro.ssmd_parser import (
            SSMDSegment,
            SSMDMetadata,
            ssmd_segments_to_phoneme_segments,
        )
        from pykokoro.tokenizer import create_tokenizer

        tokenizer = create_tokenizer()

        # Create SSMD segments with voice metadata
        ssmd_segments = [
            SSMDSegment(
                text="Hello",
                pause_after=0.5,
                metadata=SSMDMetadata(voice_name="af_sarah"),
            ),
            SSMDSegment(
                text="World",
                pause_after=0.0,
                metadata=SSMDMetadata(voice_name="am_michael"),
            ),
        ]

        phoneme_segments = ssmd_segments_to_phoneme_segments(
            ssmd_segments,
            initial_pause=0.0,
            tokenizer=tokenizer,
        )

        assert len(phoneme_segments) == 2
        assert phoneme_segments[0].ssmd_metadata is not None
        assert phoneme_segments[0].ssmd_metadata["voice_name"] == "af_sarah"
        assert phoneme_segments[1].ssmd_metadata is not None
        assert phoneme_segments[1].ssmd_metadata["voice_name"] == "am_michael"

    def test_parse_ssmd_with_voice_creates_metadata(self):
        """Test that parsing SSMD text with voice creates proper metadata."""
        from pykokoro.ssmd_parser import parse_ssmd_to_segments
        from pykokoro.tokenizer import create_tokenizer

        tokenizer = create_tokenizer()

        text = "[Hello](voice: af_sarah) ...s [World](voice: am_michael)"
        initial_pause, segments = parse_ssmd_to_segments(text, tokenizer)

        assert len(segments) == 2
        assert segments[0].metadata.voice_name == "af_sarah"
        assert segments[0].pause_after == 0.6  # sentence pause
        assert segments[1].metadata.voice_name == "am_michael"

    def test_voice_resolver_called_for_segment_with_voice(self):
        """Test that AudioGenerator calls voice_resolver for segments with voice metadata."""
        from pykokoro.audio_generator import AudioGenerator
        from pykokoro.phonemes import PhonemeSegment
        from pykokoro.tokenizer import create_tokenizer
        from unittest.mock import Mock
        import numpy as np

        tokenizer = create_tokenizer()

        # Create mock session
        mock_session = Mock()
        mock_session.get_inputs.return_value = [Mock(name="input_ids")]
        mock_session.run.return_value = [np.zeros((1, 100), dtype=np.float32)]

        generator = AudioGenerator(mock_session, tokenizer)

        # Create segments with voice metadata
        segments = [
            PhonemeSegment(
                text="Hello",
                phonemes="hɛˈloʊ",
                tokens=[1, 2, 3],
                ssmd_metadata={"voice_name": "af_sarah"},
            ),
            PhonemeSegment(
                text="World",
                phonemes="wɝld",
                tokens=[4, 5],
                ssmd_metadata={"voice_name": "am_michael"},
            ),
        ]

        # Mock voice resolver
        voice_calls = []

        def mock_voice_resolver(voice_name: str) -> np.ndarray:
            voice_calls.append(voice_name)
            return np.zeros(512, dtype=np.float32)

        default_voice = np.zeros(512, dtype=np.float32)

        # Generate with voice resolver
        audio = generator.generate_from_segments(
            segments,
            default_voice,
            speed=1.0,
            trim_silence=False,
            voice_resolver=mock_voice_resolver,
        )

        # Verify voice_resolver was called for each segment
        assert len(voice_calls) == 2
        assert voice_calls[0] == "af_sarah"
        assert voice_calls[1] == "am_michael"

    def test_voice_switching_without_resolver_uses_default(self):
        """Test that segments with voice metadata but no resolver use default voice."""
        from pykokoro.audio_generator import AudioGenerator
        from pykokoro.phonemes import PhonemeSegment
        from pykokoro.tokenizer import create_tokenizer
        from unittest.mock import Mock
        import numpy as np

        tokenizer = create_tokenizer()

        # Create mock session
        mock_session = Mock()
        mock_session.get_inputs.return_value = [Mock(name="input_ids")]
        mock_session.run.return_value = [np.zeros((1, 100), dtype=np.float32)]

        generator = AudioGenerator(mock_session, tokenizer)

        # Create segment with voice metadata
        segments = [
            PhonemeSegment(
                text="Hello",
                phonemes="hɛˈloʊ",
                tokens=[1, 2, 3],
                ssmd_metadata={"voice_name": "af_sarah"},
            ),
        ]

        default_voice = np.zeros(512, dtype=np.float32)

        # Generate WITHOUT voice resolver (should use default)
        audio = generator.generate_from_segments(
            segments,
            default_voice,
            speed=1.0,
            trim_silence=False,
            voice_resolver=None,  # No resolver
        )

        # Should succeed and use default voice
        assert isinstance(audio, np.ndarray)

    def test_text_to_phoneme_segments_preserves_voice_metadata(self):
        """Test end-to-end: text with SSMD voice → PhonemeSegments with metadata."""
        from pykokoro.phonemes import text_to_phoneme_segments
        from pykokoro.tokenizer import create_tokenizer

        tokenizer = create_tokenizer()

        text = "[Hello there](voice: af_sarah) ...s [Goodbye](voice: am_michael)"

        segments = text_to_phoneme_segments(
            text=text,
            tokenizer=tokenizer,
            lang="en-us",
        )

        # Should have 2 segments with voice metadata
        assert len(segments) >= 2

        # Find segments with actual text (not empty pause segments)
        text_segments = [s for s in segments if s.text.strip()]
        assert len(text_segments) == 2

        assert text_segments[0].ssmd_metadata is not None
        assert text_segments[0].ssmd_metadata["voice_name"] == "af_sarah"
        assert text_segments[1].ssmd_metadata is not None
        assert text_segments[1].ssmd_metadata["voice_name"] == "am_michael"
