"""Tests for SSMD (Speech Synthesis Markdown) integration in pykokoro."""


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

        # SSMD splits on emphasis markers, creating segments for each part
        assert len(segments) == 3
        # First segment: text before emphasis
        assert segments[0].text == "This is"
        # Second segment: emphasized text (markup stripped)
        assert segments[1].text == "important"
        assert "*" not in segments[1].text  # Markup removed
        # Third segment: text after pause
        assert "Really!" in segments[2].text

    def test_parse_ssmd_to_segments_without_markup(self):
        """Test SSMD parsing strips markup from text."""
        from pykokoro.ssmd_parser import parse_ssmd_to_segments
        from pykokoro.tokenizer import create_tokenizer

        tokenizer = create_tokenizer()
        initial, segments = parse_ssmd_to_segments(
            "Hello this is great. Really!",
            tokenizer=tokenizer,
        )

        assert len(segments) == 2
        # Markup should be stripped from text
        assert "Hello this is great." in segments[0].text
        assert "Really!" in segments[1].text

    def test_ssmd_segments_to_phoneme_segments(self):
        """Test converting SSMD segments to phoneme segments."""
        from pykokoro.ssmd_parser import (
            SSMDMetadata,
            SSMDSegment,
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
            SSMDMetadata,
            SSMDSegment,
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
        from pykokoro.ssmd_parser import SSMDMetadata, SSMDSegment

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


class TestSSMDVoiceSwitching:
    """Tests for per-segment voice switching functionality."""

    def test_ssmd_metadata_preserved_in_phoneme_segments(self):
        """Test that voice metadata is preserved in PhonemeSegment."""
        from pykokoro.ssmd_parser import (
            SSMDMetadata,
            SSMDSegment,
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

        # Use proper @voice: marker syntax (not inline annotation)
        text = "@voice: af_sarah\nHello ...s\n\n@voice: am_michael\nWorld"
        initial_pause, segments = parse_ssmd_to_segments(text, tokenizer)

        assert len(segments) == 2
        assert segments[0].metadata.voice_name == "af_sarah"
        assert segments[0].pause_after == 0.6  # sentence pause
        assert segments[1].metadata.voice_name == "am_michael"

    def test_voice_resolver_called_for_segment_with_voice(self):
        """Test AudioGenerator calls voice_resolver for voice metadata."""
        from unittest.mock import Mock

        import numpy as np

        from pykokoro.audio_generator import AudioGenerator
        from pykokoro.phonemes import PhonemeSegment
        from pykokoro.tokenizer import create_tokenizer

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

        # Verify audio was generated
        assert isinstance(audio, np.ndarray)

        # Verify voice_resolver was called for each segment
        assert len(voice_calls) == 2
        assert voice_calls[0] == "af_sarah"
        assert voice_calls[1] == "am_michael"

    def test_voice_switching_without_resolver_uses_default(self):
        """Test that segments with voice metadata but no resolver use default voice."""
        from unittest.mock import Mock

        import numpy as np

        from pykokoro.audio_generator import AudioGenerator
        from pykokoro.phonemes import PhonemeSegment
        from pykokoro.tokenizer import create_tokenizer

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

        # Use proper @voice: marker syntax
        text = "@voice: af_sarah\nHello there ...s\n\n@voice: am_michael\nGoodbye"

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


class TestSSMDSayAsSupport:
    """Tests for say-as support in SSMD integration."""

    def test_say_as_cardinal_normalization(self):
        """Test say-as cardinal number normalization."""
        from pykokoro.phonemes import text_to_phoneme_segments
        from pykokoro.tokenizer import Tokenizer

        tokenizer = Tokenizer()
        text = "I have [123](as: cardinal) apples"
        segments = text_to_phoneme_segments(text, tokenizer, lang="en-us")

        # Find segment with normalized number
        normalized_seg = next(
            (
                s
                for s in segments
                if s.ssmd_metadata
                and s.ssmd_metadata.get("say_as_interpret") == "cardinal"
            ),
            None,
        )
        assert normalized_seg is not None
        assert "hundred" in normalized_seg.text.lower()
        assert normalized_seg.ssmd_metadata["say_as_interpret"] == "cardinal"

    def test_say_as_ordinal_normalization(self):
        """Test say-as ordinal number normalization."""
        from pykokoro.phonemes import text_to_phoneme_segments
        from pykokoro.tokenizer import Tokenizer

        tokenizer = Tokenizer()
        text = "I came in [3](as: ordinal) place"
        segments = text_to_phoneme_segments(text, tokenizer, lang="en-us")

        normalized_seg = next(
            (
                s
                for s in segments
                if s.ssmd_metadata
                and s.ssmd_metadata.get("say_as_interpret") == "ordinal"
            ),
            None,
        )
        assert normalized_seg is not None
        assert normalized_seg.text.lower() == "third"
        assert normalized_seg.ssmd_metadata["say_as_interpret"] == "ordinal"

    def test_say_as_digits_normalization(self):
        """Test say-as digits normalization."""
        from pykokoro.phonemes import text_to_phoneme_segments
        from pykokoro.tokenizer import Tokenizer

        tokenizer = Tokenizer()
        text = "My PIN is [1234](as: digits)"
        segments = text_to_phoneme_segments(text, tokenizer, lang="en-us")

        normalized_seg = next(
            (
                s
                for s in segments
                if s.ssmd_metadata
                and s.ssmd_metadata.get("say_as_interpret") == "digits"
            ),
            None,
        )
        assert normalized_seg is not None
        assert "one" in normalized_seg.text.lower()
        assert "two" in normalized_seg.text.lower()
        assert "three" in normalized_seg.text.lower()
        assert "four" in normalized_seg.text.lower()

    def test_say_as_characters_normalization(self):
        """Test say-as characters normalization."""
        from pykokoro.phonemes import text_to_phoneme_segments
        from pykokoro.tokenizer import Tokenizer

        tokenizer = Tokenizer()
        text = "Spell [ABC](as: characters)"
        segments = text_to_phoneme_segments(text, tokenizer, lang="en-us")

        normalized_seg = next(
            (
                s
                for s in segments
                if s.ssmd_metadata
                and s.ssmd_metadata.get("say_as_interpret") == "characters"
            ),
            None,
        )
        assert normalized_seg is not None
        assert normalized_seg.text == "A B C"

    def test_say_as_telephone_normalization(self):
        """Test say-as telephone normalization."""
        from pykokoro.phonemes import text_to_phoneme_segments
        from pykokoro.tokenizer import Tokenizer

        tokenizer = Tokenizer()
        text = "Call [+1-555-0123](as: telephone)"
        segments = text_to_phoneme_segments(text, tokenizer, lang="en-us")

        normalized_seg = next(
            (
                s
                for s in segments
                if s.ssmd_metadata
                and s.ssmd_metadata.get("say_as_interpret") == "telephone"
            ),
            None,
        )
        assert normalized_seg is not None
        assert "plus" in normalized_seg.text.lower()
        assert "one" in normalized_seg.text.lower()

    def test_say_as_expletive_censoring(self):
        """Test say-as expletive censoring."""
        from pykokoro.phonemes import text_to_phoneme_segments
        from pykokoro.tokenizer import Tokenizer

        tokenizer = Tokenizer()
        text = "This is [inappropriate](as: expletive)"
        segments = text_to_phoneme_segments(text, tokenizer, lang="en-us")

        normalized_seg = next(
            (
                s
                for s in segments
                if s.ssmd_metadata
                and s.ssmd_metadata.get("say_as_interpret") == "expletive"
            ),
            None,
        )
        assert normalized_seg is not None
        assert normalized_seg.text == "beep"

    def test_say_as_metadata_preserved(self):
        """Test that say-as metadata is preserved in segments."""
        from pykokoro.phonemes import text_to_phoneme_segments
        from pykokoro.tokenizer import Tokenizer

        tokenizer = Tokenizer()
        text = "The year [2024](as: cardinal)"
        segments = text_to_phoneme_segments(text, tokenizer, lang="en-us")

        # Check metadata is preserved
        say_as_seg = next(
            (
                s
                for s in segments
                if s.ssmd_metadata and s.ssmd_metadata.get("say_as_interpret")
            ),
            None,
        )
        assert say_as_seg is not None
        assert say_as_seg.ssmd_metadata["say_as_interpret"] == "cardinal"
        assert say_as_seg.ssmd_metadata.get("say_as_format") is None
        assert say_as_seg.ssmd_metadata.get("say_as_detail") is None
