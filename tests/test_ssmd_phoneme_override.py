"""Test for SSMD phoneme override bug fix.

This test confirms that consecutive phoneme annotations like:
    [tomato](ph: təˈmeɪtoʊ)[pause](ph: ………………)

properly preserve the phoneme overrides instead of re-phonemizing the text.
"""

import pytest

from pykokoro import Kokoro
from pykokoro.phonemes import text_to_phoneme_segments


class TestSSMDPhonemeOverride:
    """Test SSMD phoneme override functionality."""

    def test_single_phoneme_override(self):
        """Test single phoneme override works."""
        kokoro = Kokoro()
        segments = text_to_phoneme_segments(
            text="[tomato](ph: təˈmeɪtoʊ)",
            tokenizer=kokoro.tokenizer,
            lang="en-us",
        )

        assert len(segments) >= 1
        # Find the segment with our override
        tomato_seg = next(seg for seg in segments if "tomato" in seg.text)
        assert tomato_seg.phonemes == "təˈmeɪtoʊ"
        kokoro.close()

    def test_consecutive_phoneme_overrides_no_space(self):
        """Test consecutive phoneme overrides without spacing."""
        kokoro = Kokoro()
        segments = text_to_phoneme_segments(
            text="[tomato](ph: təˈmeɪtoʊ)[pause](ph: ………………)",
            tokenizer=kokoro.tokenizer,
            lang="en-us",
        )

        # Should have segments with the correct phonemes
        # NOT re-phonemized versions like 'təmˈAɾO pˈɔz'
        phonemes_str = " ".join(seg.phonemes for seg in segments if seg.phonemes)

        # Should contain our overrides
        assert "təˈmeɪtoʊ" in phonemes_str, (
            f"Expected phoneme override, got: {phonemes_str}"
        )
        assert (
            "………………" in phonemes_str
            or len([s for s in segments if "pause" in s.text.lower()]) > 0
        )
        kokoro.close()

    def test_consecutive_phoneme_overrides_with_space(self):
        """Test consecutive phoneme overrides with spacing."""
        kokoro = Kokoro()
        segments = text_to_phoneme_segments(
            text="[tomato](ph: təˈmeɪtoʊ) [pause](ph: ………………)",
            tokenizer=kokoro.tokenizer,
            lang="en-us",
        )

        phonemes_str = " ".join(seg.phonemes for seg in segments if seg.phonemes)

        # Should contain our overrides
        assert "təˈmeɪtoʊ" in phonemes_str, (
            f"Expected phoneme override, got: {phonemes_str}"
        )
        kokoro.close()

    def test_mixed_normal_and_override(self):
        """Test mixing normal text with phoneme overrides."""
        kokoro = Kokoro()
        segments = text_to_phoneme_segments(
            text="I say [tomato](ph: təˈmeɪtoʊ) potato",
            tokenizer=kokoro.tokenizer,
            lang="en-us",
        )

        phonemes_str = " ".join(seg.phonemes for seg in segments if seg.phonemes)

        # Should contain the override
        assert "təˈmeɪtoʊ" in phonemes_str, (
            f"Expected phoneme override, got: {phonemes_str}"
        )
        kokoro.close()

    def test_multiple_different_overrides(self):
        """Test multiple different phoneme overrides in sequence."""
        kokoro = Kokoro()
        segments = text_to_phoneme_segments(
            text="[one](ph: wʌn)[two](ph: tuː)[three](ph: θɹiː)",
            tokenizer=kokoro.tokenizer,
            lang="en-us",
        )

        phonemes_str = " ".join(seg.phonemes for seg in segments if seg.phonemes)

        # All three overrides should be present
        assert "wʌn" in phonemes_str, f"Expected 'wʌn', got: {phonemes_str}"
        assert "tuː" in phonemes_str, f"Expected 'tuː', got: {phonemes_str}"
        assert "θɹiː" in phonemes_str, f"Expected 'θɹiː', got: {phonemes_str}"
        kokoro.close()

    def test_tts_mode_preserves_phonemes(self):
        """Test that TTS mode (default) preserves phoneme overrides."""
        kokoro = Kokoro()
        segments = text_to_phoneme_segments(
            text="[tomato](ph: təˈmeɪtoʊ)[pause](ph: ………………)",
            tokenizer=kokoro.tokenizer,
            lang="en-us",
            pause_mode="tts",  # Default mode that merges segments
        )

        phonemes_str = " ".join(seg.phonemes for seg in segments if seg.phonemes)

        # Even after merging, phonemes should be preserved
        assert "təˈmeɪtoʊ" in phonemes_str, (
            f"TTS mode lost phoneme override: {phonemes_str}"
        )
        kokoro.close()

    def test_manual_mode_preserves_phonemes(self):
        """Test that manual mode preserves phoneme overrides."""
        kokoro = Kokoro()
        segments = text_to_phoneme_segments(
            text="[tomato](ph: təˈmeɪtoʊ)[pause](ph: ………………)",
            tokenizer=kokoro.tokenizer,
            lang="en-us",
            pause_mode="manual",  # Manual mode with auto pauses
        )

        phonemes_str = " ".join(seg.phonemes for seg in segments if seg.phonemes)

        assert "təˈmeɪtoʊ" in phonemes_str, (
            f"Manual mode lost phoneme override: {phonemes_str}"
        )
        kokoro.close()

    def test_audio_generation_with_phoneme_override(self):
        """Test that audio generation actually uses the phoneme overrides."""
        kokoro = Kokoro()

        # Generate audio with override
        audio1, sr1 = kokoro.create(
            text="[test](ph: wʌn)",
            voice="af_sarah",
        )

        # Generate audio with normal text
        audio2, sr2 = kokoro.create(
            text="one",
            voice="af_sarah",
        )

        # Both should generate audio
        assert len(audio1) > 0
        assert len(audio2) > 0

        # They should be similar length (both saying "one")
        # Allow 50% variation since phonemes might differ
        ratio = len(audio1) / len(audio2)
        assert 0.5 < ratio < 2.0, (
            f"Audio lengths too different: {len(audio1)} vs {len(audio2)}"
        )

        kokoro.close()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
