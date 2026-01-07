"""Tests for VoiceManager."""

import logging

import numpy as np
import pytest

from pykokoro.voice_manager import VoiceBlend, VoiceManager


@pytest.fixture
def voice_data():
    """Create mock voice data."""
    return {
        "voice1": np.random.rand(512, 256).astype(np.float32),
        "voice2": np.random.rand(512, 256).astype(np.float32),
        "voice3": np.random.rand(512, 256).astype(np.float32),
    }


@pytest.fixture
def mock_npz_file(tmp_path, voice_data):
    """Create a mock .npz file with voice data."""
    voices_path = tmp_path / "voices.npz"
    np.savez(str(voices_path), **voice_data)
    return voices_path


class TestVoiceBlend:
    """Test VoiceBlend dataclass."""

    def test_voice_blend_creation(self):
        """Test creating a VoiceBlend."""
        blend = VoiceBlend(voices=[("voice1", 0.5), ("voice2", 0.5)])
        assert len(blend.voices) == 2
        assert blend.voices[0] == ("voice1", 0.5)
        assert blend.voices[1] == ("voice2", 0.5)

    def test_voice_blend_validation_weights_sum_to_one(self):
        """Test that voice blend weights must sum to 1.0."""
        # Should work
        blend = VoiceBlend(voices=[("voice1", 0.3), ("voice2", 0.7)])
        assert len(blend.voices) == 2

    def test_voice_blend_validation_invalid_weights(self):
        """Test that invalid weights raise error."""
        with pytest.raises(ValueError, match="must sum to 1.0"):
            VoiceBlend(voices=[("voice1", 0.3), ("voice2", 0.3)])

    def test_voice_blend_validation_empty(self):
        """Test that empty voice list raises error."""
        with pytest.raises(ValueError, match="at least one voice"):
            VoiceBlend(voices=[])

    def test_voice_blend_single_voice(self):
        """Test VoiceBlend with a single voice."""
        blend = VoiceBlend(voices=[("voice1", 1.0)])
        assert len(blend.voices) == 1


class TestVoiceManagerInit:
    """Test VoiceManager initialization."""

    def test_init_defaults(self):
        """Test initialization with defaults."""
        manager = VoiceManager()
        assert manager._model_source == "huggingface"
        assert manager._voices_data is None
        assert not manager.is_loaded()

    def test_init_with_github_source(self):
        """Test initialization with GitHub source."""
        manager = VoiceManager(model_source="github")
        assert manager._model_source == "github"
        assert manager._voices_data is None


class TestVoiceLoading:
    """Test voice loading."""

    def test_load_voices_huggingface(self, mock_npz_file, voice_data):
        """Test loading voices from HuggingFace .npz format."""
        manager = VoiceManager(model_source="huggingface")
        manager.load_voices(mock_npz_file)

        assert manager.is_loaded()
        assert len(manager._voices_data) == 3
        assert set(manager._voices_data.keys()) == {"voice1", "voice2", "voice3"}

    def test_load_voices_github(self, mock_npz_file):
        """Test loading voices from GitHub .bin format."""
        manager = VoiceManager(model_source="github")
        manager.load_voices(mock_npz_file)

        assert manager.is_loaded()
        assert len(manager._voices_data) == 3

    def test_load_voices_logging(self, mock_npz_file, caplog):
        """Test that voice loading is logged."""
        caplog.set_level(logging.INFO)
        manager = VoiceManager()
        manager.load_voices(mock_npz_file)

        assert any("Successfully loaded" in record.message for record in caplog.records)
        assert any("3 voices" in record.message for record in caplog.records)


class TestGetVoices:
    """Test getting voice list."""

    def test_get_voices(self, mock_npz_file):
        """Test getting list of available voices."""
        manager = VoiceManager()
        manager.load_voices(mock_npz_file)

        voices = manager.get_voices()
        assert isinstance(voices, list)
        assert len(voices) == 3
        assert voices == ["voice1", "voice2", "voice3"]  # Should be sorted

    def test_get_voices_not_loaded(self):
        """Test that getting voices before loading raises error."""
        manager = VoiceManager()
        with pytest.raises(RuntimeError, match="not loaded"):
            manager.get_voices()


class TestGetVoiceStyle:
    """Test getting voice styles."""

    def test_get_voice_style(self, mock_npz_file, voice_data):
        """Test getting a voice style by name."""
        manager = VoiceManager()
        manager.load_voices(mock_npz_file)

        style = manager.get_voice_style("voice1")
        assert isinstance(style, np.ndarray)
        assert style.shape == (512, 256)
        np.testing.assert_array_equal(style, voice_data["voice1"])

    def test_get_voice_style_not_loaded(self):
        """Test that getting style before loading raises error."""
        manager = VoiceManager()
        with pytest.raises(RuntimeError, match="not loaded"):
            manager.get_voice_style("voice1")

    def test_get_voice_style_not_found(self, mock_npz_file):
        """Test that getting non-existent voice raises KeyError."""
        manager = VoiceManager()
        manager.load_voices(mock_npz_file)

        with pytest.raises(KeyError, match="Voice 'nonexistent' not found"):
            manager.get_voice_style("nonexistent")

    def test_get_voice_style_error_message_includes_available(self, mock_npz_file):
        """Test that error message includes available voices."""
        manager = VoiceManager()
        manager.load_voices(mock_npz_file)

        with pytest.raises(KeyError, match="Available voices:"):
            manager.get_voice_style("nonexistent")


class TestCreateBlendedVoice:
    """Test voice blending."""

    def test_create_blended_voice(self, mock_npz_file, voice_data):
        """Test creating a blended voice."""
        manager = VoiceManager()
        manager.load_voices(mock_npz_file)

        blend = VoiceBlend(voices=[("voice1", 0.6), ("voice2", 0.4)])
        blended = manager.create_blended_voice(blend)

        assert isinstance(blended, np.ndarray)
        assert blended.shape == (512, 256)

        # Verify blend is correct weighted sum
        expected = voice_data["voice1"] * 0.6 + voice_data["voice2"] * 0.4
        np.testing.assert_array_almost_equal(blended, expected)

    def test_create_blended_voice_single(self, mock_npz_file, voice_data):
        """Test that single voice blend returns the voice directly."""
        manager = VoiceManager()
        manager.load_voices(mock_npz_file)

        blend = VoiceBlend(voices=[("voice1", 1.0)])
        blended = manager.create_blended_voice(blend)

        # Should return original voice, not a copy
        np.testing.assert_array_equal(blended, voice_data["voice1"])

    def test_create_blended_voice_three_voices(self, mock_npz_file, voice_data):
        """Test blending three voices."""
        manager = VoiceManager()
        manager.load_voices(mock_npz_file)

        blend = VoiceBlend(
            voices=[
                ("voice1", 0.5),
                ("voice2", 0.3),
                ("voice3", 0.2),
            ]
        )
        blended = manager.create_blended_voice(blend)

        expected = (
            voice_data["voice1"] * 0.5
            + voice_data["voice2"] * 0.3
            + voice_data["voice3"] * 0.2
        )
        np.testing.assert_array_almost_equal(blended, expected)

    def test_create_blended_voice_not_loaded(self):
        """Test that blending before loading raises error."""
        manager = VoiceManager()
        blend = VoiceBlend(voices=[("voice1", 1.0)])

        with pytest.raises(RuntimeError, match="not loaded"):
            manager.create_blended_voice(blend)

    def test_create_blended_voice_invalid_voice(self, mock_npz_file):
        """Test that blending with non-existent voice raises error."""
        manager = VoiceManager()
        manager.load_voices(mock_npz_file)

        blend = VoiceBlend(voices=[("nonexistent", 1.0)])
        with pytest.raises(KeyError, match="not found"):
            manager.create_blended_voice(blend)


class TestResolveVoice:
    """Test voice resolution."""

    def test_resolve_voice_string(self, mock_npz_file, voice_data):
        """Test resolving a voice name string."""
        manager = VoiceManager()
        manager.load_voices(mock_npz_file)

        style = manager.resolve_voice("voice1")
        np.testing.assert_array_equal(style, voice_data["voice1"])

    def test_resolve_voice_blend(self, mock_npz_file, voice_data):
        """Test resolving a VoiceBlend."""
        manager = VoiceManager()
        manager.load_voices(mock_npz_file)

        blend = VoiceBlend(voices=[("voice1", 0.7), ("voice2", 0.3)])
        style = manager.resolve_voice(blend)

        expected = voice_data["voice1"] * 0.7 + voice_data["voice2"] * 0.3
        np.testing.assert_array_almost_equal(style, expected)

    def test_resolve_voice_ndarray(self, mock_npz_file):
        """Test resolving a numpy array (style vector)."""
        manager = VoiceManager()
        manager.load_voices(mock_npz_file)

        custom_style = np.random.rand(512, 256).astype(np.float32)
        style = manager.resolve_voice(custom_style)

        # Should return the same array
        np.testing.assert_array_equal(style, custom_style)

    def test_resolve_voice_not_loaded_string(self):
        """Test that resolving voice name before loading raises error."""
        manager = VoiceManager()
        with pytest.raises(RuntimeError, match="not loaded"):
            manager.resolve_voice("voice1")

    def test_resolve_voice_not_loaded_ndarray(self):
        """Test that resolving ndarray works even without loading."""
        manager = VoiceManager()
        custom_style = np.random.rand(512, 256).astype(np.float32)

        # Should work fine, just returns the array
        style = manager.resolve_voice(custom_style)
        np.testing.assert_array_equal(style, custom_style)


class TestIsLoaded:
    """Test is_loaded() method."""

    def test_is_loaded_false_initially(self):
        """Test that is_loaded() returns False initially."""
        manager = VoiceManager()
        assert not manager.is_loaded()

    def test_is_loaded_true_after_loading(self, mock_npz_file):
        """Test that is_loaded() returns True after loading."""
        manager = VoiceManager()
        manager.load_voices(mock_npz_file)
        assert manager.is_loaded()


class TestEdgeCases:
    """Test edge cases."""

    def test_load_voices_twice(self, mock_npz_file):
        """Test that loading voices twice works."""
        manager = VoiceManager()
        manager.load_voices(mock_npz_file)
        first_voices = manager.get_voices()

        # Load again
        manager.load_voices(mock_npz_file)
        second_voices = manager.get_voices()

        assert first_voices == second_voices

    def test_empty_voice_file(self, tmp_path):
        """Test loading an empty voice file."""
        empty_path = tmp_path / "empty.npz"
        np.savez(str(empty_path))  # Empty npz

        manager = VoiceManager()
        manager.load_voices(empty_path)

        assert manager.is_loaded()
        assert manager.get_voices() == []
