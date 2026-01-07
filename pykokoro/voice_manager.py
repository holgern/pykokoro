"""Voice management for PyKokoro."""

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import numpy as np

logger = logging.getLogger(__name__)

# Model source type
ModelSource = Literal["huggingface", "github"]


@dataclass
class VoiceBlend:
    """Configuration for blending multiple voices.

    Args:
        voices: List of (voice_name, weight) tuples
    """

    voices: list[tuple[str, float]]

    def __post_init__(self):
        """Validate voice blend configuration."""
        if not self.voices:
            raise ValueError("VoiceBlend must have at least one voice")

        total_weight = sum(weight for _, weight in self.voices)
        if not 0.99 <= total_weight <= 1.01:  # Allow small floating point errors
            raise ValueError(f"Voice blend weights must sum to 1.0, got {total_weight}")

    @classmethod
    def parse(cls, blend_str: str) -> "VoiceBlend":
        """Parse a voice blend string.

        Format: "voice1:weight1,voice2:weight2" or "voice1:50,voice2:50"
        Weights should sum to 100 (percentages).

        Example: "af_nicole:50,am_michael:50"

        Args:
            blend_str: String representation of voice blend

        Returns:
            VoiceBlend instance
        """
        voices = []
        for part in blend_str.split(","):
            part = part.strip()
            if ":" in part:
                voice_name, weight_str = part.split(":", 1)
                weight = float(weight_str) / 100.0  # Convert percentage to fraction
            else:
                voice_name = part
                weight = 1.0
            voices.append((voice_name.strip(), weight))

        # Normalize weights if they don't sum to 1
        total_weight = sum(w for _, w in voices)
        if total_weight > 0 and abs(total_weight - 1.0) > 0.01:
            voices = [(v, w / total_weight) for v, w in voices]

        return cls(voices=voices)


class VoiceManager:
    """Manages voice loading and blending for PyKokoro.

    Handles:
    - Loading voices from different formats (.npz for HuggingFace, .bin for GitHub)
    - Voice style retrieval by name
    - Voice blending (weighted combination of multiple voices)
    - Voice listing

    Args:
        model_source: Source of the model ('huggingface' or 'github')
    """

    def __init__(self, model_source: ModelSource = "huggingface"):
        """Initialize the voice manager."""
        self._model_source = model_source
        self._voices_data: dict[str, np.ndarray] | None = None

    def load_voices(self, voices_path: Path) -> None:
        """Load voices from file.

        Args:
            voices_path: Path to the voices file (.npz or .bin)
        """
        if self._model_source == "github":
            self._voices_data = self._load_voices_bin_github(voices_path)
        else:
            # HuggingFace format: .npz archive with named voice arrays
            self._voices_data = dict(np.load(str(voices_path), allow_pickle=True))
            logger.info(
                f"Successfully loaded {len(self._voices_data)} voices "
                f"from {voices_path}"
            )
            logger.debug(
                f"Available voices: {', '.join(sorted(self._voices_data.keys()))}"
            )

    def _load_voices_bin_github(self, voices_path: Path) -> dict[str, np.ndarray]:
        """Load voices from GitHub format .bin file.

        The GitHub voices.bin format is a NumPy archive file (.npz format)
        containing voice arrays with voice names as keys.

        Args:
            voices_path: Path to the voices.bin file

        Returns:
            Dictionary mapping voice names to numpy arrays
        """
        # Load the NumPy file - it's actually .npz format despite .bin extension
        voices_npz = np.load(str(voices_path), allow_pickle=True)

        # Convert NpzFile to dictionary
        voices: dict[str, np.ndarray] = dict(voices_npz)

        logger.info(f"Successfully loaded {len(voices)} voices from {voices_path}")
        logger.debug(f"Available voices: {', '.join(sorted(voices.keys()))}")

        return voices

    def get_voices(self) -> list[str]:
        """Get list of available voice names.

        Returns:
            Sorted list of voice names

        Raises:
            RuntimeError: If voices have not been loaded yet
        """
        if self._voices_data is None:
            raise RuntimeError("Voices not loaded. Call load_voices() first.")
        return list(sorted(self._voices_data.keys()))

    def get_voice_style(self, voice_name: str) -> np.ndarray:
        """Get the style vector for a voice.

        Args:
            voice_name: Name of the voice

        Returns:
            Numpy array representing the voice style

        Raises:
            RuntimeError: If voices have not been loaded yet
            KeyError: If voice_name is not found
        """
        if self._voices_data is None:
            raise RuntimeError("Voices not loaded. Call load_voices() first.")

        if voice_name not in self._voices_data:
            available = ", ".join(sorted(self._voices_data.keys()))
            raise KeyError(
                f"Voice '{voice_name}' not found. Available voices: {available}"
            )

        return self._voices_data[voice_name]

    def create_blended_voice(self, blend: VoiceBlend) -> np.ndarray:
        """Create a blended voice from multiple voices.

        Args:
            blend: VoiceBlend object specifying voices and weights

        Returns:
            Numpy array representing the blended voice style

        Raises:
            RuntimeError: If voices have not been loaded yet
            KeyError: If any voice in the blend is not found
        """
        if self._voices_data is None:
            raise RuntimeError("Voices not loaded. Call load_voices() first.")

        # Optimize: single voice doesn't need blending
        if len(blend.voices) == 1:
            voice_name, _ = blend.voices[0]
            return self.get_voice_style(voice_name)

        # Get style vectors and blend them
        blended: np.ndarray | None = None
        for voice_name, weight in blend.voices:
            style = self.get_voice_style(voice_name)
            weighted = style * weight
            if blended is None:
                blended = weighted
            else:
                blended = np.add(blended, weighted)

        # This should never be None if blend.voices is not empty
        assert blended is not None, "No voices in blend"
        return blended

    def resolve_voice(self, voice: str | np.ndarray | VoiceBlend) -> np.ndarray:
        """Resolve a voice specification to a style vector.

        Convenience method that handles all voice input types:
        - str: looks up voice by name
        - VoiceBlend: creates blended voice
        - np.ndarray: returns as-is (already a style vector)

        Args:
            voice: Voice specification (name, blend, or vector)

        Returns:
            Numpy array representing the voice style

        Raises:
            RuntimeError: If voices have not been loaded yet
            KeyError: If voice name is not found
        """
        if isinstance(voice, VoiceBlend):
            return self.create_blended_voice(voice)
        elif isinstance(voice, str):
            return self.get_voice_style(voice)
        else:
            # Already a style vector
            return voice

    def is_loaded(self) -> bool:
        """Check if voices have been loaded.

        Returns:
            True if voices are loaded, False otherwise
        """
        return self._voices_data is not None
