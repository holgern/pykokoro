"""Audio generation for PyKokoro."""

import logging
import re
from typing import Literal

import numpy as np
import onnxruntime as rt

from .phonemes import PhonemeSegment
from .tokenizer import Tokenizer
from .trim import trim as trim_audio
from .utils import generate_silence

logger = logging.getLogger(__name__)

# Maximum phoneme length for a single inference
MAX_PHONEME_LENGTH = 510

# Sample rate for Kokoro models
SAMPLE_RATE = 24000

# Model source type
ModelSource = Literal["huggingface", "github"]


class AudioGenerator:
    """Generates audio from phonemes, tokens, and segments using ONNX inference.

    This class handles:
    - ONNX inference for single phoneme batches
    - Phoneme splitting for long inputs
    - Batch generation from phoneme lists
    - Segment-based generation with pause support
    - Token-to-audio generation

    Args:
        session: ONNX Runtime inference session
        tokenizer: Tokenizer for phoneme<->token conversion
        model_source: Model source ('huggingface' or 'github')
    """

    def __init__(
        self,
        session: rt.InferenceSession,
        tokenizer: Tokenizer,
        model_source: ModelSource = "huggingface",
    ):
        """Initialize the audio generator."""
        self._session = session
        self._tokenizer = tokenizer
        self._model_source = model_source

    def generate_from_phonemes(
        self,
        phonemes: str,
        voice_style: np.ndarray,
        speed: float,
    ) -> tuple[np.ndarray, int]:
        """Generate audio from a single phoneme batch.

        Core ONNX inference for a single phoneme batch.

        Args:
            phonemes: Phoneme string (will be truncated if > MAX_PHONEME_LENGTH)
            voice_style: Voice style vector
            speed: Speech speed multiplier

        Returns:
            Tuple of (audio samples, sample rate)
        """
        # Truncate phonemes if too long
        phonemes = phonemes[:MAX_PHONEME_LENGTH]
        tokens = self._tokenizer.tokenize(phonemes)

        # Get voice style for this token length (clamp to valid range)
        style_idx = min(len(tokens), MAX_PHONEME_LENGTH - 1)
        voice_style_indexed = voice_style[style_idx]

        # Pad tokens with start/end tokens
        tokens_padded = [[0, *tokens, 0]]

        # Check input names to determine model version
        input_names = [i.name for i in self._session.get_inputs()]

        # GitHub models (v1.0 and v1.1-zh) use "input_ids" and int32 speed
        # HuggingFace newer models also use "input_ids" but with float32 speed
        if "input_ids" in input_names:
            # Check if this is a GitHub model by checking model source
            if self._model_source == "github":
                # GitHub models: input_ids, style (float32), speed (int32)
                speed_int = max(1, int(round(speed)))
                inputs = {
                    "input_ids": np.array(tokens_padded, dtype=np.int64),
                    "style": np.array(voice_style_indexed, dtype=np.float32),
                    "speed": np.array([speed_int], dtype=np.int32),
                }
            else:
                # HuggingFace original format: input_ids, float32 speed
                inputs = {
                    "input_ids": tokens_padded,
                    "style": voice_style_indexed,
                    "speed": np.ones(1, dtype=np.float32) * speed,
                }
        else:
            # Original model format (uses "tokens" input, float speed)
            inputs = {
                "tokens": tokens_padded,
                "style": voice_style_indexed,
                "speed": np.ones(1, dtype=np.float32) * speed,
            }

        result = self._session.run(None, inputs)[0]
        audio = np.asarray(result).T
        # Ensure audio is 1D for compatibility with trim and other operations
        audio = np.squeeze(audio)
        return audio, SAMPLE_RATE

    def split_phonemes(self, phonemes: str) -> list[str]:
        """Split phonemes into batches at sentence-ending punctuation marks.

        Args:
            phonemes: Full phoneme string

        Returns:
            List of phoneme batches, each <= MAX_PHONEME_LENGTH
        """
        # Split on sentence-ending punctuation (., !, ?) while keeping them
        # Use lookbehind to split AFTER the punctuation
        sentences = re.split(r"(?<=[.!?])\s*", phonemes)

        batches = []
        current = ""

        for sentence in sentences:
            sentence = sentence.strip()
            if not sentence:
                continue

            # If adding sentence would exceed limit, save current batch, start new
            if current and len(current) + len(sentence) + 1 > MAX_PHONEME_LENGTH:
                batches.append(current.strip())
                current = sentence
            # If the sentence itself is too long, we need to split it further
            elif len(sentence) > MAX_PHONEME_LENGTH:
                # Save current batch if any
                if current:
                    batches.append(current.strip())
                    current = ""
                # Split long sentence on any punctuation or spaces
                words = re.split(r"([.,;:!?\s])", sentence)
                # If there's no punctuation or spaces, force chunk by character count
                if len(words) == 1 and len(words[0]) > MAX_PHONEME_LENGTH:
                    # Chunk the string at MAX_PHONEME_LENGTH boundaries
                    chunk = words[0]
                    while len(chunk) > MAX_PHONEME_LENGTH:
                        batches.append(chunk[:MAX_PHONEME_LENGTH])
                        chunk = chunk[MAX_PHONEME_LENGTH:]
                    if chunk:
                        current = chunk
                else:
                    for word in words:
                        if not word or word.isspace():
                            if current:
                                current += " "
                            continue
                        if len(current) + len(word) + 1 > MAX_PHONEME_LENGTH:
                            if current:
                                batches.append(current.strip())
                            current = word
                        else:
                            if current and not current.endswith(
                                (".", "!", "?", ",", ";", ":")
                            ):
                                current += " "
                            current += word
            else:
                # Add sentence to current batch
                if current:
                    current += " "
                current += sentence

        if current:
            batches.append(current.strip())

        return batches if batches else [phonemes]

    def generate_from_phoneme_batches(
        self,
        batches: list[str],
        voice_style: np.ndarray,
        speed: float,
        trim_silence: bool,
    ) -> np.ndarray:
        """Generate and concatenate audio from phoneme batches.

        Args:
            batches: List of phoneme strings (each <= MAX_PHONEME_LENGTH)
            voice_style: Voice style vector
            speed: Speech speed
            trim_silence: Whether to trim silence from each batch

        Returns:
            Concatenated audio array
        """
        audio_parts = []

        for batch in batches:
            audio, _ = self.generate_from_phonemes(batch, voice_style, speed)
            if trim_silence:
                audio, _ = trim_audio(audio)
            audio_parts.append(audio)

        return (
            np.concatenate(audio_parts)
            if audio_parts
            else np.array([], dtype=np.float32)
        )

    def generate_from_segments(
        self,
        segments: list[PhonemeSegment],
        voice_style: np.ndarray,
        speed: float,
        trim_silence: bool,
    ) -> np.ndarray:
        """Generate audio from list of PhonemeSegment instances.

        Unified audio generation method that handles:
        - Segments with phonemes (generate speech)
        - Empty segments (skip, only use pause_after)
        - Pause insertion based on pause_after field
        - Optional silence trimming

        Args:
            segments: List of PhonemeSegment instances
            voice_style: Voice style vector
            speed: Speech speed multiplier
            trim_silence: Whether to trim silence from segment boundaries

        Returns:
            Concatenated audio array
        """
        audio_parts = []

        for segment in segments:
            # Generate speech if phonemes present
            if segment.phonemes.strip():
                # Handle long phonemes by splitting
                if len(segment.phonemes) > MAX_PHONEME_LENGTH:
                    batches = self.split_phonemes(segment.phonemes)
                    for batch in batches:
                        audio, _ = self.generate_from_phonemes(
                            batch, voice_style, speed
                        )
                        if trim_silence:
                            audio, _ = trim_audio(audio)
                        audio_parts.append(audio)
                else:
                    audio, _ = self.generate_from_phonemes(
                        segment.phonemes, voice_style, speed
                    )
                    if trim_silence:
                        audio, _ = trim_audio(audio)
                    audio_parts.append(audio)

            # Add pause after segment (works for both empty and non-empty phonemes)
            if segment.pause_after > 0:
                audio_parts.append(generate_silence(segment.pause_after, SAMPLE_RATE))

        return (
            np.concatenate(audio_parts)
            if audio_parts
            else np.array([], dtype=np.float32)
        )

    def generate_from_tokens(
        self,
        tokens: list[int],
        voice_style: np.ndarray,
        speed: float,
    ) -> tuple[np.ndarray, int]:
        """Generate audio from token IDs directly.

        This provides the lowest-level interface, useful for pre-tokenized
        content and maximum control.

        Args:
            tokens: List of token IDs
            voice_style: Voice style vector
            speed: Speech speed

        Returns:
            Tuple of (audio samples as numpy array, sample rate)
        """
        # Detokenize to phonemes and generate audio
        phonemes = self._tokenizer.detokenize(tokens)

        # Split phonemes into batches and generate audio
        batches = self.split_phonemes(phonemes)
        audio = self.generate_from_phoneme_batches(
            batches, voice_style, speed, trim_silence=False
        )

        return audio, SAMPLE_RATE
