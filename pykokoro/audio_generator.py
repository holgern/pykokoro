"""Audio generation for PyKokoro."""

from __future__ import annotations

import logging
import re
from collections.abc import Callable
from typing import TYPE_CHECKING, Literal

import numpy as np
import onnxruntime as rt

from .constants import MAX_PHONEME_LENGTH, SAMPLE_RATE
from .prosody import apply_prosody
from .stages.g2p.kokorog2p import PhonemeSegment
from .tokenizer import Tokenizer
from .trim import trim as trim_audio
from .utils import generate_silence

if TYPE_CHECKING:
    from .short_sentence_handler import ShortSentenceConfig

logger = logging.getLogger(__name__)

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
    - Short sentence handling via phoneme pretext

    Args:
        session: ONNX Runtime inference session
        tokenizer: Tokenizer for phoneme<->token conversion
        model_source: Model source ('huggingface' or 'github')
        short_sentence_config: Configuration for short sentence handling
    """

    def __init__(
        self,
        session: rt.InferenceSession,
        tokenizer: Tokenizer,
        model_source: ModelSource = "huggingface",
        short_sentence_config: ShortSentenceConfig | None = None,
    ):
        """Initialize the audio generator."""
        self._session = session
        self._tokenizer = tokenizer
        self._model_source = model_source
        self._short_sentence_config = short_sentence_config

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
        # Ensure index doesn't exceed voice_style array bounds
        max_style_idx = voice_style.shape[0] - 1 if len(voice_style.shape) > 0 else 0
        style_idx = min(len(tokens), MAX_PHONEME_LENGTH - 1, max_style_idx)
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

        def token_len(text: str) -> int:
            if not text:
                return 0
            return len(self._tokenizer.tokenize(text))

        # Split on sentence-ending punctuation (., !, ?) while keeping them
        # Use lookbehind to split AFTER the punctuation
        sentences = re.split(r"(?<=[.!?])\s*", phonemes)

        batches: list[str] = []
        current = ""
        current_tokens = 0

        for sentence in sentences:
            sentence = sentence.strip()
            if not sentence:
                continue

            sentence_tokens = token_len(sentence)

            # If adding sentence would exceed limit, save current batch, start new
            if current and current_tokens + sentence_tokens > MAX_PHONEME_LENGTH:
                batches.append(current.strip())
                current = sentence
                current_tokens = sentence_tokens
            # If the sentence itself is too long, we need to split it further
            elif sentence_tokens > MAX_PHONEME_LENGTH:
                # Save current batch if any
                if current:
                    batches.append(current.strip())
                    current = ""
                    current_tokens = 0
                # Split long sentence on any punctuation or spaces
                words = re.split(r"([.,;:!?\s])", sentence)
                if len(words) == 1:
                    word_tokens = self._tokenizer.tokenize(words[0]) if words[0] else []
                    if len(word_tokens) > MAX_PHONEME_LENGTH:
                        for i in range(0, len(word_tokens), MAX_PHONEME_LENGTH):
                            chunk_tokens = word_tokens[i : i + MAX_PHONEME_LENGTH]
                            batches.append(self._tokenizer.detokenize(chunk_tokens))
                        continue
                for word in words:
                    if not word or word.isspace():
                        if current:
                            current += " "
                            current_tokens = token_len(current)
                        continue
                    word_tokens = self._tokenizer.tokenize(word)
                    if len(word_tokens) > MAX_PHONEME_LENGTH:
                        if current:
                            batches.append(current.strip())
                            current = ""
                            current_tokens = 0
                        for i in range(0, len(word_tokens), MAX_PHONEME_LENGTH):
                            chunk_tokens = word_tokens[i : i + MAX_PHONEME_LENGTH]
                            batches.append(self._tokenizer.detokenize(chunk_tokens))
                        continue
                    if current_tokens + len(word_tokens) > MAX_PHONEME_LENGTH:
                        if current:
                            batches.append(current.strip())
                        current = word
                        current_tokens = token_len(current)
                    else:
                        if current and not current.endswith(
                            (".", "!", "?", ",", ";", ":")
                        ):
                            current += " "
                        current += word
                        current_tokens = token_len(current)
            else:
                # Add sentence to current batch
                if current:
                    current += " "
                current += sentence
                current_tokens = token_len(current)

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

    def _resolve_segment_voice(
        self,
        segment: PhonemeSegment,
        default_voice_style: np.ndarray,
        voice_resolver: Callable[[str], np.ndarray] | None,
    ) -> np.ndarray:
        """Resolve voice style for a segment, checking SSMD voice metadata.

        Args:
            segment: Phoneme segment to process
            default_voice_style: Default voice style if no metadata present
            voice_resolver: Optional callback to resolve voice names

        Returns:
            Voice style array for this segment
        """
        # Use default voice by default
        segment_voice_style = default_voice_style

        # Check for SSMD voice metadata override
        if voice_resolver and segment.ssmd_metadata:
            voice_name = segment.ssmd_metadata.get("voice_name")
            if not voice_name:
                voice_name = segment.ssmd_metadata.get("voice")
            if voice_name:
                try:
                    segment_voice_style = voice_resolver(voice_name)
                except Exception as e:
                    logger.warning(
                        f"Failed to resolve voice '{voice_name}' for segment, "
                        f"using default voice: {e}"
                    )

        return segment_voice_style

    def _segment_token_length(self, segment: PhonemeSegment) -> int:
        if segment.tokens:
            return len(segment.tokens)
        if not segment.phonemes:
            return 0
        return len(self._tokenizer.tokenize(segment.phonemes))

    def _generate_single_segment_audio(
        self,
        segment: PhonemeSegment,
        voice_style: np.ndarray,
        speed: float,
        trim_silence: bool,
        enable_short_sentence_override: bool | None = None,
    ) -> list[np.ndarray]:
        """Generate audio for a single segment with phonemes.

        Handles splitting long phonemes and applying prosody modifications.

        Args:
            segment: Phoneme segment to process
            voice_style: Voice style to use
            speed: Speech speed multiplier
            trim_silence: Whether to trim silence from segment boundaries
            enable_short_sentence_override: Override short sentence handling.
                None (default): Use config setting
                True: Force enable short sentence handling
                False: Force disable short sentence handling

        Returns:
            List of audio arrays (may be multiple if phonemes were split)
        """
        from .short_sentence_handler import ShortSentenceConfig, is_segment_empty

        audio_parts: list[np.ndarray] = []

        # Skip empty phoneme segments
        if not segment.phonemes.strip():
            return audio_parts

        # Determine effective short sentence config with override support
        import dataclasses

        effective_config = self._short_sentence_config

        if enable_short_sentence_override is not None:
            # Override requested
            if enable_short_sentence_override:
                # Force enable: create config if None, or ensure enabled=True
                if effective_config is None:
                    effective_config = ShortSentenceConfig(enabled=True)
                else:
                    effective_config = dataclasses.replace(
                        effective_config, enabled=True
                    )
            else:
                # Force disable: ensure enabled=False if config exists
                if effective_config is not None:
                    effective_config = dataclasses.replace(
                        effective_config, enabled=False
                    )
                # If config is None and override is False, keep it None
        elif effective_config is None:
            # No override, no config: use default
            effective_config = ShortSentenceConfig()

        if effective_config and is_segment_empty(segment, effective_config):
            logger.debug(f"Skipping phoneme segment: '{segment.text[:50]}'")
            return audio_parts

        # Handle long phonemes by splitting
        if self._segment_token_length(segment) > MAX_PHONEME_LENGTH:
            batches = self.split_phonemes(segment.phonemes)
            for batch in batches:
                audio = self._generate_and_process_audio(
                    batch, voice_style, speed, trim_silence, segment
                )
                audio_parts.append(audio)
        else:
            audio = self._generate_and_process_audio(
                segment.phonemes, voice_style, speed, trim_silence, segment
            )
            audio_parts.append(audio)

        return audio_parts

    def _generate_and_process_audio(
        self,
        phonemes: str,
        voice_style: np.ndarray,
        speed: float,
        trim_silence: bool,
        segment: PhonemeSegment,
    ) -> np.ndarray:
        """Generate audio from phonemes and apply processing.

        Args:
            phonemes: Phoneme string to generate
            voice_style: Voice style to use
            speed: Speech speed multiplier
            trim_silence: Whether to trim silence
            segment: Original segment for prosody metadata

        Returns:
            Processed audio array
        """
        # Generate raw audio
        audio, _ = self.generate_from_phonemes(phonemes, voice_style, speed)

        # Trim silence if requested
        if trim_silence:
            audio, _ = trim_audio(audio)

        # Apply prosody modifications if present
        audio = self._apply_segment_prosody(audio, segment)

        return audio

    def generate_from_segments(
        self,
        segments: list[PhonemeSegment],
        voice_style: np.ndarray,
        speed: float,
        trim_silence: bool,
        voice_resolver: Callable[[str], np.ndarray] | None = None,
        enable_short_sentence_override: bool | None = None,
    ) -> np.ndarray:
        """Generate audio from list of PhonemeSegment instances.

        Unified audio generation method that handles:
        - Segments with phonemes (generate speech)
        - Empty segments (skip, only use pause_after)
        - Pause insertion based on pause_before and pause_after fields
        - Per-segment voice switching via SSMD voice metadata
        - Optional silence trimming
        - Per-call short sentence handling override

        Args:
            segments: List of PhonemeSegment instances
            voice_style: Default voice style vector (used when no voice metadata)
            speed: Speech speed multiplier
            trim_silence: Whether to trim silence from segment boundaries
            voice_resolver: Optional callback to resolve voice names to style vectors.
                Takes voice name (str) and returns voice style array.
                If provided and segment has voice metadata, uses per-segment voice.
            enable_short_sentence_override: Override short sentence handling.
                None (default): Use config setting
                True: Force enable short sentence handling
                False: Force disable short sentence handling

        Returns:
            Concatenated audio array
        """
        audio_parts = []

        for segment in segments:
            # Add pause before segment (if specified) - used for headings
            if segment.pause_before > 0:
                audio_parts.append(generate_silence(segment.pause_before, SAMPLE_RATE))

            # Resolve voice style for this segment (may use SSMD metadata)
            segment_voice_style = self._resolve_segment_voice(
                segment, voice_style, voice_resolver
            )

            # Generate audio for segment phonemes
            segment_audio = self._generate_single_segment_audio(
                segment,
                segment_voice_style,
                speed,
                trim_silence,
                enable_short_sentence_override,
            )
            audio_parts.extend(segment_audio)

            # Add pause after segment (if specified)
            if segment.pause_after > 0:
                audio_parts.append(generate_silence(segment.pause_after, SAMPLE_RATE))

        return (
            np.concatenate(audio_parts)
            if audio_parts
            else np.array([], dtype=np.float32)
        )

    def _apply_segment_prosody(
        self, audio: np.ndarray, segment: PhonemeSegment
    ) -> np.ndarray:
        """Apply prosody modifications from segment metadata to audio.

        Args:
            audio: Input audio array
            segment: PhonemeSegment with potential prosody metadata

        Returns:
            Audio with prosody modifications applied
        """
        if not segment.ssmd_metadata:
            return audio

        volume = segment.ssmd_metadata.get("prosody_volume")
        pitch = segment.ssmd_metadata.get("prosody_pitch")
        rate = segment.ssmd_metadata.get("prosody_rate")

        # Apply prosody if any prosody metadata is present
        if volume or pitch or rate:
            audio = apply_prosody(
                audio, SAMPLE_RATE, volume=volume, pitch=pitch, rate=rate
            )

        return audio

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
