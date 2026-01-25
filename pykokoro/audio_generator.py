"""Audio generation for PyKokoro."""

from __future__ import annotations

import dataclasses
import logging
import re
from collections.abc import Callable
from typing import TYPE_CHECKING, Literal

import numpy as np
import onnxruntime as rt

from .constants import MAX_PHONEME_LENGTH, SAMPLE_RATE
from .prosody import apply_prosody
from .tokenizer import Tokenizer
from .trim import trim as trim_audio
from .types import PhonemeSegment
from .utils import generate_silence
from .voice_manager import normalize_voice_style

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
        self._uses_input_ids = any(
            input_meta.name == "input_ids" for input_meta in session.get_inputs()
        )

    def _tokenize_phonemes(self, phonemes: str) -> list[int]:
        trimmed = phonemes[:MAX_PHONEME_LENGTH]
        return self._tokenizer.tokenize(trimmed)

    def _select_voice_style(
        self, voice_style: np.ndarray, token_count: int
    ) -> np.ndarray:
        voice_style = normalize_voice_style(voice_style, expected_length=None)
        max_style_idx = voice_style.shape[0] - 1 if len(voice_style.shape) > 0 else 0
        style_idx = min(token_count, MAX_PHONEME_LENGTH - 1, max_style_idx)
        voice_style_indexed = voice_style[style_idx]
        if voice_style_indexed.ndim == 1:
            voice_style_indexed = voice_style_indexed[None, :]
        return voice_style_indexed

    @staticmethod
    def _pad_tokens(tokens: list[int]) -> list[list[int]]:
        return [[0, *tokens, 0]]

    def _float_speed_input(self, speed: float) -> np.ndarray:
        return np.ones(1, dtype=np.float32) * speed

    def _int_speed_input(self, speed: float) -> np.ndarray:
        speed_int = max(1, int(round(speed)))
        return np.array([speed_int], dtype=np.int32)

    def _build_onnx_inputs(
        self,
        tokens_padded: list[list[int]],
        voice_style: np.ndarray,
        speed: float,
    ) -> dict[str, np.ndarray | list[list[int]]]:
        if self._uses_input_ids:
            if self._model_source == "github":
                return {
                    "input_ids": np.array(tokens_padded, dtype=np.int64),
                    "style": np.array(voice_style, dtype=np.float32),
                    "speed": self._int_speed_input(speed),
                }
            return {
                "input_ids": tokens_padded,
                "style": voice_style,
                "speed": self._float_speed_input(speed),
            }
        return {
            "tokens": tokens_padded,
            "style": voice_style,
            "speed": self._float_speed_input(speed),
        }

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
        tokens = self._tokenize_phonemes(phonemes)
        voice_style_indexed = self._select_voice_style(voice_style, len(tokens))
        tokens_padded = self._pad_tokens(tokens)
        inputs = self._build_onnx_inputs(tokens_padded, voice_style_indexed, speed)
        result = self._session.run(None, inputs)[0]
        audio = np.asarray(result).T
        audio = np.squeeze(audio)
        return audio, SAMPLE_RATE

    def split_phonemes(self, phonemes: str) -> list[str]:  # noqa: C901
        """Split phonemes into batches at sentence-ending punctuation marks.

        Args:
            phonemes: Full phoneme string

        Returns:
            List of phoneme batches, each <= MAX_PHONEME_LENGTH
        """

        batches: list[str] = []
        current = ""
        current_tokens = 0

        def token_len(text: str) -> int:
            if not text:
                return 0
            return len(self._tokenizer.tokenize(text))

        def append_batch(text: str) -> None:
            if text:
                batches.append(text.strip())

        def split_long_sentence(sentence: str) -> bool:
            nonlocal current, current_tokens
            if current:
                append_batch(current)
                current = ""
                current_tokens = 0
            words = re.split(r"([.,;:!?\s])", sentence)
            if len(words) == 1:
                word_tokens = self._tokenizer.tokenize(words[0]) if words[0] else []
                if len(word_tokens) > MAX_PHONEME_LENGTH:
                    for i in range(0, len(word_tokens), MAX_PHONEME_LENGTH):
                        chunk_tokens = word_tokens[i : i + MAX_PHONEME_LENGTH]
                        batches.append(self._tokenizer.detokenize(chunk_tokens))
                    return True
            for word in words:
                if not word or word.isspace():
                    if current:
                        current += " "
                        current_tokens = token_len(current)
                    continue
                word_tokens = self._tokenizer.tokenize(word)
                if len(word_tokens) > MAX_PHONEME_LENGTH:
                    if current:
                        append_batch(current)
                        current = ""
                        current_tokens = 0
                    for i in range(0, len(word_tokens), MAX_PHONEME_LENGTH):
                        chunk_tokens = word_tokens[i : i + MAX_PHONEME_LENGTH]
                        batches.append(self._tokenizer.detokenize(chunk_tokens))
                    continue
                if current_tokens + len(word_tokens) > MAX_PHONEME_LENGTH:
                    if current:
                        append_batch(current)
                    current = word
                    current_tokens = token_len(current)
                else:
                    if current and not current.endswith((".", "!", "?", ",", ";", ":")):
                        current += " "
                    current += word
                    current_tokens = token_len(current)
            return False

        # Split on sentence-ending punctuation (., !, ?) while keeping them
        # Use lookbehind to split AFTER the punctuation
        sentences = re.split(r"(?<=[.!?])\s*", phonemes)

        for sentence in sentences:
            sentence = sentence.strip()
            if not sentence:
                continue

            sentence_tokens = token_len(sentence)

            # If adding sentence would exceed limit, save current batch, start new
            if current and current_tokens + sentence_tokens > MAX_PHONEME_LENGTH:
                append_batch(current)
                current = sentence
                current_tokens = sentence_tokens
            # If the sentence itself is too long, we need to split it further
            elif sentence_tokens > MAX_PHONEME_LENGTH:
                if split_long_sentence(sentence):
                    continue
            else:
                # Add sentence to current batch
                if current:
                    current += " "
                current += sentence
                current_tokens = token_len(current)

        if current:
            append_batch(current)

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

    def _resolve_short_sentence_config(
        self, enable_short_sentence_override: bool | None
    ) -> ShortSentenceConfig | None:
        from .short_sentence_handler import ShortSentenceConfig

        effective_config = self._short_sentence_config

        if enable_short_sentence_override is not None:
            if enable_short_sentence_override:
                if effective_config is None:
                    effective_config = ShortSentenceConfig(enabled=True)
                else:
                    effective_config = dataclasses.replace(
                        effective_config, enabled=True
                    )
            else:
                if effective_config is not None:
                    effective_config = dataclasses.replace(
                        effective_config, enabled=False
                    )
        elif effective_config is None:
            effective_config = ShortSentenceConfig()

        return effective_config

    def _preprocess_segments(
        self,
        segments: list[PhonemeSegment],
        enable_short_sentence_override: bool | None,
    ) -> list[PhonemeSegment]:
        from .short_sentence_handler import is_segment_empty

        effective_config = self._resolve_short_sentence_config(
            enable_short_sentence_override
        )
        processed: list[PhonemeSegment] = []

        for segment in segments:
            phonemes = segment.phonemes or ""
            tokens = self._tokenizer.tokenize(phonemes) if phonemes.strip() else []
            skip_audio = False

            if effective_config and is_segment_empty(segment, effective_config):
                logger.debug(f"Skipping phoneme segment: '{segment.text[:50]}'")
                skip_audio = True

            if skip_audio or not phonemes.strip():
                processed.append(
                    dataclasses.replace(
                        segment,
                        phonemes="",
                        tokens=[],
                        raw_audio=None,
                        processed_audio=None,
                    )
                )
                continue

            if len(tokens) > MAX_PHONEME_LENGTH:
                batches = [
                    tokens[i : i + MAX_PHONEME_LENGTH]
                    for i in range(0, len(tokens), MAX_PHONEME_LENGTH)
                ]
                total_batches = len(batches)
                for idx, batch_tokens in enumerate(batches):
                    batch_phonemes = self._tokenizer.detokenize(batch_tokens)
                    processed.append(
                        dataclasses.replace(
                            segment,
                            id=f"{segment.id}_ph{idx}",
                            phoneme_id=idx,
                            phonemes=batch_phonemes,
                            tokens=list(batch_tokens),
                            pause_before=segment.pause_before if idx == 0 else 0.0,
                            pause_after=(
                                segment.pause_after if idx == total_batches - 1 else 0.0
                            ),
                            raw_audio=None,
                            processed_audio=None,
                        )
                    )
            else:
                processed.append(
                    dataclasses.replace(
                        segment,
                        tokens=tokens,
                        pause_before=segment.pause_before,
                        pause_after=segment.pause_after,
                        raw_audio=None,
                        processed_audio=None,
                    )
                )

        return processed

    def _generate_raw_audio_segments(
        self,
        segments: list[PhonemeSegment],
        voice_style: np.ndarray,
        speed: float,
        voice_resolver: Callable[[str], np.ndarray] | None,
    ) -> list[PhonemeSegment]:
        for segment in segments:
            if not segment.phonemes.strip():
                segment.raw_audio = None
                continue

            segment_voice_style = self._resolve_segment_voice(
                segment, voice_style, voice_resolver
            )
            audio, _ = self.generate_from_phonemes(
                segment.phonemes, segment_voice_style, speed
            )
            segment.raw_audio = audio

        return segments

    def _postprocess_audio_segments(
        self, segments: list[PhonemeSegment], trim_silence: bool
    ) -> list[PhonemeSegment]:
        for segment in segments:
            if segment.raw_audio is None:
                segment.processed_audio = None
                continue

            if not trim_silence and not segment.ssmd_metadata:
                segment.processed_audio = segment.raw_audio
                continue

            audio = segment.raw_audio
            if trim_silence:
                audio, _ = trim_audio(audio)
            segment.processed_audio = self._apply_segment_prosody(audio, segment)

        return segments

    def _concatenate_audio_segments(self, segments: list[PhonemeSegment]) -> np.ndarray:
        audio_parts: list[np.ndarray] = []

        for segment in segments:
            if segment.pause_before > 0:
                audio_parts.append(generate_silence(segment.pause_before, SAMPLE_RATE))
            if segment.processed_audio is not None:
                audio_parts.append(segment.processed_audio)
            if segment.pause_after > 0:
                audio_parts.append(generate_silence(segment.pause_after, SAMPLE_RATE))

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
        preprocessed = self._preprocess_segments(
            segments, enable_short_sentence_override
        )
        generated = self._generate_raw_audio_segments(
            preprocessed, voice_style, speed, voice_resolver
        )
        processed = self._postprocess_audio_segments(generated, trim_silence)
        return self._concatenate_audio_segments(processed)

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
