"""ONNX backend for pykokoro - native ONNX TTS without external dependencies."""

import asyncio
import io
import logging
import os
import re
import sqlite3
from collections.abc import AsyncGenerator, Callable, Generator
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal, Optional

import numpy as np
import onnxruntime as rt
from huggingface_hub import hf_hub_download

from .tokenizer import EspeakConfig, Tokenizer, TokenizerConfig
from .trim import trim as trim_audio
from .utils import get_user_cache_path

# Logger for debugging
logger = logging.getLogger(__name__)

if TYPE_CHECKING:
    from .phonemes import PhonemeSegment

# Maximum phoneme length for a single inference
MAX_PHONEME_LENGTH = 510

# Sample rate for Kokoro models
SAMPLE_RATE = 24000

# Model quality type
ModelQuality = Literal[
    "fp32", "fp16", "q8", "q8f16", "q4", "q4f16", "uint8", "uint8f16"
]
DEFAULT_MODEL_QUALITY: ModelQuality = "fp32"

# Provider type
ProviderType = Literal["auto", "cpu", "cuda", "openvino", "directml", "coreml"]

# Quality to filename mapping
MODEL_QUALITY_FILES: dict[str, str] = {
    "fp32": "model.onnx",
    "fp16": "model_fp16.onnx",
    "q8": "model_quantized.onnx",
    "q8f16": "model_q8f16.onnx",
    "q4": "model_q4.onnx",
    "q4f16": "model_q4f16.onnx",
    "uint8": "model_uint8.onnx",
    "uint8f16": "model_uint8f16.onnx",
}

# URLs for model files (Hugging Face)
HF_REPO_ID = "onnx-community/Kokoro-82M-v1.0-ONNX"
HF_MODEL_SUBFOLDER = "onnx"
HF_VOICES_SUBFOLDER = "voices"
HF_CONFIG_FILENAME = "config.json"

# All available voice names
VOICE_NAMES = [
    "af",
    "af_alloy",
    "af_aoede",
    "af_bella",
    "af_heart",
    "af_jessica",
    "af_kore",
    "af_nicole",
    "af_nova",
    "af_river",
    "af_sarah",
    "af_sky",
    "am_adam",
    "am_echo",
    "am_eric",
    "am_fenrir",
    "am_liam",
    "am_michael",
    "am_onyx",
    "am_puck",
    "am_santa",
    "bf_alice",
    "bf_emma",
    "bf_isabella",
    "bf_lily",
    "bm_daniel",
    "bm_fable",
    "bm_george",
    "bm_lewis",
    "ef_dora",
    "em_alex",
    "em_santa",
    "ff_siwis",
    "hf_alpha",
    "hf_beta",
    "hm_omega",
    "hm_psi",
    "if_sara",
    "im_nicola",
    "jf_alpha",
    "jf_gongitsune",
    "jf_nezumi",
    "jf_tebukuro",
    "jm_kumo",
    "pf_dora",
    "pm_alex",
    "pm_santa",
    "zf_xiaobei",
    "zf_xiaoni",
    "zf_xiaoxiao",
    "zm_yunjian",
    "zm_yunxi",
    "zm_yunxia",
    "zm_yunyang",
]


@dataclass
class VoiceBlend:
    """Represents a blend of multiple voices."""

    voices: list[tuple[str, float]]  # List of (voice_name, weight) tuples

    @classmethod
    def parse(cls, blend_str: str) -> "VoiceBlend":
        """
        Parse a voice blend string.

        Format: "voice1:weight1,voice2:weight2" or "voice1:50,voice2:50"
        Weights should sum to 100 (percentages).

        Example: "af_nicole:50,am_michael:50"
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


# =============================================================================
# Path helper functions
# =============================================================================


def get_model_dir() -> Path:
    """Get the directory for storing ONNX model files."""
    return get_user_cache_path("models") / "onnx"


def get_voices_dir() -> Path:
    """Get the directory for storing voice files."""
    return get_user_cache_path("voices")


def get_config_path() -> Path:
    """Get the path to the cached config.json."""
    return get_user_cache_path() / "config.json"


def get_voices_bin_path() -> Path:
    """Get the path to the combined voices.bin.npz file."""
    return get_user_cache_path() / "voices.bin.npz"


def get_model_filename(quality: ModelQuality = DEFAULT_MODEL_QUALITY) -> str:
    """Get the model filename for a quality level."""
    return MODEL_QUALITY_FILES[quality]


def get_model_path(quality: ModelQuality = DEFAULT_MODEL_QUALITY) -> Path:
    """Get the full path to a model file for a given quality."""
    filename = get_model_filename(quality)
    return get_model_dir() / filename


def get_voice_path(voice_name: str) -> Path:
    """Get the full path to an individual voice file."""
    return get_voices_dir() / f"{voice_name}.bin"


# =============================================================================
# Download check functions
# =============================================================================


def is_config_downloaded() -> bool:
    """Check if config.json is downloaded."""
    config_path = get_config_path()
    return config_path.exists() and config_path.stat().st_size > 0


def is_model_downloaded(quality: ModelQuality = DEFAULT_MODEL_QUALITY) -> bool:
    """Check if a model file is already downloaded for a given quality."""
    model_path = get_model_path(quality)
    return model_path.exists() and model_path.stat().st_size > 0


def is_voice_downloaded(voice_name: str) -> bool:
    """Check if an individual voice file is already downloaded."""
    voice_path = get_voice_path(voice_name)
    return voice_path.exists() and voice_path.stat().st_size > 0


def are_voices_downloaded() -> bool:
    """Check if the combined voices.bin file exists."""
    voices_bin_path = get_voices_bin_path()
    return voices_bin_path.exists() and voices_bin_path.stat().st_size > 0


def are_models_downloaded(quality: ModelQuality = DEFAULT_MODEL_QUALITY) -> bool:
    """Check if model, config, and voices.bin are downloaded."""
    return (
        is_config_downloaded()
        and is_model_downloaded(quality)
        and are_voices_downloaded()
    )


# =============================================================================
# Download functions
# =============================================================================


def _download_from_hf(
    repo_id: str,
    filename: str,
    subfolder: str | None = None,
    local_dir: Path | None = None,
    force: bool = False,
) -> Path:
    """
    Download a file from Hugging Face Hub.

    Args:
        repo_id: Hugging Face repository ID
        filename: File to download
        subfolder: Subfolder in the repository
        local_dir: Local directory to save to
        force: Force re-download even if file exists

    Returns:
        Path to the downloaded file
    """
    # Use hf_hub_download to download the file
    # It handles caching automatically
    downloaded_path = hf_hub_download(
        repo_id=repo_id,
        filename=filename,
        subfolder=subfolder,
        local_dir=str(local_dir) if local_dir else None,
        force_download=force,
    )
    return Path(downloaded_path)


def download_config(
    force: bool = False,
) -> Path:
    """
    Download config.json from Hugging Face.

    Args:
        force: Force re-download even if file exists

    Returns:
        Path to the downloaded config file
    """
    return _download_from_hf(
        repo_id="hexgrad/Kokoro-82M",
        filename=HF_CONFIG_FILENAME,
        local_dir=get_user_cache_path(),
        force=force,
    )


def download_model(
    quality: ModelQuality = DEFAULT_MODEL_QUALITY,
    force: bool = False,
) -> Path:
    """
    Download a model file for the specified quality.

    Args:
        quality: Model quality/quantization level
        force: Force re-download even if file exists

    Returns:
        Path to the downloaded model file
    """
    filename = get_model_filename(quality)
    return _download_from_hf(
        repo_id=HF_REPO_ID,
        filename=filename,
        subfolder=HF_MODEL_SUBFOLDER,
        local_dir=get_model_dir(),
        force=force,
    )


def download_voice(
    voice_name: str,
    force: bool = False,
) -> Path:
    """
    Download a single voice file.

    Args:
        voice_name: Name of the voice to download
        force: Force re-download even if file exists

    Returns:
        Path to the downloaded voice file
    """
    filename = f"{voice_name}.bin"
    return _download_from_hf(
        repo_id=HF_REPO_ID,
        filename=filename,
        subfolder=HF_VOICES_SUBFOLDER,
        local_dir=get_voices_dir(),
        force=force,
    )


def download_all_voices(
    progress_callback: Callable[[str, int, int], None] | None = None,
    force: bool = False,
) -> Path:
    """
    Download all voice files and config, then combine into a single voices.bin file.

    Downloads individual voice .bin files and config.json from Hugging Face,
    loads them, and saves them as a combined numpy archive (voices.bin.npz) for
    efficient loading.

    Args:
        progress_callback: Optional callback (voice_name, current_index, total_count)
        force: Force re-download even if files exist

    Returns:
        Path to the combined voices.bin.npz file
    """
    voices_bin_path = get_voices_bin_path()

    # If voices.bin already exists and not forcing, skip download
    if voices_bin_path.exists() and not force:
        return voices_bin_path

    voices_dir = get_voices_dir()
    voices_dir.mkdir(parents=True, exist_ok=True)

    # Download config.json first
    if progress_callback:
        progress_callback("config.json", 0, len(VOICE_NAMES) + 1)
    download_config(force=force)

    total = len(VOICE_NAMES)
    voices: dict[str, np.ndarray] = {}

    for idx, voice_name in enumerate(VOICE_NAMES):
        if progress_callback:
            progress_callback(voice_name, idx + 1, total)

        # Download individual voice file
        voice_path = download_voice(voice_name, force=force)

        # Load the voice data from .bin file
        voice_data = np.fromfile(voice_path, dtype=np.float32).reshape(-1, 1, 256)
        voices[voice_name] = voice_data

    # Save all voices to a single .npz file using np.savez
    voices_bin_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(str(voices_bin_path), **voices)  # type: ignore[call-overload]

    return voices_bin_path


def download_all_models(
    quality: ModelQuality = DEFAULT_MODEL_QUALITY,
    progress_callback: Callable[[str, int, int], None] | None = None,
    force: bool = False,
) -> dict[str, Path]:
    """
    Download config, model, and all voice files.

    Args:
        quality: Model quality/quantization level
        progress_callback: Optional callback (filename, current, total)
        force: Force re-download even if files exist

    Returns:
        Dict mapping filename to path
    """
    paths: dict[str, Path] = {}

    # Download config
    if progress_callback:
        progress_callback("config.json", 0, 0)
    paths["config.json"] = download_config(force=force)

    # Download model
    model_filename = get_model_filename(quality)
    if progress_callback:
        progress_callback(model_filename, 0, 0)
    paths[model_filename] = download_model(quality, force=force)

    # Download all voices and combine into voices.bin.npz
    paths["voices.bin.npz"] = download_all_voices(progress_callback, force)

    return paths


class Kokoro:
    """
    Native ONNX backend for TTS generation.

    This class provides direct ONNX inference without external dependencies.
    Includes embedded tokenizer for phoneme/token-based generation.
    """

    def __init__(
        self,
        model_path: Path | None = None,
        voices_path: Path | None = None,
        use_gpu: bool = False,
        provider: ProviderType | None = None,
        vocab_version: str = "v1.0",
        espeak_config: EspeakConfig | None = None,
        tokenizer_config: Optional["TokenizerConfig"] = None,
        model_quality: ModelQuality | None = None,
    ) -> None:
        """
        Initialize the Kokoro ONNX backend.

        Args:
            model_path: Path to the ONNX model file (auto-downloaded if None)
            voices_path: Path to the voices.bin file (auto-downloaded if None)
            use_gpu: Deprecated. Use provider parameter instead.
                Legacy GPU flag for backward compatibility.
            provider: Execution provider for ONNX Runtime. Options:
                "auto" (auto-select best), "cpu", "cuda" (NVIDIA),
                "openvino" (Intel), "directml" (Windows), "coreml" (macOS)
            vocab_version: Vocabulary version for tokenizer
            espeak_config: Optional espeak-ng configuration
                (deprecated, use tokenizer_config)
            tokenizer_config: Optional tokenizer configuration
                (for mixed-language support)
            model_quality: Model quality/quantization level (default from config)
        """
        self._session: rt.InferenceSession | None = None
        self._voices_data: dict[str, np.ndarray] | None = None
        self._np = np

        # Deprecation warning for use_gpu
        if use_gpu:
            logger.warning(
                "The 'use_gpu' parameter is deprecated and will be removed in a "
                "future version. Use 'provider' parameter instead. "
                "Example: Kokoro(provider='cuda') or Kokoro(provider='auto')"
            )

        self._use_gpu = use_gpu
        self._provider: ProviderType | None = provider

        # Resolve model quality from config if not specified
        resolved_quality: ModelQuality = DEFAULT_MODEL_QUALITY
        if model_quality is not None:
            resolved_quality = model_quality
        else:
            from .utils import load_config

            cfg = load_config()
            quality_from_cfg = cfg.get("model_quality", DEFAULT_MODEL_QUALITY)
            # Validate it's a valid quality option and cast to ModelQuality
            if quality_from_cfg in MODEL_QUALITY_FILES:
                resolved_quality = str(quality_from_cfg)  # type: ignore[assignment]
        self._model_quality: ModelQuality = resolved_quality

        # Resolve paths
        if model_path is None:
            model_path = get_model_path(self._model_quality)
        if voices_path is None:
            voices_path = get_voices_bin_path()

        self._model_path = model_path
        self._voices_path = voices_path

        # Voice database connection (for kokovoicelab integration)
        self._voice_db: sqlite3.Connection | None = None

        # Tokenizer for phoneme-based generation
        self._tokenizer: Tokenizer | None = None
        self._vocab_version = vocab_version
        self._espeak_config = espeak_config
        self._tokenizer_config = tokenizer_config

    @property
    def tokenizer(self) -> Tokenizer:
        """Get the tokenizer instance (lazily initialized)."""
        if self._tokenizer is None:
            self._tokenizer = Tokenizer(
                config=self._tokenizer_config,
                espeak_config=self._espeak_config,
                vocab_version=self._vocab_version,
            )
        return self._tokenizer

    def _ensure_models(self) -> None:
        """Ensure model and voice files are downloaded."""
        if not self._model_path.exists():
            download_model(self._model_quality)
        if not self._voices_path.exists():
            download_all_voices()
        if not is_config_downloaded():
            download_config()

    def _select_providers(
        self,
        provider: ProviderType | None,
        use_gpu: bool,
    ) -> list[str]:
        """
        Select ONNX Runtime execution providers based on preference.

        Args:
            provider: Explicit provider ('auto', 'cpu', 'cuda', 'openvino', etc.)
            use_gpu: Legacy GPU flag (for backward compatibility)

        Returns:
            List of providers in priority order

        Raises:
            RuntimeError: If requested provider is not available
            ValueError: If provider name is invalid
        """
        available = rt.get_available_providers()

        # Environment variable override (highest priority)
        env_provider = os.getenv("ONNX_PROVIDER")
        if env_provider:
            logger.info(f"Using provider from ONNX_PROVIDER env: {env_provider}")
            return [env_provider, "CPUExecutionProvider"]

        # Auto-selection logic
        if provider == "auto" or (provider is None and use_gpu):
            # Priority: CUDA > OpenVINO > CoreML > DirectML
            for prov in [
                "CUDAExecutionProvider",
                "OpenVINOExecutionProvider",
                "CoreMLExecutionProvider",
                "DmlExecutionProvider",
            ]:
                if prov in available:
                    logger.info(f"Auto-selected provider: {prov}")
                    return [prov, "CPUExecutionProvider"]
            logger.info("Auto-selection: No accelerators found, using CPU")
            return ["CPUExecutionProvider"]

        # Default to CPU if no provider specified and use_gpu=False
        if provider is None:
            logger.info("Using CPU provider")
            return ["CPUExecutionProvider"]

        # Explicit provider selection
        provider_map = {
            "cpu": "CPUExecutionProvider",
            "cuda": "CUDAExecutionProvider",
            "openvino": "OpenVINOExecutionProvider",
            "directml": "DmlExecutionProvider",
            "coreml": "CoreMLExecutionProvider",
        }

        selected = provider_map.get(provider.lower())
        if not selected:
            raise ValueError(
                f"Unknown provider: {provider}. "
                f"Valid options: {list(provider_map.keys())}"
            )

        if selected not in available:
            # Provide helpful installation message
            install_hints = {
                "CUDAExecutionProvider": "pip install pykokoro[gpu]",
                "OpenVINOExecutionProvider": "pip install pykokoro[openvino]",
                "DmlExecutionProvider": "pip install pykokoro[directml]",
                "CoreMLExecutionProvider": "pip install pykokoro[coreml]",
            }
            hint = install_hints.get(
                selected, f"install the required package for {selected}"
            )
            raise RuntimeError(
                f"{provider.upper()} provider requested but not available.\n"
                f"Install with: {hint}\n"
                f"Available providers: {available}"
            )

        logger.info(f"Using explicitly selected provider: {selected}")
        return [selected, "CPUExecutionProvider"]

    def _init_kokoro(self) -> None:
        """Initialize the ONNX session and load voices."""
        if self._session is not None:
            return

        self._ensure_models()

        # Select execution providers
        providers = self._select_providers(self._provider, self._use_gpu)

        # Try to load ONNX model with automatic fallback
        # If the primary provider fails and we're in auto mode, try CPU
        session_loaded = False
        last_error = None

        for attempt, provider_list in enumerate([providers, ["CPUExecutionProvider"]]):
            # Skip second attempt if we already tried CPU or
            # if explicit provider was requested
            if attempt == 1:
                if providers == ["CPUExecutionProvider"]:
                    break  # Already tried CPU
                if self._provider and self._provider != "auto":
                    break  # User explicitly requested a provider, don't fallback
                if providers[0] == "CPUExecutionProvider":
                    break  # Primary was already CPU

            try:
                self._session = rt.InferenceSession(
                    str(self._model_path), providers=provider_list
                )
                session_loaded = True

                # Log what was actually loaded
                actual_providers = self._session.get_providers()
                logger.info(f"Loaded ONNX session with providers: {actual_providers}")

                # Warn if we had to fallback
                if attempt == 1:
                    failed_provider = providers[0]
                    logger.warning(
                        f"Failed to load model with {failed_provider}, "
                        f"fell back to CPU. Error: {last_error}"
                    )

                break

            except Exception as e:
                last_error = str(e)
                if attempt == 0:
                    # First attempt failed, will try fallback
                    logger.debug(f"Provider {provider_list[0]} failed: {e}")
                    continue
                else:
                    # Fallback also failed, re-raise
                    raise

        if not session_loaded:
            raise RuntimeError(
                f"Failed to initialize ONNX session with providers {providers}. "
                f"Last error: {last_error}"
            )

        # Load voices (numpy archive with voice style vectors)
        self._voices_data = dict(np.load(str(self._voices_path), allow_pickle=True))

    def _create_audio_internal(
        self, phonemes: str, voice: np.ndarray, speed: float, new_format: bool = True
    ) -> tuple[np.ndarray, int]:
        """
        Core ONNX inference for a single phoneme batch.

        Args:
            phonemes: Phoneme string (will be truncated if > MAX_PHONEME_LENGTH)
            voice: Voice style vector
            speed: Speech speed multiplier

        Returns:
            Tuple of (audio samples, sample rate)
        """
        assert self._session is not None

        # Truncate phonemes if too long
        phonemes = phonemes[:MAX_PHONEME_LENGTH]
        tokens = self.tokenizer.tokenize(phonemes)

        # Get voice style for this token length (clamp to valid range)
        style_idx = min(len(tokens), MAX_PHONEME_LENGTH - 1)
        voice_style = voice[style_idx]

        # Pad tokens with start/end tokens
        tokens_padded = [[0, *tokens, 0]]

        # Check input names to determine model version
        input_names = [i.name for i in self._session.get_inputs()]
        if "input_ids" in input_names and not new_format:
            # Newer model format (exported with input_ids, expects int32 speed)
            # Speed is typically 1 for normal speed, convert float to int
            speed_int = max(1, int(round(speed)))
            inputs = {
                "input_ids": np.array(tokens_padded, dtype=np.int64),
                "style": np.array(voice_style, dtype=np.float32),
                "speed": np.array([speed_int], dtype=np.int32),
            }
        elif "input_ids" in input_names and new_format:
            # Original model format (kokoro-onnx release model, uses float speed)
            inputs = {
                "input_ids": tokens_padded,
                "style": voice_style,
                "speed": np.ones(1, dtype=np.float32) * speed,
            }

        else:
            # Original model format (kokoro-onnx release model, uses float speed)
            inputs = {
                "tokens": tokens_padded,
                "style": voice_style,
                "speed": np.ones(1, dtype=np.float32) * speed,
            }

        result = self._session.run(None, inputs)[0]
        if new_format:
            audio: np.ndarray = np.asarray(result).T
        else:
            audio: np.ndarray = np.asarray(result)
        # Ensure audio is 1D for compatibility with trim and other operations
        audio = np.squeeze(audio)
        return audio, SAMPLE_RATE

    def _split_phonemes(self, phonemes: str) -> list[str]:
        """
        Split phonemes into batches at sentence-ending punctuation marks.

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

    def get_voices(self) -> list[str]:
        """Get list of available voice names."""
        self._init_kokoro()
        assert self._voices_data is not None
        return list(sorted(self._voices_data.keys()))

    def get_voice_style(self, voice_name: str) -> np.ndarray:
        """
        Get the style vector for a voice.

        Args:
            voice_name: Name of the voice

        Returns:
            Numpy array representing the voice style
        """
        self._init_kokoro()
        assert self._voices_data is not None
        return self._voices_data[voice_name]

    def create_blended_voice(self, blend: VoiceBlend) -> np.ndarray:
        """
        Create a blended voice from multiple voices.

        Args:
            blend: VoiceBlend object specifying voices and weights

        Returns:
            Numpy array representing the blended voice style
        """
        self._init_kokoro()

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

    def _generate_from_phoneme_batches(
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
            audio, _ = self._create_audio_internal(batch, voice_style, speed)
            if trim_silence:
                audio, _ = trim_audio(audio)
            audio_parts.append(audio)

        return (
            np.concatenate(audio_parts)
            if audio_parts
            else np.array([], dtype=np.float32)
        )

    def _process_with_split_mode(
        self,
        text: str,
        voice_style: np.ndarray,
        speed: float,
        lang: str,
        split_mode: str,
        trim_silence: bool,
    ) -> np.ndarray:
        """Process text using sentence/paragraph splitting for better prosody.

        Args:
            text: Input text
            voice_style: Voice style vector
            speed: Speech speed
            lang: Language code
            split_mode: Split mode ('paragraph', 'sentence', 'clause')
            trim_silence: Whether to trim silence from segments

        Returns:
            Generated audio array
        """
        # Validate spaCy requirement
        if split_mode in ["sentence", "clause"]:
            try:
                import spacy  # noqa: F401
            except ImportError as err:
                raise ImportError(
                    f"spaCy is required for split_mode='{split_mode}'. "
                    "Install with: pip install spacy && "
                    "python -m spacy download en_core_web_sm"
                ) from err

        from .phonemes import split_and_phonemize_text

        # Split text directly into segments
        # split_and_phonemize_text() uses cascading split modes to ensure
        # all segments stay within max_phoneme_length (510)
        segments = split_and_phonemize_text(
            text,
            tokenizer=self.tokenizer,
            lang=lang,
            split_mode=split_mode,
        )

        # Generate audio for each segment
        segment_parts = []
        for segment in segments:
            # segment.phonemes is guaranteed to be <= 510 by cascading logic
            audio, _ = self._create_audio_internal(segment.phonemes, voice_style, speed)
            if trim_silence:
                audio, _ = trim_audio(audio)
            segment_parts.append(audio)

        return (
            np.concatenate(segment_parts)
            if segment_parts
            else np.array([], dtype=np.float32)
        )

    def _process_text_segment(
        self,
        text: str,
        voice_style: np.ndarray,
        speed: float,
        lang: str,
        split_mode: str | None,
        trim_silence: bool,
    ) -> np.ndarray:
        """Process a text segment into audio, handling splitting automatically.

        Args:
            text: Input text segment
            voice_style: Voice style vector
            speed: Speech speed
            lang: Language code
            split_mode: Optional split mode for better prosody
            trim_silence: Whether to trim silence

        Returns:
            Generated audio array
        """
        if split_mode is not None:
            # Use text-level splitting for better prosody
            return self._process_with_split_mode(
                text, voice_style, speed, lang, split_mode, trim_silence
            )
        else:
            # Simple approach: phonemize → check length → split if needed
            phonemes = self.tokenizer.phonemize(text, lang=lang)

            # Check if phonemes exceed limit
            if len(phonemes) <= MAX_PHONEME_LENGTH:
                # Single batch
                audio, _ = self._create_audio_internal(phonemes, voice_style, speed)
                if trim_silence:
                    audio, _ = trim_audio(audio)
                return audio
            else:
                # Need to split phonemes
                batches = self._split_phonemes(phonemes)
                return self._generate_from_phoneme_batches(
                    batches, voice_style, speed, trim_silence
                )

    def create(
        self,
        text: str,
        voice: str | np.ndarray | VoiceBlend,
        speed: float = 1.0,
        lang: str = "en-us",
        is_phonemes: bool = False,
        enable_pauses: bool = False,
        pause_short: float = 0.3,
        pause_medium: float = 0.6,
        pause_long: float = 1.0,
        split_mode: str | None = None,
        trim_silence: bool = True,
    ) -> tuple[np.ndarray, int]:
        """
        Generate audio from text or phonemes.

        Args:
            text: Text to synthesize (or phonemes if is_phonemes=True)
            voice: Voice name, style vector, or VoiceBlend
            speed: Speech speed (1.0 = normal)
            lang: Language code (e.g., 'en-us', 'en-gb', 'es', 'fr')
            is_phonemes: If True, treat 'text' as phonemes instead of text
            enable_pauses: If True, process pause markers (.), (..), (...)
            pause_short: Duration for (.) in seconds
            pause_medium: Duration for (..) in seconds
            pause_long: Duration for (...) in seconds
            split_mode: Optional text splitting mode. Options: None (default,
                automatic phoneme-based), "paragraph" (double newlines),
                "sentence" (requires spaCy), "clause" (sentences + commas,
                requires spaCy)
            trim_silence: Whether to trim silence from segment boundaries

        Returns:
            Tuple of (audio samples as numpy array, sample rate)
        """
        self._init_kokoro()

        # Resolve voice to style vector
        if isinstance(voice, VoiceBlend):
            voice_style = self.create_blended_voice(voice)
        elif isinstance(voice, str):
            voice_style = self.get_voice_style(voice)
        else:
            voice_style = voice

        # If already phonemes, use directly
        if is_phonemes:
            phonemes = text
            batches = self._split_phonemes(phonemes)
            return self._generate_from_phoneme_batches(
                batches, voice_style, speed, trim_silence
            ), SAMPLE_RATE

        audio_parts = []

        # Process pauses if enabled
        if enable_pauses:
            from .utils import generate_silence

            initial_pause, segments_with_pauses = self.tokenizer.split_with_pauses(
                text, pause_short, pause_medium, pause_long
            )

            # Add initial silence
            if initial_pause > 0:
                audio_parts.append(generate_silence(initial_pause, SAMPLE_RATE))

            # Process each pause-delimited segment
            for segment_text, pause_after in segments_with_pauses:
                if not segment_text.strip():
                    if pause_after > 0:
                        audio_parts.append(generate_silence(pause_after, SAMPLE_RATE))
                    continue

                # Process segment (with optional split_mode)
                segment_audio = self._process_text_segment(
                    segment_text, voice_style, speed, lang, split_mode, trim_silence
                )
                audio_parts.append(segment_audio)

                # Add pause after segment
                if pause_after > 0:
                    audio_parts.append(generate_silence(pause_after, SAMPLE_RATE))

        else:
            # No pause processing
            segment_audio = self._process_text_segment(
                text, voice_style, speed, lang, split_mode, trim_silence
            )
            audio_parts.append(segment_audio)

        if not audio_parts:
            return np.array([], dtype=np.float32), SAMPLE_RATE

        return np.concatenate(audio_parts), SAMPLE_RATE

    def create_from_phonemes(
        self,
        phonemes: str,
        voice: str | np.ndarray | VoiceBlend,
        speed: float = 1.0,
    ) -> tuple[np.ndarray, int]:
        """
        Generate audio from phonemes directly.

        This bypasses text-to-phoneme conversion, useful when working
        with pre-tokenized phoneme content.

        Args:
            phonemes: IPA phoneme string
            voice: Voice name, style vector, or VoiceBlend
            speed: Speech speed (1.0 = normal)

        Returns:
            Tuple of (audio samples as numpy array, sample rate)
        """
        self._init_kokoro()

        # Resolve voice to style vector if needed
        if isinstance(voice, VoiceBlend):
            voice_style = self.create_blended_voice(voice)
        elif isinstance(voice, str):
            voice_style = self.get_voice_style(voice)
        else:
            voice_style = voice

        # kokoro-onnx supports direct phoneme input via create method with ps parameter
        # But we need to convert to tokens first
        tokens = self.tokenizer.tokenize(phonemes)

        # Debug logging for phoneme generation
        if os.getenv("TTSFORGE_DEBUG_PHONEMES"):
            logger.info(f"Phonemes: {phonemes}")
            logger.info(f"Tokens: {tokens}")

        return self.create_from_tokens(tokens, voice_style, speed)

    def create_from_tokens(
        self,
        tokens: list[int],
        voice: str | np.ndarray | VoiceBlend,
        speed: float = 1.0,
    ) -> tuple[np.ndarray, int]:
        """
        Generate audio from token IDs directly.

        This provides the lowest-level interface, useful for pre-tokenized
        content and maximum control.

        Args:
            tokens: List of token IDs
            voice: Voice name, style vector, or VoiceBlend
            speed: Speech speed (1.0 = normal)

        Returns:
            Tuple of (audio samples as numpy array, sample rate)
        """
        self._init_kokoro()

        # Resolve voice to style vector
        if isinstance(voice, VoiceBlend):
            voice_style = self.create_blended_voice(voice)
        elif isinstance(voice, str):
            voice_style = self.get_voice_style(voice)
        else:
            voice_style = voice

        # Detokenize to phonemes and generate audio
        phonemes = self.tokenizer.detokenize(tokens)

        # Split phonemes into batches and generate audio
        batches = self._split_phonemes(phonemes)
        audio_parts = []

        for batch in batches:
            audio_part, _ = self._create_audio_internal(batch, voice_style, speed)
            # Trim silence from each part
            audio_part, _ = trim_audio(audio_part)
            audio_parts.append(audio_part)

        if not audio_parts:
            return np.array([], dtype=np.float32), SAMPLE_RATE

        return np.concatenate(audio_parts), SAMPLE_RATE

    def create_from_segment(
        self,
        segment: "PhonemeSegment",
        voice: str | np.ndarray | VoiceBlend,
        speed: float = 1.0,
        lang: str | None = None,
    ) -> tuple[np.ndarray, int]:
        """
        Generate audio from a PhonemeSegment.

        Args:
            segment: PhonemeSegment with phonemes and tokens
            voice: Voice name, style vector, or VoiceBlend
            speed: Speech speed (1.0 = normal)
            lang: Language code override (e.g., 'de', 'en-us')

        Returns:
            Tuple of (audio samples as numpy array, sample rate)
        """
        # Debug logging for segment
        if os.getenv("TTSFORGE_DEBUG_PHONEMES"):
            logger.info(f"Segment text: {segment.text[:100]}...")
            logger.info(f"Segment phonemes: {segment.phonemes}")
            logger.info(f"Segment tokens: {segment.tokens}")
            logger.info(f"Segment lang: {segment.lang}")

        # Use tokens if available, otherwise use phonemes
        if segment.tokens:
            return self.create_from_tokens(segment.tokens, voice, speed)
        elif segment.phonemes:
            return self.create_from_phonemes(segment.phonemes, voice, speed)
        else:
            # Fall back to text
            # Use lang override if provided, otherwise use segment's lang
            effective_lang = lang if lang is not None else segment.lang
            return self.create(segment.text, voice, speed, effective_lang)

    def phonemize(self, text: str, lang: str = "en-us") -> str:
        """
        Convert text to phonemes.

        Args:
            text: Input text
            lang: Language code

        Returns:
            Phoneme string
        """
        return self.tokenizer.phonemize(text, lang=lang)

    def tokenize(self, phonemes: str) -> list[int]:
        """
        Convert phonemes to tokens.

        Args:
            phonemes: Phoneme string

        Returns:
            List of token IDs
        """
        return self.tokenizer.tokenize(phonemes)

    def detokenize(self, tokens: list[int]) -> str:
        """
        Convert tokens back to phonemes.

        Args:
            tokens: List of token IDs

        Returns:
            Phoneme string
        """
        return self.tokenizer.detokenize(tokens)

    def text_to_tokens(self, text: str, lang: str = "en-us") -> list[int]:
        """
        Convert text directly to tokens.

        Args:
            text: Input text
            lang: Language code

        Returns:
            List of token IDs
        """
        return self.tokenizer.text_to_tokens(text, lang=lang)

    def generate_chunks(
        self,
        text: str,
        voice: str | np.ndarray | VoiceBlend,
        speed: float = 1.0,
        lang: str = "en-us",
        chunk_size: int = 500,
    ):
        """
        Generate audio in chunks for long text.

        This splits text into manageable chunks and yields audio for each.
        Useful for progress tracking during long conversions.

        Args:
            text: Text to synthesize
            voice: Voice name, style vector, or VoiceBlend
            speed: Speech speed
            lang: Language code
            chunk_size: Approximate character count per chunk

        Yields:
            Tuple of (audio samples, sample rate, text chunk)
        """
        self._init_kokoro()

        # Resolve voice once
        if isinstance(voice, VoiceBlend):
            voice_style = self.create_blended_voice(voice)
        elif isinstance(voice, str):
            voice_style = self.get_voice_style(voice)
        else:
            voice_style = voice

        # Split text into chunks at sentence boundaries
        chunks = self._split_text(text, chunk_size)

        for chunk in chunks:
            if not chunk.strip():
                continue

            # Convert chunk to phonemes and generate audio
            phonemes = self.tokenizer.phonemize(chunk, lang=lang)
            batches = self._split_phonemes(phonemes)
            audio_parts = []

            for batch in batches:
                audio_part, _ = self._create_audio_internal(batch, voice_style, speed)
                audio_part, _ = trim_audio(audio_part)
                audio_parts.append(audio_part)

            if audio_parts:
                samples = np.concatenate(audio_parts)
                yield samples, SAMPLE_RATE, chunk

    def _split_text(self, text: str, chunk_size: int) -> list[str]:
        """
        Split text into chunks at sentence boundaries.

        Args:
            text: Text to split
            chunk_size: Target chunk size in characters

        Returns:
            List of text chunks
        """
        # Split on sentence boundaries while keeping the delimiter
        sentences = re.split(r"(?<=[.!?])\s+", text)

        chunks = []
        current_chunk = ""

        for sentence in sentences:
            if len(current_chunk) + len(sentence) <= chunk_size:
                current_chunk += sentence + " "
            else:
                if current_chunk:
                    chunks.append(current_chunk.strip())
                current_chunk = sentence + " "

        if current_chunk:
            chunks.append(current_chunk.strip())

        return chunks

    # Voice Database Integration (from kokovoicelab)

    def load_voice_database(self, db_path: Path) -> None:
        """
        Load a voice database for custom/synthetic voices.

        Args:
            db_path: Path to the SQLite voice database
        """
        if self._voice_db is not None:
            self._voice_db.close()

        # Register numpy array converter
        sqlite3.register_converter("array", self._convert_array)
        self._voice_db = sqlite3.connect(
            str(db_path), detect_types=sqlite3.PARSE_DECLTYPES
        )

    def _convert_array(self, blob: bytes) -> np.ndarray:
        """Convert binary blob back to numpy array."""
        out = io.BytesIO(blob)
        return np.load(out)

    def get_voice_from_database(self, voice_name: str) -> np.ndarray | None:
        """
        Get a voice style vector from the database.

        Args:
            voice_name: Name of the voice in the database

        Returns:
            Voice style vector or None if not found
        """
        if self._voice_db is None:
            return None

        cursor = self._voice_db.cursor()
        cursor.execute(
            "SELECT style_vector FROM voices WHERE name = ?",
            (voice_name,),
        )
        row = cursor.fetchone()

        if row:
            return row[0]
        return None

    def list_database_voices(self) -> list[dict[str, Any]]:
        """
        List all voices in the database.

        Returns:
            List of voice metadata dictionaries
        """
        if self._voice_db is None:
            return []

        cursor = self._voice_db.cursor()
        cursor.execute(
            """
            SELECT name, gender, language, quality, is_synthetic, notes
            FROM voices
            ORDER BY quality DESC
            """
        )

        voices = []
        for row in cursor.fetchall():
            voices.append(
                {
                    "name": row[0],
                    "gender": row[1],
                    "language": row[2],
                    "quality": row[3],
                    "is_synthetic": bool(row[4]),
                    "notes": row[5],
                }
            )

        return voices

    def interpolate_voices(
        self,
        voice1: str | np.ndarray,
        voice2: str | np.ndarray,
        factor: float = 0.5,
    ) -> np.ndarray:
        """
        Interpolate between two voices.

        This uses the interpolation method from kokovoicelab to create
        voices that lie on the line between two source voices.

        Args:
            voice1: First voice (name or style vector)
            voice2: Second voice (name or style vector)
            factor: Interpolation factor (0.0 = voice1, 1.0 = voice2)

        Returns:
            Interpolated voice style vector
        """
        self._init_kokoro()

        # Resolve to style vectors
        if isinstance(voice1, str):
            style1 = self.get_voice_style(voice1)
        else:
            style1 = voice1

        if isinstance(voice2, str):
            style2 = self.get_voice_style(voice2)
        else:
            style2 = voice2

        # Use kokovoicelab's interpolation method
        diff_vector = style2 - style1
        midpoint = (style1 + style2) / 2
        return midpoint + (diff_vector * factor / 2)

    async def create_stream(
        self,
        text: str,
        voice: str | np.ndarray | VoiceBlend,
        speed: float = 1.0,
        lang: str = "en-us",
    ) -> AsyncGenerator[tuple[np.ndarray, int, str], None]:
        """
        Stream audio creation asynchronously, yielding chunks as they are processed.

        This method generates audio in the background and yields chunks as soon as
        they're ready, enabling real-time playback while generation continues.

        Args:
            text: Text to synthesize
            voice: Voice name, style vector, or VoiceBlend
            speed: Speech speed (1.0 = normal)
            lang: Language code (e.g., 'en-us', 'en-gb')

        Yields:
            Tuple of (audio samples as numpy array, sample rate, text chunk)
        """
        self._init_kokoro()

        # Resolve voice to style vector
        if isinstance(voice, VoiceBlend):
            voice_style = self.create_blended_voice(voice)
        elif isinstance(voice, str):
            voice_style = self.get_voice_style(voice)
        else:
            voice_style = voice

        # Convert text to phonemes
        phonemes = self.tokenizer.phonemize(text, lang=lang)

        # Split phonemes into batches
        batched_phonemes = self._split_phonemes(phonemes)

        # Create a queue for passing audio chunks
        queue: asyncio.Queue[tuple[np.ndarray, int, str] | None] = asyncio.Queue()

        async def process_batches() -> None:
            """Process phoneme batches in the background."""
            loop = asyncio.get_event_loop()
            for phoneme_batch in batched_phonemes:
                # Execute blocking ONNX inference in a thread executor
                audio_part, sample_rate = await loop.run_in_executor(
                    None, self._create_audio_internal, phoneme_batch, voice_style, speed
                )
                # Trim silence
                audio_part, _ = trim_audio(audio_part)
                await queue.put((audio_part, sample_rate, phoneme_batch))
            await queue.put(None)  # Signal end of stream

        # Start processing in the background
        asyncio.create_task(process_batches())

        # Yield chunks as they become available
        while True:
            chunk = await queue.get()
            if chunk is None:
                break
            yield chunk

    def create_stream_sync(
        self,
        text: str,
        voice: str | np.ndarray | VoiceBlend,
        speed: float = 1.0,
        lang: str = "en-us",
    ) -> Generator[tuple[np.ndarray, int, str], None, None]:
        """
        Stream audio creation synchronously, yielding chunks as they are processed.

        This is a synchronous version of create_stream for use in non-async contexts.
        It yields audio chunks immediately as they're generated.

        Args:
            text: Text to synthesize
            voice: Voice name, style vector, or VoiceBlend
            speed: Speech speed (1.0 = normal)
            lang: Language code (e.g., 'en-us', 'en-gb')

        Yields:
            Tuple of (audio samples as numpy array, sample rate, text chunk)
        """
        self._init_kokoro()

        # Resolve voice to style vector
        if isinstance(voice, VoiceBlend):
            voice_style = self.create_blended_voice(voice)
        elif isinstance(voice, str):
            voice_style = self.get_voice_style(voice)
        else:
            voice_style = voice

        # Convert text to phonemes
        phonemes = self.tokenizer.phonemize(text, lang=lang)

        # Split phonemes into batches
        batched_phonemes = self._split_phonemes(phonemes)

        for phoneme_batch in batched_phonemes:
            audio_part, sample_rate = self._create_audio_internal(
                phoneme_batch, voice_style, speed
            )
            # Trim silence
            audio_part, _ = trim_audio(audio_part)
            yield audio_part, sample_rate, phoneme_batch

    def close(self) -> None:
        """Clean up resources."""
        if self._voice_db is not None:
            self._voice_db.close()
            self._voice_db = None


# Language code mapping for kokoro-onnx
LANG_CODE_TO_ONNX = {
    "a": "en-us",  # American English
    "b": "en-gb",  # British English
    "e": "es",  # Spanish
    "f": "fr",  # French
    "h": "hi",  # Hindi
    "i": "it",  # Italian
    "j": "ja",  # Japanese
    "p": "pt",  # Portuguese
    "z": "zh",  # Chinese
}


def get_onnx_lang_code(ttsforge_lang: str) -> str:
    """Convert ttsforge language code to kokoro-onnx language code."""
    return LANG_CODE_TO_ONNX.get(ttsforge_lang, "en-us")
