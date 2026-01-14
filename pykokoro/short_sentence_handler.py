"""Short sentence handling for pykokoro using single-word context approach.

This module provides functionality to improve audio quality for short, single-word
sentences by using a "context-prepending" technique:

1. Only activates for short (<10 phonemes) AND single-word sentences (no spaces)
2. Duplicates the target word with a pause (e.g., "Hi ... Hi")
3. Generates TTS for combined text to add context
4. Detects a boundary near the midpoint between duplicates
5. Extracts audio from after the boundary to get only the target sentence

This approach produces better prosody and intonation compared to generating
very short sentences directly, as neural TTS models typically need more context
to produce natural-sounding speech.

Multi-word or sentences with internal breaks will NOT use this handler, as they
already have sufficient context for natural prosody.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from .audio_generator import AudioGenerator
    from .phonemes import PhonemeSegment
    from .tokenizer import Tokenizer

logger = logging.getLogger(__name__)
# Enable debug logging for this module
logger.setLevel(logging.DEBUG)

# Default thresholds for short sentence handling
DEFAULT_MIN_PHONEME_LENGTH = 5  # Sentences with fewer phonemes are "short"


def is_single_word(text: str) -> bool:
    """Check if text is a single word (contains no spaces).

    Args:
        text: Text to check

    Returns:
        True if text contains no spaces (single word), False otherwise

    Examples:
        >>> is_single_word("Hi!")
        True
        >>> is_single_word("Hi there!")
        False
        >>> is_single_word("Don't!")
        True
        >>> is_single_word("Oh, really?")
        False
    """
    return " " not in text.strip()


@dataclass
class ShortSentenceConfig:
    """Configuration for short sentence handling using single-word context.

    Short, single-word sentences (< 10 phonemes, no spaces) often sound robotic
    when generated alone. This module improves quality by:
    1. Checking sentence is both short AND single-word (no spaces)
    2. Duplicating the target word with a pause (e.g., "Hi ... Hi")
    3. Detecting a boundary near the midpoint between duplicates
    4. Extracting from that boundary to get clean target audio

    Multi-word sentences or sentences with breaks will NOT use this handler.

    Attributes:
        min_phoneme_length: Threshold below which sentences are considered "short"
            and will use context extraction. Default: 10 phonemes.
        cut_offset_ms: Milliseconds to cut earlier than detected pause point.
            Helps remove any trailing context sounds.
            Default: -10ms (preserves more audio).
        frame_ms: Frame size in milliseconds for boundary detection.
            Default: 20ms. Smaller values = finer resolution but more computation.
        hop_ms: Hop size in milliseconds for boundary detection.
            Default: 10ms. Smaller values = finer resolution but more computation.
        energy_weight: Weight for energy feature in boundary detection (0.0-1.0).
            Default: 0.333 (equal weighting). Can be tuned for different voices.
        zcr_weight: Weight for zero-crossing rate in boundary detection (0.0-1.0).
            Default: 0.333 (equal weighting).
        flux_weight: Weight for spectral flux in boundary detection (0.0-1.0).
            Default: 0.334 (equal weighting).
        early_search_window_ms: Expanded search window around midpoint if needed.
            Default: 400ms. Used when midpoint window yields no candidates.
        midpoint_window_ms: Search window (± ms) around midpoint.
            Default: 200ms. Used to bias boundary selection.
        midpoint_bias_weight: Weight for midpoint distance penalty.
            Higher values bias closer to midpoint. Default: 0.6.
        enabled: Whether short sentence handling is enabled. Default: True.

    Note: Weights should sum to ~1.0.
        Equal weights (0.33 each) work well for most cases.
        For voices with unclear boundaries,
        try increasing energy_weight to 0.5-0.6.
    """

    min_phoneme_length: int = DEFAULT_MIN_PHONEME_LENGTH
    cut_offset_ms: int = 0
    frame_ms: int = 20
    hop_ms: int = 5
    energy_weight: float = 0.6
    zcr_weight: float = 0.2
    flux_weight: float = 0.2
    early_search_window_ms: int = 200
    midpoint_window_ms: int = 100
    midpoint_bias_weight: float = 0.7
    disable_cutoff_detection: bool = False
    enabled: bool = True

    def should_use_context_prepending(self, phoneme_length: int, text: str) -> bool:
        """Check if segment should use context-prepending.

        Args:
            phoneme_length: Number of phonemes in the segment
            text: The text content to check for single-word status

        Returns:
            True if context-prepending should be applied
            (sentence is short AND single-word)
        """
        return (
            self.enabled
            and phoneme_length < self.min_phoneme_length
            and is_single_word(text)
        )


def energy_based_vad(
    audio: np.ndarray,
    sample_rate: int,
    frame_duration_ms: int = 5,
    energy_threshold: float = 0.02,
) -> np.ndarray:
    """Simple energy-based voice activity detection.

    Args:
        audio: Audio signal as numpy array
        sample_rate: Sample rate of audio
        frame_duration_ms: Frame size in milliseconds (default: 5ms for high resolution)
        energy_threshold: Energy threshold (0.0-1.0) - lower = more sensitive

    Returns:
        Boolean array indicating speech/silence for each frame
    """
    audio = audio.astype(np.float32)

    # Frame parameters
    frame_length = int(sample_rate * frame_duration_ms / 1000)
    n_frames = len(audio) // frame_length

    # Compute short-time energy for each frame
    energy = np.zeros(n_frames)
    for i in range(n_frames):
        frame = audio[i * frame_length : (i + 1) * frame_length]
        energy[i] = np.sqrt(np.sum(frame**2) / len(frame))

    # Normalize energy
    energy_norm = (energy - energy.min()) / (energy.max() - energy.min() + 1e-8)

    # Apply threshold to detect speech
    voice_activity = energy_norm > energy_threshold

    return voice_activity


def find_speech_start(
    audio: np.ndarray,
    sample_rate: int,
    energy_threshold: float = 0.05,
    frame_duration_ms: int = 5,
) -> int:
    """Find where speech starts in audio (end of initial silence).

    Args:
        audio: Audio signal to analyze
        sample_rate: Sample rate of audio
        energy_threshold: Energy threshold for VAD (0.0-1.0)
        frame_duration_ms: Frame duration in milliseconds

    Returns:
        Sample index where speech starts (0 if no silence detected)
    """
    voice_activity = energy_based_vad(
        audio,
        sample_rate,
        frame_duration_ms=frame_duration_ms,
        energy_threshold=energy_threshold,
    )

    samples_per_frame = int(sample_rate * frame_duration_ms / 1000)

    # Find first speech frame
    for i, is_speech in enumerate(voice_activity):
        if is_speech:
            speech_start_sample = i * samples_per_frame
            logger.debug(
                f"Speech starts at frame {i}, sample {speech_start_sample} "
                f"({speech_start_sample / sample_rate:.3f}s)"
            )
            return speech_start_sample

    # No speech found, return 0
    return 0


def frame_signal(
    audio: np.ndarray,
    sample_rate: int,
    frame_ms: int = 20,
    hop_ms: int = 10,
) -> np.ndarray:
    """Split audio signal into overlapping frames.

    Args:
        audio: Audio signal as numpy array
        sample_rate: Sample rate of audio
        frame_ms: Frame size in milliseconds (default: 20ms)
        hop_ms: Hop size in milliseconds (default: 10ms)

    Returns:
        2D array of frames (n_frames × frame_length)
    """
    frame_length = int(sample_rate * frame_ms / 1000)
    hop_length = int(sample_rate * hop_ms / 1000)

    # Calculate number of frames
    n_frames = (len(audio) - frame_length) // hop_length + 1

    # Create frames array
    frames = np.zeros((n_frames, frame_length), dtype=audio.dtype)

    for i in range(n_frames):
        start = i * hop_length
        end = start + frame_length
        frames[i] = audio[start:end]

    return frames


def short_time_energy(frames: np.ndarray) -> np.ndarray:
    """Calculate short-time energy for each frame.

    Args:
        frames: 2D array of frames (n_frames × frame_length)

    Returns:
        1D array of normalized energy values (0-1) per frame
    """
    # Calculate RMS energy for each frame
    energy = np.sqrt(np.mean(frames**2, axis=1))

    # Normalize to [0, 1] range
    energy_min = energy.min()
    energy_max = energy.max()
    if energy_max > energy_min:
        energy_norm = (energy - energy_min) / (energy_max - energy_min)
    else:
        energy_norm = np.zeros_like(energy)

    return energy_norm


def zero_crossing_rate(frames: np.ndarray) -> np.ndarray:
    """Calculate zero crossing rate for each frame.

    Zero crossing rate indicates voiced (low ZCR) vs unvoiced (high ZCR) speech.

    Args:
        frames: 2D array of frames (n_frames × frame_length)

    Returns:
        1D array of normalized ZCR values (0-1) per frame
    """
    # Count sign changes in each frame
    # Sign change occurs when adjacent samples have opposite signs
    signs = np.sign(frames)
    sign_changes = np.abs(np.diff(signs, axis=1))
    zcr = np.sum(sign_changes > 0, axis=1) / frames.shape[1]

    # Normalize to [0, 1] range
    zcr_min = zcr.min()
    zcr_max = zcr.max()
    if zcr_max > zcr_min:
        zcr_norm = (zcr - zcr_min) / (zcr_max - zcr_min)
    else:
        zcr_norm = np.zeros_like(zcr)

    return zcr_norm


def spectral_flux(frames: np.ndarray, sample_rate: int) -> np.ndarray:
    """Calculate spectral flux for each frame using pure NumPy FFT.

    Spectral flux measures the rate of change in the power spectrum.
    High flux indicates transitions between different sounds.

    Args:
        frames: 2D array of frames (n_frames × frame_length)
        sample_rate: Sample rate of audio

    Returns:
        1D array of normalized spectral flux values (0-1) per frame
    """
    n_frames, frame_length = frames.shape

    # Apply Hamming window to each frame
    window = np.hamming(frame_length)
    windowed_frames = frames * window

    # Compute FFT for each frame
    # Use rfft for real-valued signals (more efficient)
    fft_frames = np.fft.rfft(windowed_frames, axis=1)
    magnitude_spectra = np.abs(fft_frames)

    # Calculate spectral flux (sum of positive differences from previous frame)
    flux = np.zeros(n_frames)
    for i in range(1, n_frames):
        diff = magnitude_spectra[i] - magnitude_spectra[i - 1]
        # Only count positive differences (increases in magnitude)
        flux[i] = np.sum(diff[diff > 0])

    # First frame has no previous frame, so flux = 0

    # Normalize to [0, 1] range
    flux_min = flux.min()
    flux_max = flux.max()
    if flux_max > flux_min:
        flux_norm = (flux - flux_min) / (flux_max - flux_min)
    else:
        flux_norm = np.zeros_like(flux)

    return flux_norm


def median_filter_numpy(data: np.ndarray, window_size: int = 5) -> np.ndarray:
    """Apply median filter using pure NumPy (manual sliding window).

    Args:
        data: 1D array to filter
        window_size: Window size for median filter (default: 5)

    Returns:
        Filtered 1D array (same length as input)
    """
    n = len(data)
    filtered = np.zeros(n)
    half_window = window_size // 2

    for i in range(n):
        # Determine window bounds
        start = max(0, i - half_window)
        end = min(n, i + half_window + 1)

        # Extract window and compute median
        window = data[start:end]
        filtered[i] = np.median(window)

    return filtered


def find_boundary_valley(
    audio: np.ndarray,
    sample_rate: int,
    frame_ms: int = 20,
    hop_ms: int = 10,
    energy_weight: float = 0.333,
    zcr_weight: float = 0.333,
    flux_weight: float = 0.334,
    early_search_window_ms: int = 400,
    midpoint_window_ms: int = 200,
    midpoint_bias_weight: float = 0.6,
    depth_threshold: float = 1.0,
) -> int:
    """Find boundary between duplicated word halves.

    Uses multi-feature valley detection.

    This function uses a robust approach combining:
    - Short-time energy (STE): Detects silence/low energy regions
    - Zero crossing rate (ZCR): Detects voiced/unvoiced transitions
    - Spectral flux: Detects spectral changes

    The algorithm:
    1. Frames the audio (20ms frames, 10ms hop)
    2. Extracts and normalizes STE, ZCR, and spectral flux
    3. Smooths features with median filter
    4. Combines features to find valleys (low combined value = boundary)
    5. Detects speech boundaries using energy
    6. Selects a valley near the midpoint (depth + distance scoring)
    7. Returns sample index to cut (with offset)

    Args:
        audio: Audio signal containing repeated word or context
            (e.g., "word ... word" or "Good. word")
        sample_rate: Sample rate of audio
        frame_ms: Frame size in milliseconds (default: 20ms)
        hop_ms: Hop size in milliseconds (default: 10ms)
        energy_weight: Weight for energy feature (default: 0.333).
            Higher = more emphasis on energy.
        zcr_weight: Weight for ZCR feature (default: 0.333)
        flux_weight: Weight for flux feature (default: 0.334)
        early_search_window_ms: Expanded search window around midpoint
            (default: 400ms). Used if midpoint window is empty.
        midpoint_window_ms: Search window (± ms) around midpoint.
            (default: 200ms). Limits candidate valleys.
        midpoint_bias_weight: Weight for midpoint distance penalty.
            (default: 0.6). Higher biases closer to midpoint.
        depth_threshold: Prefer valleys deeper than this
            (default: 1.00). Lower = more selective.

    Returns:
        Sample index where to cut the audio

    Raises:
        ValueError: If speech boundaries cannot be detected
    """
    audio_duration = len(audio) / sample_rate

    logger.debug(
        f"\n=== Multi-Feature Boundary Detection ==="
        f"\nAudio duration: {audio_duration:.3f}s ({len(audio)} samples)"
        f"\nFrame: {frame_ms}ms, Hop: {hop_ms}ms"
    )

    # Step 1: Frame the signal
    frames = frame_signal(audio, sample_rate, frame_ms=frame_ms, hop_ms=hop_ms)
    n_frames = len(frames)
    hop_length = int(sample_rate * hop_ms / 1000)

    logger.debug(f"Frames: {n_frames} frames")

    # Step 2: Extract features
    logger.debug("Extracting features...")

    ste = short_time_energy(frames)
    zcr = zero_crossing_rate(frames)
    flux = spectral_flux(frames, sample_rate)

    logger.debug(
        f"  STE range: [{ste.min():.4f}, {ste.max():.4f}]"
        f"\n  ZCR range: [{zcr.min():.4f}, {zcr.max():.4f}]"
        f"\n  Flux range: [{flux.min():.4f}, {flux.max():.4f}]"
    )

    # Step 3: Smooth features with median filter
    logger.debug("Applying median filter (window=5)...")

    ste_smooth = median_filter_numpy(ste, window_size=5)
    zcr_smooth = median_filter_numpy(zcr, window_size=5)
    flux_smooth = median_filter_numpy(flux, window_size=5)

    # Step 4: Combine features with configurable weights
    # Low STE = silence
    # Low ZCR = voiced speech (we want this at target word start)
    # Low flux = stable spectrum
    # Invert ZCR and flux so valleys indicate boundaries
    # Default weights: energy=0.6, zcr=0.2, flux=0.2 (energy is most reliable)
    combined = (
        ste_smooth * energy_weight
        + (1 - zcr_smooth) * zcr_weight
        + (1 - flux_smooth) * flux_weight
    )

    logger.debug(
        f"Combined feature range: [{combined.min():.4f}, {combined.max():.4f}]"
        f"\nWeights: energy={energy_weight}, zcr={zcr_weight}, flux={flux_weight}"
    )

    # Step 5: Detect speech boundaries dynamically using energy
    speech_threshold = 0.05  # Energy above this is considered speech
    speech_start_frame = None
    speech_end_frame = None

    # Find first frame where energy exceeds threshold
    for i in range(n_frames):
        if ste_smooth[i] > speech_threshold:
            speech_start_frame = i
            break

    # Find last frame where energy exceeds threshold
    for i in range(n_frames - 1, -1, -1):
        if ste_smooth[i] > speech_threshold:
            speech_end_frame = i
            break

    if speech_start_frame is None or speech_end_frame is None:
        raise ValueError(
            f"Could not detect speech boundaries in audio.\n"
            f"  Duration: {audio_duration:.3f}s, Frames: {n_frames}\n"
            f"  STE range: [{ste.min():.6f}, {ste.max():.6f}]\n"
            f"  Speech threshold: {speech_threshold}\n"
            f"  Try adjusting audio or the duplicated word/pause marker."
        )

    speech_start_time = speech_start_frame * hop_ms / 1000.0
    speech_end_time = speech_end_frame * hop_ms / 1000.0

    logger.debug(
        f"Speech boundaries detected:\n"
        f"  Start: frame {speech_start_frame} ({speech_start_time:.3f}s)\n"
        f"  End: frame {speech_end_frame} ({speech_end_time:.3f}s)\n"
        f"  Duration: {speech_end_time - speech_start_time:.3f}s"
    )

    # Step 6: Find valleys (local minimums) in combined feature
    valleys = []

    for i in range(1, n_frames - 1):
        # Valley: combined[i] < both neighbors
        if combined[i] < combined[i - 1] and combined[i] < combined[i + 1]:
            frame_time = i * hop_ms / 1000.0
            depth = combined[i]  # Lower is deeper
            valleys.append((i, frame_time, depth))

    logger.debug(f"Total valleys found: {len(valleys)}")

    midpoint_sample = len(audio) // 2
    midpoint_frame = int(round(midpoint_sample / hop_length))
    midpoint_window_frames = max(1, int(midpoint_window_ms / hop_ms))

    logger.debug(
        f"Midpoint target: frame {midpoint_frame} "
        f"({midpoint_sample / sample_rate:.3f}s)"
        f"\nMidpoint window: ±{midpoint_window_ms}ms "
        f"({midpoint_window_frames} frames)"
    )

    def score_valley(frame: int, depth: float, distance_denominator: int) -> float:
        distance = abs(frame - midpoint_frame) / max(1, distance_denominator)
        return depth + midpoint_bias_weight * distance

    def select_best_valley(
        candidates: list[tuple[int, float, float]], distance_denominator: int
    ) -> tuple[int, float, float] | None:
        if not candidates:
            return None
        deep_candidates = [
            candidate for candidate in candidates if candidate[2] < depth_threshold
        ]
        considered = deep_candidates or candidates
        return min(
            considered,
            key=lambda candidate: score_valley(
                candidate[0], candidate[2], distance_denominator
            ),
        )

    search_start_frame = midpoint_frame - midpoint_window_frames
    search_end_frame = midpoint_frame + midpoint_window_frames

    midpoint_candidates = [
        (frame, time, depth)
        for frame, time, depth in valleys
        if search_start_frame <= frame <= search_end_frame
    ]

    selected_valley = select_best_valley(midpoint_candidates, midpoint_window_frames)

    if selected_valley is None and early_search_window_ms > midpoint_window_ms:
        fallback_window_frames = max(
            midpoint_window_frames, int(early_search_window_ms / hop_ms)
        )
        fallback_start_frame = midpoint_frame - fallback_window_frames
        fallback_end_frame = midpoint_frame + fallback_window_frames
        fallback_candidates = [
            (frame, time, depth)
            for frame, time, depth in valleys
            if fallback_start_frame <= frame <= fallback_end_frame
        ]
        logger.debug("No valleys in midpoint window, expanding search around midpoint")
        selected_valley = select_best_valley(
            fallback_candidates, fallback_window_frames
        )

    if selected_valley is None:
        logger.debug("No valleys found; falling back to midpoint cut")
        return midpoint_sample

    selected_frame, selected_time, selected_depth = selected_valley

    logger.debug(
        f"Selected valley: frame {selected_frame}, "
        f"time {selected_time:.3f}s, depth {selected_depth:.4f}"
    )

    # Step 8: Convert frame to sample index
    cut_sample = selected_frame * hop_length

    logger.debug(f"Cut point: {cut_sample} samples ({cut_sample / sample_rate:.3f}s) ")

    return cut_sample


def generate_short_sentence_audio(
    segment: PhonemeSegment,
    audio_generator: AudioGenerator,
    voice_style: np.ndarray,
    speed: float,
    config: ShortSentenceConfig | None = None,
    tokenizer: Tokenizer | None = None,
) -> np.ndarray:
    """Generate high-quality audio for short, single-word sentences using context.

    This function duplicates the word with a pause and finds a midpoint boundary:
    1. Only activates for short (<10 phonemes) AND single-word sentences (no spaces)
    2. Generates "{word} ... {word}" to add context
    3. Finds a boundary near the midpoint using valley detection
    4. Extracts from that boundary (with configurable offset) to get clean target

    Multi-word sentences will NOT use this handler and generate normally.

    Args:
        segment: PhonemeSegment containing the sentence
        audio_generator: AudioGenerator instance for TTS
        voice_style: Voice style vector
        speed: Speech speed multiplier
        config: Short sentence configuration (uses defaults if None)
        tokenizer: Tokenizer for phonemizing combined text (uses audio_generator's
            tokenizer if None)

    Returns:
        High-quality audio for the sentence (with context extraction if applicable)

    Note:
        This function makes 1 TTS call: generates context + target together
        (or target alone if multi-word)
    """
    if config is None:
        config = ShortSentenceConfig()

    if tokenizer is None:
        tokenizer = audio_generator._tokenizer

    phoneme_length = len(segment.phonemes)

    # Check if should use context prepending (short AND single-word)
    if not config.should_use_context_prepending(phoneme_length, segment.text):
        # Multi-word or long sentence - generate normally
        audio, _ = audio_generator.generate_from_phonemes(
            segment.phonemes, voice_style, speed
        )

        if not is_single_word(segment.text):
            logger.debug(
                f"Skipping short handler for multi-word sentence: '{segment.text[:50]}'"
            )

        return audio

    logger.debug(
        f"Using context extraction for short single-word: '{segment.text[:50]}' "
        f"({phoneme_length} phonemes)"
    )

    phonemes = tokenizer.phonemize(segment.text, lang=segment.lang)
    combined_phonemes = phonemes + ".….….….….….…. " + phonemes
    combined_phonemes = phonemes + ".—.—.—.—.—.—.—. " + phonemes
    combined_phonemes = phonemes + "........ " + phonemes
    # combined_phonemes = ".. " + phonemes + ".. "
    # combined_phonemes = "…" + phonemes + "…"
    # combined_phonemes = "—" + phonemes + "—"
    combined_audio, sample_rate = audio_generator.generate_from_phonemes(
        combined_phonemes, voice_style, speed
    )

    logger.debug(
        f"Combined audio: {len(combined_audio)} samples "
        f"({len(combined_audio) / sample_rate:.3f}s)"
    )
    if config.disable_cutoff_detection:
        # Skip boundary detection, cut at midpoint
        cut_sample = len(combined_audio) // 2

        logger.debug(
            f"Cutoff detection disabled; using midpoint cut at sample {cut_sample} "
            f"({cut_sample / sample_rate:.3f}s)"
        )
    else:
        # Step 2: Use multi-feature boundary detection to find midpoint cut
        logger.debug("Step 2: Using multi-feature valley detection to find boundary")

        try:
            cut_sample = find_boundary_valley(
                combined_audio,
                sample_rate,
                frame_ms=config.frame_ms,
                hop_ms=config.hop_ms,
                energy_weight=config.energy_weight,
                zcr_weight=config.zcr_weight,
                flux_weight=config.flux_weight,
                early_search_window_ms=config.early_search_window_ms,
                midpoint_window_ms=config.midpoint_window_ms,
                midpoint_bias_weight=config.midpoint_bias_weight,
            )
            logger.debug(
                f"Boundary detected at sample {cut_sample} "
                f"({cut_sample / sample_rate:.3f}s)"
            )
        except ValueError as e:
            logger.error(f"Boundary detection failed: {e}")
            raise RuntimeError(
                f"Failed to detect boundary between duplicated words "
                f"in '{segment.text}'. "
                f"This may indicate an issue with TTS generation or audio quality. "
                f"Audio duration: {len(combined_audio) / sample_rate:.3f}s"
            ) from e

    cut_offset_samples = int(config.cut_offset_ms * sample_rate / 1000)
    cut_sample = max(0, cut_sample - cut_offset_samples)

    # Extract audio from detected boundary
    # extracted_audio = combined_audio[cut_sample:]
    extracted_audio = combined_audio[1:]

    logger.debug(
        f"Extracted: {len(extracted_audio)} samples "
        f"({len(extracted_audio) / sample_rate:.3f}s)"
    )

    return extracted_audio


def is_segment_short(
    segment: PhonemeSegment,
    config: ShortSentenceConfig | None = None,
) -> bool:
    """Check if segment should use context-prepending.

    Checks if segment is BOTH short (<10 phonemes) AND single-word (no spaces).

    Args:
        segment: PhonemeSegment to check
        config: Configuration (uses defaults if None)

    Returns:
        True if segment should use context-prepending (short AND single-word)
    """
    if config is None:
        config = ShortSentenceConfig()

    # Skip empty segments
    if not segment.phonemes.strip():
        return False

    return config.should_use_context_prepending(len(segment.phonemes), segment.text)
