"""Short sentence handling for pykokoro using single-word context approach.

This module provides functionality to improve audio quality for short, single-word
sentences by using a "context-prepending" technique:

1. Only activates for short (<10 phonemes) AND single-word sentences (no spaces)
2. Prepends simple context "Two. " before the target (e.g., "Two. Hi!")
3. Generates TTS for combined text (better quality with context)
4. Detects the ONE pause between "Two." and target using adaptive threshold
5. Extracts audio from after the pause to get only the target sentence

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

from .trim import trim as trim_audio

if TYPE_CHECKING:
    from .audio_generator import AudioGenerator
    from .phonemes import PhonemeSegment
    from .tokenizer import Tokenizer

logger = logging.getLogger(__name__)
# Enable debug logging for this module
logger.setLevel(logging.DEBUG)

# Default thresholds for short sentence handling
DEFAULT_MIN_PHONEME_LENGTH = 10  # Sentences with fewer phonemes are "short"
DEFAULT_CUT_OFFSET_MS = 2  # Milliseconds to cut earlier (removes trailing context)


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
    2. Prepending context "Two. " before target (e.g., "Two. Hi!")
    3. Detecting the ONE pause between context and target
    4. Extracting from that pause to get clean target audio

    Multi-word sentences or sentences with breaks will NOT use this handler.

    Attributes:
        min_phoneme_length: Threshold below which sentences are considered "short"
            and will use context extraction. Default: 10 phonemes.
        context_sentence: Context to prepend. Should be a single word with period
            and space. Default: "Two. " (creates one clear pause).
        cut_offset_ms: Milliseconds to cut earlier than detected pause point.
            Helps remove any trailing context sounds. Default: 2ms.
        enabled: Whether short sentence handling is enabled. Default: True.
    """

    min_phoneme_length: int = DEFAULT_MIN_PHONEME_LENGTH
    context_sentence: str = "Two. "
    cut_offset_ms: int = DEFAULT_CUT_OFFSET_MS
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


def create_context_text(
    text: str,
    context_sentence: str,
) -> str:
    """Create text with context sentence prepended.

    Ensures the context has a space separator for natural sentence boundary.

    Args:
        text: Original sentence text
        context_sentence: Sentence to prepend for context (should end with space)

    Returns:
        Combined text string

    Example:
        >>> create_context_text("Hi!", "Two. ")
        "Two. Hi!"
    """
    # Ensure context ends with space for natural boundary
    if not context_sentence.endswith(" "):
        context_sentence = context_sentence + " "
    return context_sentence + text


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
    cut_offset_ms: int = 2,
) -> int:
    """Find boundary between context and target word using multi-feature valley detection.

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
    6. Finds deepest valley within speech range
    7. Returns sample index to cut (with offset)

    Args:
        audio: Audio signal containing "Two. {target}"
        sample_rate: Sample rate of audio
        frame_ms: Frame size in milliseconds (default: 20ms)
        hop_ms: Hop size in milliseconds (default: 10ms)
        cut_offset_ms: Milliseconds to cut before detected boundary (default: 2ms)

    Returns:
        Sample index where to cut the audio

    Raises:
        ValueError: If no boundary valley can be detected
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

    # Step 4: Combine features
    # Low STE = silence
    # Low ZCR = voiced speech (we want this at target word start)
    # Low flux = stable spectrum
    # Invert ZCR and flux so valleys indicate boundaries
    combined = (ste_smooth + (1 - zcr_smooth) + (1 - flux_smooth)) / 3

    logger.debug(
        f"Combined feature range: [{combined.min():.4f}, {combined.max():.4f}]"
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
            f"  Try adjusting audio or using different context word."
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

    # Find the FIRST DEEP valley after speech start - this is our boundary region indicator
    # Fine-tune by searching in a small window AFTER it for the best boundary
    # Strategy: Look for valley with depth < 0.80 in first 400ms
    # If not found, use deepest valley in first 400ms (prevents cutting too late)
    first_valley_frame = None
    first_valley_time = None
    first_valley_depth = None
    depth_threshold = 0.80  # Prefer valleys deeper than this
    early_search_window_ms = 400  # Look for valleys in first 400ms of speech
    early_search_end = speech_start_frame + int(early_search_window_ms / hop_ms)
    deepest_early_valley = None  # Track deepest valley in early window

    logger.debug(
        f"Looking for first deep valley (depth < {depth_threshold}) "
        f"in first {early_search_window_ms}ms after speech start (frame {speech_start_frame}):"
    )

    # First pass: look for deep valleys in early window
    for frame, time, depth in valleys:
        if frame > speech_start_frame + 5:  # Skip valleys too close to speech start
            if frame <= early_search_end:
                # Track the deepest valley in early window
                if deepest_early_valley is None or depth < deepest_early_valley[2]:
                    deepest_early_valley = (frame, time, depth)

                # Prefer valleys deeper than threshold in early window
                if depth < depth_threshold:
                    first_valley_frame = frame
                    first_valley_time = time
                    first_valley_depth = depth
                    logger.debug(
                        f"  Found deep valley at frame {frame} ({time:.3f}s), depth {depth:.4f}"
                    )
                    break
                else:
                    logger.debug(
                        f"  Skipping shallow valley at frame {frame} ({time:.3f}s), depth {depth:.4f}"
                    )
            else:
                # Past early window, stop searching
                break

    # If no deep valley found in early window, use deepest one
    if first_valley_frame is None and deepest_early_valley is not None:
        first_valley_frame, first_valley_time, first_valley_depth = deepest_early_valley
        logger.debug(
            f"  No valley < {depth_threshold} in first {early_search_window_ms}ms, "
            f"using deepest: frame {first_valley_frame} ({first_valley_time:.3f}s), depth {first_valley_depth:.4f}"
        )
    elif first_valley_frame is None:
        logger.debug(f"  No suitable valley found, using full speech range")

    # Fine-tuning: search in a narrow window around the first valley
    # This allows selecting a nearby valley but prevents searching too far ahead
    if first_valley_frame is not None:
        # Search from 20ms before to 50ms after the first valley
        # This small window refines the cut point without jumping to later valleys
        search_start_frame = max(speech_start_frame, first_valley_frame - 2)  # -20ms
        search_end_frame = min(speech_end_frame, first_valley_frame + 5)  # +50ms
    else:
        # Fallback: use full speech range
        search_start_frame = speech_start_frame
        search_end_frame = speech_end_frame

    search_start_time = search_start_frame * hop_ms / 1000.0
    search_end_time = search_end_frame * hop_ms / 1000.0

    logger.debug(
        f"Fine-tuning search (-20ms to +50ms around first valley): "
        f"{search_start_time:.3f}s to {search_end_time:.3f}s "
        f"(frames {search_start_frame}-{search_end_frame})"
    )

    valid_valleys = [
        (frame, time, depth)
        for frame, time, depth in valleys
        if search_start_frame <= frame <= search_end_frame
    ]

    logger.debug(f"Valleys in fine-tuning range: {len(valid_valleys)}")

    if not valid_valleys:
        # No valleys found in search range
        raise ValueError(
            f"No boundary valleys found in search range.\n"
            f"  Duration: {audio_duration:.3f}s, Frames: {n_frames}\n"
            f"  Search range: {search_start_time:.3f}s to {search_end_time:.3f}s\n"
            f"  Total valleys found: {len(valleys)}\n"
            f"  Valleys in range: 0\n"
            f"  Combined feature range: [{combined.min():.4f}, {combined.max():.4f}]"
        )

    # Step 7: Select the DEEPEST valley (lowest combined value)
    valid_valleys.sort(key=lambda x: x[2])  # Sort by depth (ascending)
    selected_frame, selected_time, selected_depth = valid_valleys[0]

    # Log top 3 deepest valleys for debugging
    logger.debug("Top 3 deepest valleys:")
    for idx, (frame, time, depth) in enumerate(valid_valleys[:3], 1):
        logger.debug(f"  #{idx}: frame {frame}, time {time:.3f}s, depth {depth:.4f}")

    logger.debug(
        f"Selected deepest valley: frame {selected_frame}, "
        f"time {selected_time:.3f}s, depth {selected_depth:.4f}"
    )

    # Step 8: Convert frame to sample index and apply offset
    cut_sample = selected_frame * hop_length
    cut_offset_samples = int(cut_offset_ms * sample_rate / 1000)
    cut_sample = max(0, cut_sample - cut_offset_samples)

    logger.debug(
        f"Cut point: {cut_sample} samples ({cut_sample / sample_rate:.3f}s) "
        f"[{cut_offset_ms}ms offset applied]"
    )

    return cut_sample


def find_silence_gap(
    audio: np.ndarray,
    sample_rate: int,
    gap_index: int = 1,
    threshold_db: float = 60,
    min_frames: int = 2,
    energy_threshold: float = 0.05,
    frame_duration_ms: int = 5,
) -> tuple[int, int]:
    """Find Nth silence gap in audio with adaptive threshold.

    This function adaptively adjusts the energy threshold to find sentence boundaries:
    1. Starts with the given energy_threshold
    2. Finds gaps, filters out those in final 30% (ending silence)
    3. If no valid gaps found, lowers threshold and retries
    4. Returns the Nth gap by position (earliest first)

    Args:
        audio: Audio signal to analyze
        sample_rate: Sample rate of audio
        gap_index: Which gap to use when sorted by position (1 = first, 2 = second, etc.)
        threshold_db: Threshold in dB (deprecated)
        min_frames: Minimum consecutive silent frames to consider a gap
        energy_threshold: Initial energy threshold for VAD (0.0-1.0)
        frame_duration_ms: Frame duration in milliseconds

    Returns:
        Tuple of (gap_end_sample, actual_gap_index)
        If no valid gaps found even with adaptive threshold, returns (0, 0).
    """
    # Try multiple thresholds, starting high and going lower
    thresholds_to_try = [energy_threshold, 0.04, 0.03, 0.02, 0.01]

    for attempt, current_threshold in enumerate(thresholds_to_try):
        logger.debug(
            f"\n=== Attempt {attempt + 1}: energy_threshold={current_threshold} ==="
        )

        # Get voice activity using energy-based VAD
        voice_activity = energy_based_vad(
            audio,
            sample_rate,
            frame_duration_ms=frame_duration_ms,
            energy_threshold=current_threshold,
        )

        # Find all gaps (transitions: speech → silence → speech)
        in_speech = False
        gap_start_frame = None
        samples_per_frame = int(sample_rate * frame_duration_ms / 1000)
        gaps_found = []
        silence_counter = 0

        logger.debug(
            f"Voice activity: speech frames={voice_activity.sum()}/{len(voice_activity)}"
        )

        for i, is_speech in enumerate(voice_activity):
            if is_speech:
                # We're in a speech frame
                if gap_start_frame is not None and silence_counter >= min_frames:
                    # We just ended a silence gap that was long enough
                    gap_end_frame = i
                    gap_end_sample = gap_end_frame * samples_per_frame
                    gap_start_sample = gap_start_frame * samples_per_frame
                    gaps_found.append(
                        {
                            "end_sample": gap_end_sample,
                            "start_sample": gap_start_sample,
                            "length_frames": silence_counter,
                            "length_ms": silence_counter * frame_duration_ms,
                            "end_time_s": gap_end_sample / sample_rate,
                        }
                    )

                # Reset gap tracking
                gap_start_frame = None
                silence_counter = 0
                in_speech = True
            else:
                # We're in a silence frame
                if in_speech:
                    # Start of a new gap (just transitioned from speech to silence)
                    gap_start_frame = i
                    silence_counter = 1
                    in_speech = False
                else:
                    # Continue silence
                    if gap_start_frame is not None:
                        silence_counter += 1

        logger.debug(f"Total gaps found: {len(gaps_found)}")

        if gaps_found:
            for i, gap in enumerate(gaps_found):
                logger.debug(
                    f"  Gap {i + 1}: {gap['length_ms']}ms ending at {gap['end_time_s']:.3f}s"
                )

        # Filter out gaps in the final 30% of audio (likely ending silence)
        audio_duration = len(audio) / sample_rate
        cutoff_time = audio_duration * 0.70

        gaps_filtered = [g for g in gaps_found if g["end_time_s"] <= cutoff_time]

        logger.debug(f"Gaps in first 70% (< {cutoff_time:.3f}s): {len(gaps_filtered)}")

        # If we found valid gaps, use them
        if gaps_filtered:
            # Sort by position (earliest first)
            gaps_sorted = sorted(gaps_filtered, key=lambda x: x["end_sample"])

            # Return the Nth gap by position
            actual_index = min(gap_index, len(gaps_sorted))
            selected_gap = gaps_sorted[actual_index - 1]
            gap_end_sample = selected_gap["end_sample"]

            logger.debug(
                f"✓ Selected gap #{actual_index}: {selected_gap['length_ms']}ms "
                f"at {gap_end_sample / sample_rate:.3f}s (threshold={current_threshold})"
            )

            return gap_end_sample, actual_index

        # No valid gaps with this threshold, try next one
        logger.debug(
            f"✗ No valid gaps with threshold={current_threshold}, trying lower..."
        )

    # Exhausted all thresholds, no valid gaps found
    logger.warning("No valid silence gaps found even with adaptive threshold")
    return 0, 0


def generate_short_sentence_audio(
    segment: PhonemeSegment,
    audio_generator: AudioGenerator,
    voice_style: np.ndarray,
    speed: float,
    config: ShortSentenceConfig | None = None,
    tokenizer: Tokenizer | None = None,
) -> np.ndarray:
    """Generate high-quality audio for short, single-word sentences using context.

    This function uses the "Two. " context with local minimum detection:
    1. Only activates for short (<10 phonemes) AND single-word sentences (no spaces)
    2. Generates with context: "Two. {target}" (e.g., "Two. Hi!")
    3. Finds the 5 deepest local energy minimums
    4. Selects the earliest minimum not in final 20% as the pause point
    5. Extracts from that pause point (with configurable offset) to get clean target

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

    # Generate with context prepended
    combined_text = create_context_text(segment.text, config.context_sentence)
    logger.debug(f"Generating with context: '{combined_text[:80]}'")

    combined_phonemes = tokenizer.phonemize(combined_text, lang=segment.lang)
    combined_audio, sample_rate = audio_generator.generate_from_phonemes(
        combined_phonemes, voice_style, speed
    )

    logger.debug(
        f"Combined audio: {len(combined_audio)} samples ({len(combined_audio) / sample_rate:.3f}s)"
    )

    # Step 2: Use multi-feature boundary detection to find pause between "Two." and target
    logger.debug("Step 2: Using multi-feature valley detection to find boundary")

    try:
        cut_sample = find_boundary_valley(
            combined_audio,
            sample_rate,
            frame_ms=20,
            hop_ms=10,
            cut_offset_ms=config.cut_offset_ms,
        )
        logger.debug(
            f"Boundary detected at sample {cut_sample} ({cut_sample / sample_rate:.3f}s)"
        )
    except ValueError as e:
        logger.error(f"Boundary detection failed: {e}")
        raise RuntimeError(
            f"Failed to detect boundary between context and target word in '{segment.text}'. "
            f"This may indicate an issue with TTS generation or audio quality. "
            f"Audio duration: {len(combined_audio) / sample_rate:.3f}s"
        ) from e

    # Extract audio from detected boundary
    extracted_audio = combined_audio[cut_sample:]

    logger.debug(
        f"Extracted: {len(extracted_audio)} samples ({len(extracted_audio) / sample_rate:.3f}s)"
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
