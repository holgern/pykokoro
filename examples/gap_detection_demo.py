#!/usr/bin/env python3
"""
Gap Detection and Manipulation Demo.

This example demonstrates:
1. Finding all silence gaps in TTS-generated audio
2. Extracting individual words/segments between gaps
3. Reconstructing sentences with custom pauses
4. Testing different gap detection parameters

The demo helps verify that find_silence_gap() works correctly
and shows how to use it for audio manipulation.

Usage:
    python examples/gap_detection_demo.py

Output:
    - gap_detection_*.wav - Various audio files for verification
    - Console output with detailed gap analysis
"""

import numpy as np
import soundfile as sf

import pykokoro
from pykokoro.short_sentence_handler import energy_based_vad, find_silence_gap


def find_all_gaps(
    audio: np.ndarray,
    sample_rate: int,
    energy_threshold: float = 0.05,
    min_frames: int = 2,
) -> list[tuple[int, int, int]]:
    """Find all silence gaps in audio and return detailed information.

    Args:
        audio: Audio signal as numpy array
        sample_rate: Sample rate of audio
        energy_threshold: Energy threshold for VAD (0.0-1.0)
        min_frames: Minimum consecutive silent frames to consider a gap

    Returns:
        List of tuples: (gap_index, gap_start_sample, gap_length_frames)
    """
    gaps = []
    gap_idx = 1

    # Get voice activity for analysis
    voice_activity = energy_based_vad(
        audio, sample_rate, frame_duration_ms=20, energy_threshold=energy_threshold
    )

    # Find all gaps by detecting transitions
    in_speech = False
    gap_start_frame = None
    frame_duration_ms = 20
    samples_per_frame = int(sample_rate * frame_duration_ms / 1000)

    for i, is_speech in enumerate(voice_activity):
        if is_speech and not in_speech:
            # Start of speech
            in_speech = True
        elif not is_speech and in_speech:
            # Start of silence after speech
            gap_start_frame = i
            in_speech = False
        elif is_speech and gap_start_frame is not None:
            # End of silence gap
            gap_length = i - gap_start_frame

            # Check if gap is long enough
            if gap_length >= min_frames:
                gap_start_sample = gap_start_frame * samples_per_frame
                gaps.append((gap_idx, gap_start_sample, gap_length))
                gap_idx += 1

            # Reset for next potential gap
            gap_start_frame = None
            in_speech = True

    return gaps


def extract_segments(
    audio: np.ndarray, gaps: list[tuple[int, int, int]], sample_rate: int
) -> list[np.ndarray]:
    """Extract audio segments between gaps.

    Args:
        audio: Audio signal as numpy array
        gaps: List of gaps from find_all_gaps()
        sample_rate: Sample rate of audio

    Returns:
        List of audio segments (numpy arrays)
    """
    segments = []

    if not gaps:
        # No gaps found, return entire audio
        return [audio]

    # Extract segment before first gap
    first_gap_start = gaps[0][1]
    if first_gap_start > 0:
        segments.append(audio[:first_gap_start])

    # Extract segments between gaps
    for i in range(len(gaps) - 1):
        gap_start = gaps[i][1]
        next_gap_start = gaps[i + 1][1]
        segments.append(audio[gap_start:next_gap_start])

    # Extract segment after last gap
    last_gap_start = gaps[-1][1]
    if last_gap_start < len(audio):
        segments.append(audio[last_gap_start:])

    return segments


def reconstruct_with_pauses(
    segments: list[np.ndarray], pauses_ms: list[float], sample_rate: int
) -> np.ndarray:
    """Reconstruct audio with custom pause durations.

    Args:
        segments: List of audio segments
        pauses_ms: List of pause durations in milliseconds (len = len(segments) - 1)
        sample_rate: Sample rate of audio

    Returns:
        Reconstructed audio as numpy array
    """
    if len(segments) == 0:
        return np.array([], dtype=np.float32)

    if len(segments) == 1:
        return segments[0]

    # Build reconstructed audio
    reconstructed = []

    for i, segment in enumerate(segments):
        reconstructed.append(segment)

        # Add pause if not the last segment
        if i < len(segments) - 1:
            pause_duration = pauses_ms[i] if i < len(pauses_ms) else 0
            pause_samples = int(sample_rate * pause_duration / 1000)
            pause = np.zeros(pause_samples, dtype=np.float32)
            reconstructed.append(pause)

    return np.concatenate(reconstructed)


def print_gap_table(
    gaps: list[tuple[int, int, int]], sample_rate: int, frame_duration_ms: int = 20
):
    """Print a formatted table of gap information."""
    if not gaps:
        print("  No gaps found!")
        return

    print("\n  Gap # | Start Time | Start Sample | Length (frames) | Length (ms)")
    print("  ------|------------|--------------|-----------------|-------------")

    for gap_idx, gap_start_sample, gap_length_frames in gaps:
        start_time_s = gap_start_sample / sample_rate
        gap_length_ms = gap_length_frames * frame_duration_ms
        print(
            f"    {gap_idx:1d}   |   {start_time_s:6.3f}s  | {gap_start_sample:12d} | "
            f"{gap_length_frames:15d} | {gap_length_ms:11.0f}ms"
        )


def print_segments_info(
    segments: list[np.ndarray], gaps: list[tuple[int, int, int]], sample_rate: int
):
    """Print information about extracted segments."""
    print("\nSegments extracted:")

    if not segments:
        print("  No segments found!")
        return

    start_time = 0.0
    for i, segment in enumerate(segments):
        duration_s = len(segment) / sample_rate
        end_time = start_time + duration_s
        print(
            f"  Segment {i + 1}: {start_time:6.3f}s - {end_time:6.3f}s "
            f"({len(segment):6d} samples, {duration_s:5.3f}s)"
        )

        # Move start_time past this segment and the gap after it (if any)
        if i < len(gaps):
            gap_length_frames = gaps[i][2]
            gap_duration_s = (gap_length_frames * 20) / 1000  # 20ms frame duration
            start_time = end_time + gap_duration_s
        else:
            start_time = end_time


def test_parameter_sensitivity(audio: np.ndarray, sample_rate: int, text: str):
    """Test different energy_threshold and min_frames values."""
    print("\n" + "=" * 70)
    print("PARAMETER SENSITIVITY TESTING")
    print("=" * 70)

    print(f'\nTest text: "{text}"')
    print(f"Audio duration: {len(audio) / sample_rate:.3f}s ({len(audio)} samples)")

    # Test different energy thresholds
    print("\n--- Testing Energy Thresholds (min_frames=2) ---\n")
    for energy_thresh in [0.01, 0.02, 0.05, 0.08, 0.10]:
        gaps = find_all_gaps(
            audio, sample_rate, energy_threshold=energy_thresh, min_frames=2
        )
        print(f"  energy_threshold={energy_thresh:.2f}: {len(gaps):2d} gaps found")
        if gaps and len(gaps) <= 5:
            print(f"    Gap positions: {[f'{g[1] / sample_rate:.3f}s' for g in gaps]}")

    # Test different min_frames
    print("\n--- Testing Minimum Frame Counts (energy_threshold=0.05) ---\n")
    for min_f in [1, 2, 5, 8, 10]:
        gaps = find_all_gaps(
            audio, sample_rate, energy_threshold=0.05, min_frames=min_f
        )
        print(f"  min_frames={min_f:2d}: {len(gaps):2d} gaps found")
        if gaps and len(gaps) <= 5:
            print(f"    Gap positions: {[f'{g[1] / sample_rate:.3f}s' for g in gaps]}")


def analyze_voice_activity(
    audio: np.ndarray, sample_rate: int, energy_threshold: float = 0.05
):
    """Analyze and print voice activity detection results."""
    print("\n--- Voice Activity Analysis ---\n")

    va = energy_based_vad(
        audio, sample_rate, frame_duration_ms=20, energy_threshold=energy_threshold
    )

    total_frames = len(va)
    speech_frames = va.sum()
    silence_frames = total_frames - speech_frames

    print(f"  Total frames: {total_frames}")
    print(
        f"  Speech frames: {speech_frames} ({speech_frames / total_frames * 100:.1f}%)"
    )
    print(
        f"  Silence frames: {silence_frames} ({silence_frames / total_frames * 100:.1f}%)"
    )
    print(f"  Frame duration: 20ms")
    print(f"  Total duration: {total_frames * 20 / 1000:.3f}s")


def main():
    """Main demo function."""
    print("=" * 70)
    print("GAP DETECTION AND MANIPULATION DEMO")
    print("=" * 70)

    print("\nInitializing PyKokoro TTS engine...")
    kokoro = pykokoro.Kokoro()

    # Test cases with varying pause patterns
    test_cases = [
        {
            "text": "Hello world. This is a test. How are you?",
            "description": "Three sentences with natural pauses",
            "voice": "af_bella",
            "lang": "en-us",
        },
        {
            "text": "One, two, three, four, five.",
            "description": "Comma-separated list",
            "voice": "af_bella",
            "lang": "en-us",
        },
        {
            "text": "Why?",
            "description": "Single short word",
            "voice": "af_bella",
            "lang": "en-us",
        },
    ]

    for case_num, test_case in enumerate(test_cases, 1):
        text = test_case["text"]
        description = test_case["description"]
        voice = test_case["voice"]
        lang = test_case["lang"]

        print("\n\n" + "=" * 70)
        print(f"TEST CASE {case_num}: {description}")
        print("=" * 70)
        print(f'Text: "{text}"')
        print(f"Voice: {voice}, Language: {lang}")

        # Generate TTS audio
        print("\nGenerating TTS audio...")
        audio, sample_rate = kokoro.create(text, voice=voice, lang=lang)

        # Save original audio
        original_file = f"gap_detection_case{case_num}_original.wav"
        sf.write(original_file, audio, sample_rate)
        print(f"  Saved: {original_file}")
        print(f"  Duration: {len(audio) / sample_rate:.3f}s ({len(audio)} samples)")

        # Analyze voice activity
        analyze_voice_activity(audio, sample_rate, energy_threshold=0.05)

        # Find all gaps with default parameters
        print("\n--- Gap Detection (energy_threshold=0.05, min_frames=2) ---")
        gaps = find_all_gaps(audio, sample_rate, energy_threshold=0.05, min_frames=2)
        print(f"\nFound {len(gaps)} gap(s)")
        print_gap_table(gaps, sample_rate)

        # Extract segments
        if gaps:
            print("\n--- Extracting Segments ---")
            segments = extract_segments(audio, gaps, sample_rate)
            print(f"Extracted {len(segments)} segment(s)")
            print_segments_info(segments, gaps, sample_rate)

            # Save individual segments
            for i, segment in enumerate(segments):
                segment_file = f"gap_detection_case{case_num}_segment{i + 1}.wav"
                sf.write(segment_file, segment, sample_rate)
                print(f"  Saved: {segment_file}")

            # Reconstruct with custom pauses
            if len(segments) > 1:
                print("\n--- Reconstructing with Custom Pauses ---")

                # Use different pause lengths: short, medium, long
                custom_pauses = []
                pause_options = [100, 300, 600]  # ms
                for i in range(len(segments) - 1):
                    custom_pauses.append(pause_options[i % len(pause_options)])

                print(f"Custom pause durations (ms): {custom_pauses}")

                reconstructed = reconstruct_with_pauses(
                    segments, custom_pauses, sample_rate
                )
                reconstructed_file = f"gap_detection_case{case_num}_reconstructed.wav"
                sf.write(reconstructed_file, reconstructed, sample_rate)

                print(f"  Original duration: {len(audio) / sample_rate:.3f}s")
                print(
                    f"  Reconstructed duration: {len(reconstructed) / sample_rate:.3f}s"
                )
                print(f"  Saved: {reconstructed_file}")

        # Test parameter sensitivity (only for first test case)
        if case_num == 1:
            test_parameter_sensitivity(audio, sample_rate, text)

    # Close TTS engine
    kokoro.close()

    print("\n\n" + "=" * 70)
    print("DEMO COMPLETE")
    print("=" * 70)
    print("\nFiles created:")
    print("  - gap_detection_case*_original.wav - Original TTS audio")
    print("  - gap_detection_case*_segment*.wav - Individual segments")
    print("  - gap_detection_case*_reconstructed.wav - Rebuilt with custom pauses")
    print("\nListen to the audio files to verify gap detection accuracy!")


if __name__ == "__main__":
    main()
