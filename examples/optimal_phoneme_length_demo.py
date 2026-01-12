#!/usr/bin/env python3
"""
Optimal Phoneme Length Demonstration.

This example demonstrates how the optimal_phoneme_length parameter improves
audio quality for dialogue with very short sentences. Short phoneme sequences
(like "Why?" = 3 phonemes) can produce poor audio quality when processed
individually. Batching them together provides more context for natural prosody.

The example compares three processing modes:
1. No batching (default): Each sentence processed separately
2. Single target (50): Batch until reaching ~50 phonemes
3. Array targets ([30, 50, 70]): Flexible batching with multiple thresholds

Usage:
    python examples/optimal_phoneme_length_demo.py

Output:
    optimal_phoneme_length_comparison.wav - All three versions in one file
    Detailed console output showing phoneme counts and segment structure
"""

import numpy as np
import soundfile as sf

import pykokoro

# Dialogue with mix of very short and normal sentences
DIALOGUE_TEXT = """
"Why?" she asked.

"Do it!" he commanded.

"Go!" they shouted.

"I know." she whispered.

He sits quietly.

She nods slowly.

"Really?" he questioned.

"Yes." she confirmed.

"The quick brown fox jumps over the lazy dog." he said with a smile.

"That's wonderful news!" she exclaimed happily.
"""

VOICE = "af_bella"  # American Female voice
LANG = "en-us"


def print_separator(title: str) -> None:
    """Print a visual separator with title."""
    print("\n" + "=" * 70)
    print(f" {title}")
    print("=" * 70)


def analyze_segments(segments: list, title: str) -> dict:
    """Analyze and print segment statistics."""
    print_separator(title)

    phoneme_counts = []
    total_phonemes = 0

    for i, seg in enumerate(segments, 1):
        phoneme_length = len(seg.phonemes)
        phoneme_counts.append(phoneme_length)
        total_phonemes += phoneme_length

        # Show first 3 and last 1 segments
        if i <= 3 or i == len(segments):
            print(f"\nSegment {i}:")
            print(f"  Text: '{seg.text[:60]}{'...' if len(seg.text) > 60 else ''}'")
            print(f"  Phonemes: {phoneme_length} chars")
            print(f"  Sentence: {seg.sentence}, Paragraph: {seg.paragraph}")
            if i == 3 and len(segments) > 4:
                print(f"\n  ... ({len(segments) - 4} more segments) ...")

    avg_phonemes = total_phonemes / len(segments) if segments else 0
    min_phonemes = min(phoneme_counts) if phoneme_counts else 0
    max_phonemes = max(phoneme_counts) if phoneme_counts else 0

    print("\nStatistics:")
    print(f"  Total segments: {len(segments)}")
    print(f"  Total phonemes: {total_phonemes}")
    print(f"  Average phonemes per segment: {avg_phonemes:.1f}")
    print(f"  Min/Max phonemes: {min_phonemes}/{max_phonemes}")

    return {
        "segments": len(segments),
        "total_phonemes": total_phonemes,
        "avg_phonemes": avg_phonemes,
        "min_phonemes": min_phonemes,
        "max_phonemes": max_phonemes,
        "phoneme_counts": phoneme_counts,
    }


def generate_version(
    kokoro,
    title: str,
    optimal_phoneme_length,
) -> tuple[np.ndarray, dict]:
    """Generate audio for one version and return samples + stats."""
    # Generate audio
    samples, sample_rate = kokoro.create(
        DIALOGUE_TEXT,
        voice=VOICE,
        lang=LANG,
        speed=1.0,
        # Use default pause_mode="tts" for natural prosody
        optimal_phoneme_length=optimal_phoneme_length,
    )

    # Analyze segments separately for display
    segments = pykokoro.phonemes.text_to_phoneme_segments(
        text=DIALOGUE_TEXT,
        tokenizer=kokoro.tokenizer,
        lang=LANG,
        pause_mode="tts",
        optimal_phoneme_length=optimal_phoneme_length,
    )

    stats = analyze_segments(segments, title)

    duration = len(samples) / sample_rate
    print(f"  Audio duration: {duration:.2f} seconds")

    return samples, sample_rate, stats


def main():
    """Generate comparison demo."""
    print_separator("OPTIMAL PHONEME LENGTH DEMONSTRATION")
    print("\nThis demo compares three processing approaches:")
    print("  1. No batching (default): Each sentence separate")
    print("  2. Single target (50): Batch until ~50 phonemes")
    print("  3. Array targets ([30, 50, 70]): Flexible multi-threshold batching")
    print("\nDialogue text has mix of very short (3-8 phonemes) and")
    print("normal (25-47 phonemes) sentences.")

    kokoro = pykokoro.Kokoro()
    all_samples = []
    sample_rate_value = 24000
    all_stats = {}

    # Version 1: No batching
    print_separator("Generating Version 1: No Batching")
    samples1, sample_rate_value, stats1 = generate_version(
        kokoro,
        "Version 1: No Batching (optimal_phoneme_length=None)",
        optimal_phoneme_length=None,
    )

    # Add announcement
    announcement1 = "Version one: No batching, each sentence separate."
    filler1, _ = kokoro.create(announcement1, voice=VOICE, lang=LANG)
    all_samples.extend([filler1, samples1])
    all_stats["no_batching"] = stats1

    # Add pause
    pause = np.zeros(int(sample_rate_value * 1.0), dtype=np.float32)
    all_samples.append(pause)

    # Version 2: Single target
    print_separator("Generating Version 2: Single Target (50)")
    samples2, sample_rate_value, stats2 = generate_version(
        kokoro,
        "Version 2: Single Target (optimal_phoneme_length=50)",
        optimal_phoneme_length=50,
    )

    announcement2 = "Version two: Single target of fifty phonemes."
    filler2, _ = kokoro.create(announcement2, voice=VOICE, lang=LANG)
    all_samples.extend([filler2, samples2])
    all_stats["single_target"] = stats2

    # Add pause
    all_samples.append(pause)

    # Version 3: Array targets
    print_separator("Generating Version 3: Array Targets ([30, 50, 70])")
    samples3, sample_rate_value, stats3 = generate_version(
        kokoro,
        "Version 3: Array Targets (optimal_phoneme_length=[30, 50, 70])",
        optimal_phoneme_length=[30, 50, 70],
    )

    announcement3 = (
        "Version three: Array targets of thirty, fifty, and seventy phonemes."
    )
    filler3, _ = kokoro.create(announcement3, voice=VOICE, lang=LANG)
    all_samples.extend([filler3, samples3])
    all_stats["array_targets"] = stats3

    # Combine all audio
    print_separator("Combining All Versions")
    combined_samples = np.concatenate(all_samples)

    output_file = "optimal_phoneme_length_comparison.wav"
    sf.write(output_file, combined_samples, sample_rate_value)

    total_duration = len(combined_samples) / sample_rate_value
    print(f"\nCreated {output_file}")
    print(f"Total duration: {total_duration:.2f}s ({total_duration / 60:.2f} minutes)")

    # Comparison summary
    print_separator("COMPARISON SUMMARY")

    print(f"\n{'Version':<25} {'Segments':<12} {'Avg Phonemes':<15} {'Min/Max':<15}")
    print("-" * 70)

    print(
        f"{'No Batching':<25} "
        f"{all_stats['no_batching']['segments']:<12} "
        f"{all_stats['no_batching']['avg_phonemes']:<15.1f} "
        f"{all_stats['no_batching']['min_phonemes']}/{all_stats['no_batching']['max_phonemes']}"
    )

    print(
        f"{'Single Target (50)':<25} "
        f"{all_stats['single_target']['segments']:<12} "
        f"{all_stats['single_target']['avg_phonemes']:<15.1f} "
        f"{all_stats['single_target']['min_phonemes']}/{all_stats['single_target']['max_phonemes']}"
    )

    print(
        f"{'Array ([30, 50, 70])':<25} "
        f"{all_stats['array_targets']['segments']:<12} "
        f"{all_stats['array_targets']['avg_phonemes']:<15.1f} "
        f"{all_stats['array_targets']['min_phonemes']}/{all_stats['array_targets']['max_phonemes']}"
    )

    print("\nKey Observations:")
    print("  • No Batching: Many very short segments (poor quality risk)")
    print("  • Single Target: Fewer, more consistent segments")
    print("  • Array Targets: Flexible batching with multiple thresholds")
    print("\nRecommendation:")
    print("  • Use optimal_phoneme_length=50 for dialogue-heavy text")
    print("  • Use optimal_phoneme_length=[30, 50] for mixed content")
    print("  • Omit parameter (None) for naturally long sentences")

    print("\nListen to the WAV file to compare audio quality!")
    print("Notice how batched versions have better prosody and flow.")

    kokoro.close()


if __name__ == "__main__":
    main()
