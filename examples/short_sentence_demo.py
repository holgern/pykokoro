#!/usr/bin/env python3
"""
Short Sentence Handler Demonstration.

This example demonstrates the context-prepending technique used by PyKokoro
to improve audio quality for very short sentences.

The short sentence handler:
1. Detects sentences with fewer phonemes than a threshold (default: 10)
2. Prepends a context sentence before the short sentence
3. Generates TTS for the combined text (better quality with more context)
4. Detects the silence gap between context and target
5. Extracts audio from after the gap to get only the target sentence

This produces higher-quality audio because neural TTS models typically need
more context to produce natural-sounding speech with proper prosody and intonation.

Usage:
    python examples/short_sentence_demo.py

Output:
    short_sentence_demo.wav - Audio demonstrating short sentence handling
    Detailed console output showing processing steps
"""

import numpy as np
import soundfile as sf

import pykokoro
from pykokoro.short_sentence_handler import ShortSentenceConfig

# Test sentences of varying lengths
TEST_SENTENCES = [
    # Very short (will trigger repeat-and-cut)
    "Hi!",
    "Why?",
    "No.",
    "Yes!",
    "Help!",
    # Short but complete (might trigger depending on phoneme count)
    "Hello there.",
    "Good morning.",
    "Thank you.",
    "I agree.",
    # Normal length (won't trigger)
    "This is a normal sentence.",
    "The quick brown fox jumps over the lazy dog.",
]

# Voice to use
VOICE = "af_bella"
LANG = "en-us"


def print_separator(title: str) -> None:
    """Print a visual separator with title."""
    print("\n" + "=" * 70)
    print(f" {title}")
    print("=" * 70)


def test_sentence_with_config(
    kokoro: pykokoro.Kokoro,
    text: str,
    config: ShortSentenceConfig | None,
    config_name: str,
) -> tuple[np.ndarray, int]:
    """Generate audio for a sentence with a specific config.

    Args:
        kokoro: Kokoro instance
        text: Text to generate
        config: Short sentence configuration (or None to disable)
        config_name: Name for logging

    Returns:
        Tuple of (audio samples, sample rate)
    """
    # Create a new Kokoro instance with the config
    kokoro_test = pykokoro.Kokoro(short_sentence_config=config)

    samples, sr = kokoro_test.create(text, voice=VOICE, lang=LANG)

    print(f"  {config_name:25} -> {len(samples):6} samples ({len(samples) / sr:.3f}s)")

    kokoro_test.close()
    return samples, sr


def main():
    """Generate audio demonstrating short sentence handling."""
    print_separator("SHORT SENTENCE HANDLER DEMONSTRATION")

    print("\nThis demo shows how PyKokoro improves audio quality for short sentences")
    print("using the 'repeat-and-cut' technique.")
    print(f"\nVoice: {VOICE}")
    print(f"Language: {LANG}")

    # Initialize with default config
    print_separator("Testing Individual Sentences")

    kokoro = pykokoro.Kokoro()

    all_samples = []
    sample_rate = 24000

    # Test each sentence with different configurations
    for text in TEST_SENTENCES:
        phoneme_count = len(kokoro.tokenizer.phonemize(text, lang=LANG))

        print(f"\nText: '{text}' ({phoneme_count} phonemes)")

        # Test with context-prepending enabled (default)
        config_enabled = ShortSentenceConfig(
            min_phoneme_length=10,
            enabled=True,
        )

        # Test with context-prepending disabled
        config_disabled = ShortSentenceConfig(enabled=False)

        # Generate with both configs
        samples_enabled, sr = test_sentence_with_config(
            kokoro, text, config_enabled, "With context-prepending"
        )

        samples_disabled, sr = test_sentence_with_config(
            kokoro, text, config_disabled, "Without context-prepending"
        )

        # Add announcement and samples to output
        # announcement = f"The sentence: {text}"
        # intro, _ = kokoro.create(announcement, voice=VOICE, lang=LANG)

        # Add: intro + enabled version + pause + disabled version + pause
        pause = np.zeros(int(sr * 0.5), dtype=np.float32)
        all_samples.extend(
            [samples_enabled, pause, samples_disabled, pause]
        )

    # Configuration comparison
    print_separator("Configuration Comparison")

    test_text = "Why?"
    phonemes = kokoro.tokenizer.phonemize(test_text, lang=LANG)
    phoneme_count = len(phonemes)

    print(f"\nTest sentence: '{test_text}' ({phoneme_count} phonemes)")
    print(f"Phonemes: {phonemes}")
    print()

    configs = [
        ("Disabled", ShortSentenceConfig(enabled=False)),
        ("Default (min=10)", ShortSentenceConfig()),
        (
            "Aggressive (min=20)",
            ShortSentenceConfig(min_phoneme_length=20),
        ),
        (
            "Conservative (min=5)",
            ShortSentenceConfig(min_phoneme_length=5),
        ),
    ]

    print("Comparing different configurations:")
    for name, config in configs:
        samples, sr = test_sentence_with_config(kokoro, test_text, config, name)

    # Save combined audio
    print_separator("Saving Combined Audio")

    combined_samples = np.concatenate(all_samples)
    output_file = "short_sentence_demo.wav"
    sf.write(output_file, combined_samples, sample_rate)

    total_duration = len(combined_samples) / sample_rate
    print(f"\nCreated {output_file}")
    print(f"Total duration: {total_duration:.2f}s ({total_duration / 60:.2f} minutes)")

    # Summary
    print_separator("SUMMARY")

    print("\nHow the Short Sentence Handler Works:")
    print("  1. Detects sentences with < min_phoneme_length phonemes")
    print("  2. Generates the short sentence alone to measure duration")
    print("  3. Repeats the text to reach target_phoneme_length")
    print("  4. Generates TTS for repeated text (better quality)")
    print("  5. Cuts at measured duration + 15% safety buffer")

    print("\nBenefits:")
    print("  • Improved prosody and intonation for short sentences")
    print("  • More natural-sounding speech")
    print("  • Better handling of single-word sentences")

    print("\nConfiguration Options:")
    print("  • min_phoneme_length: Threshold for 'short' (default: 10)")
    print("  • target_phoneme_length: Target for repetition (default: 30)")
    print("  • max_repetitions: Max times to repeat (default: 5)")
    print("  • enabled: Enable/disable the feature (default: True)")

    print("\nUsage:")
    print("  # Custom configuration")
    print(
        "  config = ShortSentenceConfig(min_phoneme_length=15, target_phoneme_length=40)"
    )
    print("  kokoro = Kokoro(short_sentence_config=config)")
    print()
    print("  # Disable short sentence handling")
    print("  config = ShortSentenceConfig(enabled=False)")
    print("  kokoro = Kokoro(short_sentence_config=config)")

    print("\n" + "=" * 70)
    print("Listen to the WAV file to hear the difference!")
    print("=" * 70)

    kokoro.close()


if __name__ == "__main__":
    main()
