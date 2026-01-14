#!/usr/bin/env python3
"""
Short Sentence Handler Demonstration.

This example demonstrates the cross-correlation extraction technique used by PyKokoro
to improve audio quality for very short sentences.

The short sentence handler:
1. Detects sentences with fewer phonemes than a threshold (default: 10)
2. Generates the short sentence alone (poor quality, but needed for pattern)
3. Generates context + short sentence together (good quality with natural prosody)
4. Uses cross-correlation to find where the short sentence appears in combined audio
5. Extracts that portion from the combined audio (maintains high quality)

This produces higher-quality audio because neural TTS models typically need
more context to produce natural-sounding speech with proper prosody and intonation.
The cross-correlation approach is robust and doesn't depend on silence gap detection.

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

# Enable debug logging to see detailed processing information
# logging.basicConfig(
#    level=logging.DEBUG, format="%(levelname)s [%(name)s] - %(message)s"
# )

# Test sentences of varying lengths
TEST_SENTENCES = [
    # Very short (will trigger repeat-and-cut)
    "Hi!",
    "Why?",
    "No.",
    "No!",
    "Yes!",
    "Help!",
    "Oh!",
    "Stop!",
    "What?",
    "Don't!",
]
TEST_SENTENCE = TEST_SENTENCES[2]
# Voice to use

VOICES = [
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
]
VOICE = "af_sarah"  # Changed from af_bella for better short sentence results

LANG = "en-us"


def print_separator(title: str) -> None:
    """Print a visual separator with title."""
    print("\n" + "=" * 70)
    print(f" {title}")
    print("=" * 70)


def test_sentence_with_config(
    kokoro: pykokoro.Kokoro,
    voice: str,
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

    samples, sr = kokoro_test.create(TEST_SENTENCE, voice=voice, lang=LANG)

    print(f"  {config_name:25} -> {len(samples):6} samples ({len(samples) / sr:.3f}s)")

    kokoro_test.close()
    return samples, sr


def main():
    """Generate audio demonstrating short sentence handling."""
    print_separator("SHORT SENTENCE HANDLER DEMONSTRATION")

    print("\nThis demo shows how PyKokoro improves audio quality for short sentences")
    print("using cross-correlation extraction with context.")
    print(f"\nText: {TEST_SENTENCE}")
    print(f"Language: {LANG}")
    print("\nNOTE: Audio duration will be similar, but QUALITY will be better")
    print("      with context-prepending. Listen to the generated files to compare!")

    # Initialize with default config
    print_separator("Testing Individual Sentences")

    kokoro = pykokoro.Kokoro()

    all_samples = []
    all_samples2 = []
    sample_rate = 24000

    pause = np.zeros(int(sample_rate * 0.5), dtype=np.float32)
    # Add announcement and samples to output
    announcement = "With pretexting"
    intro, _ = kokoro.create(announcement, voice=VOICE, lang=LANG)
    all_samples.extend([intro, pause])
    # Add announcement and samples to output
    announcement = "Without pretexting"
    intro2, _ = kokoro.create(announcement, voice=VOICE, lang=LANG)
    all_samples2.extend([pause, intro2, pause])

    # Test each sentence with different configurations
    for voice in VOICES:
        phoneme_count = len(kokoro.tokenizer.phonemize(TEST_SENTENCE, lang=LANG))

        print(f"\nVoice: '{voice}' ({phoneme_count} phonemes)")

        # Test with context-prepending enabled (default)
        config_enabled = ShortSentenceConfig(
            min_phoneme_length=10,
            disable_cutoff_detection=True,
            enabled=True,
        )

        # Test with context-prepending disabled
        config_disabled = ShortSentenceConfig(enabled=False)

        # Generate with both configs
        samples_enabled, sr = test_sentence_with_config(
            kokoro, voice, config_enabled, "With prepending"
        )

        samples_disabled, sr = test_sentence_with_config(
            kokoro, voice, config_disabled, "Without prepending"
        )
        # Add: intro + enabled version + pause + disabled version + pause
        pause = np.zeros(int(sr * 0.1), dtype=np.float32)
        # all_samples.extend([samples_enabled, pause, samples_disabled, pause])
        all_samples.extend([samples_enabled, pause])
        all_samples2.extend([samples_disabled, pause])

    # Save combined audio
    print_separator("Saving Combined Audio")

    combined_samples = np.concatenate(all_samples + all_samples2)
    output_file = "short_sentence_voices_demo.wav"
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
        "  config = ShortSentenceConfig("
        "min_phoneme_length=15, target_phoneme_length=40)"
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
