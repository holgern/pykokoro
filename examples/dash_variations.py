#!/usr/bin/env python3
"""
Dash variations example using pykokoro.

This example demonstrates how different types of dashes affect the reading
and prosody when separating sentence parts. It tests various dash styles
including hyphens, en dashes, em dashes, and alternatives.

Usage:
    python examples/dash_variations.py

Output:
    dash_variations_combined.wav - All variations in one file
"""

import numpy as np
import soundfile as sf

import pykokoro

# Six variations with different dash styles
# Base sentence: A complex sentence that can accommodate multiple dashes
VARIATIONS = [
    (
        "space",
        "The project was challenging the deadline was tight the team was small but we succeeded.",  # noqa: E501
    ),
    (
        "comma",
        "The project was challenging, the deadline was tight, the team was small, but we succeeded.",  # noqa: E501
    ),
    (
        "semicolon",
        "The project was challenging; the deadline was tight; the team was small; but we succeeded.",  # noqa: E501
    ),
    (
        "question",
        "The project was challenging? the deadline was tight? the team was small? but we succeeded.",  # noqa: E501
    ),
    (
        "exclamation",
        "The project was challenging! the deadline was tight! the team was small! but we succeeded.",  # noqa: E501
    ),
    (
        "point",
        "The project was challenging. the deadline was tight. the team was small. but we succeeded.",  # noqa: E501
    ),
    (
        "colon",
        "The project was challenging: the deadline was tight: the team was small: but we succeeded.",  # noqa: E501
    ),
    (
        "dash",
        "The project was challenging -- the deadline was tight -- the team was small -- but we succeeded.",  # noqa: E501
    ),
    (
        "ellipsise",
        "The project was challenging ... the deadline was tight ... the team was small ... but we succeeded.",  # noqa: E501
    ),
]

VOICE = "af_bella"  # American Female voice (good for expressive reading)
LANG = "en-us"  # American English


def main():
    """Generate speech with different dash variations."""
    print("Initializing TTS engine...")
    kokoro = pykokoro.Kokoro()

    print(f"Voice: {VOICE}")
    print(f"Language: {LANG}")
    print("\nGenerating audio for dash variations...\n")

    all_samples = []
    sample_rate_value = 24000  # Default, will be set by first create call

    for i, (variation_name, text) in enumerate(VARIATIONS, 1):
        print(f"=== Variation {i}: {variation_name} ===")
        print(f"Text: {text}")

        # Generate filler announcement
        filler_text = f"Variation {i}, {variation_name.replace('_', ' ')}."
        print(f"Filler: {filler_text}")

        filler_samples, sample_rate_value = kokoro.create(
            filler_text,
            voice=VOICE,
            speed=1.0,
            lang=LANG,
            trim_silence=True,
        )

        # Convert text to phonemes
        print("Converting to phonemes...")
        phonemes = kokoro.phonemize(text, lang=LANG)
        print(f"Phonemes: {phonemes}")

        print("Generating audio...")
        samples, sample_rate_value = kokoro.create(
            text,
            voice=VOICE,
            speed=1.0,
            lang=LANG,
            trim_silence=True,
        )

        duration = len(samples) / sample_rate_value
        print(f"Duration: {duration:.2f} seconds\n")

        # Add filler, then the actual variation
        all_samples.append(filler_samples)
        all_samples.append(samples)

        # Add a pause between variations (0.5 seconds of silence)
        if i < len(VARIATIONS):
            pause = np.zeros(int(sample_rate_value * 0.5), dtype=np.float32)
            all_samples.append(pause)

    # Combine all samples
    print("Combining all variations into one file...")
    combined_samples = np.concatenate(all_samples)

    output_file = "dash_variations_combined.wav"
    sf.write(output_file, combined_samples, sample_rate_value)

    total_duration = len(combined_samples) / sample_rate_value
    print(f"\nCreated {output_file}")
    print(f"Total duration: {total_duration:.2f} seconds")

    print("\n" + "=" * 70)
    print("DASH STYLE COMPARISON")
    print("=" * 70)
    print("\nListen for differences in:")
    print("  • Pause duration at dash positions")
    print("  • Intonation patterns (rising/falling pitch)")
    print("  • Overall flow and naturalness")
    print("  • Whether dashes are read aloud or treated as pauses")
    print()

    kokoro.close()
    print("Done! Listen to the combined file to compare all dash variations.")


if __name__ == "__main__":
    main()
