#!/usr/bin/env python3
"""
Spanish TTS example using pykokoro.

This example demonstrates text-to-speech synthesis in Spanish
using the Kokoro model with Spanish voices.

Usage:
    python examples/spanish.py

Output:
    spanish_demo.wav - Generated Spanish speech audio

Available Spanish voices:
    - ef_dora (female)
    - em_alex, em_santa (male)
"""

import soundfile as sf

import pykokoro

# Spanish quote about life and dreams (inspired by Don Quixote)
TEXT = (
    "La vida es un viaje maravilloso lleno de aventuras. "
    "Quien no se atreve a sonar, nunca vera sus suenos hacerse realidad. "
    "Cada dia es una nueva oportunidad para ser mejor que ayer."
)

VOICE = "ef_dora"  # Spanish Female voice
LANG = "es"  # Spanish


def main():
    """Generate Spanish speech audio."""
    print("Initializing TTS engine...")
    kokoro = pykokoro.Kokoro()

    print(f"Text: {TEXT}")
    print(f"Voice: {VOICE}")
    print(f"Language: {LANG}")

    print("\nGenerating audio...")
    samples, sample_rate = kokoro.create(
        TEXT,
        voice=VOICE,
        speed=1.0,
        lang=LANG,
    )

    output_file = "spanish_demo.wav"
    sf.write(output_file, samples, sample_rate)

    duration = len(samples) / sample_rate
    print(f"\nCreated {output_file}")
    print(f"Duration: {duration:.2f} seconds")

    kokoro.close()


if __name__ == "__main__":
    main()
