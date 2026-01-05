#!/usr/bin/env python3
"""
Hindi TTS example using pykokoro.

This example demonstrates text-to-speech synthesis in Hindi
using the Kokoro model with Hindi voices.

Usage:
    python examples/hindi.py

Output:
    hindi_demo.wav - Generated Hindi speech audio

Available Hindi voices:
    - hf_alpha, hf_beta (female)
    - hm_omega, hm_psi (male)
"""

import soundfile as sf

import pykokoro

# Hindi wisdom about life and perseverance
TEXT = (
    "जीवन में सफलता उन्हें मिलती है जो कभी हार नहीं मानते। "
    "हर सुबह एक नई शुरुआत है, हर दिन एक नया अवसर है।"
)

VOICE = "hf_alpha"  # Hindi Female voice
LANG = "hi"  # Hindi


def main():
    """Generate Hindi speech audio."""
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

    output_file = "hindi_demo.wav"
    sf.write(output_file, samples, sample_rate)

    duration = len(samples) / sample_rate
    print(f"\nCreated {output_file}")
    print(f"Duration: {duration:.2f} seconds")

    kokoro.close()


if __name__ == "__main__":
    main()
