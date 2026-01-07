#!/usr/bin/env python3
"""
Mandarin Chinese TTS example using pykokoro.

This example demonstrates text-to-speech synthesis in Mandarin Chinese
using the Kokoro v1.1-zh model with Chinese voices.

Usage:
    python examples/chinese.py

Output:
    chinese_demo.wav - Generated Chinese speech audio

Available Chinese voices (v1.1-zh):
    - Female: zf_001 through zf_099 (55 voices)
    - Male: zm_009 through zm_100 (45 voices)

Note: v1.1-zh uses numbered voices instead of named voices.
For a full list, see: https://huggingface.co/hexgrad/Kokoro-82M-v1.1-zh
"""

import soundfile as sf

import pykokoro

# Chinese proverb: "Learning is like rowing upstream; not to advance is to drop back."
TEXT = "学如逆水行舟，不进则退。知识就是力量，时间就是金钱。"

VOICE = "zf_001"  # Female Chinese voice (numbered)
LANG = "zh"  # Mandarin Chinese


def main():
    """Generate Chinese speech audio."""
    print("Initializing TTS engine...")
    # Use GitHub v1.1-zh model for proper Chinese support with Bopomofo phonemes
    # The model will automatically use the v1.1-zh vocabulary (178 tokens)
    # which includes Bopomofo characters needed for Chinese phonemization
    kokoro = pykokoro.Kokoro(
        model_source="github", model_variant="v1.1-zh", model_quality="fp32"
    )

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

    output_file = "chinese_demo.wav"
    sf.write(output_file, samples, sample_rate)

    duration = len(samples) / sample_rate
    print(f"\nCreated {output_file}")
    print(f"Duration: {duration:.2f} seconds")

    kokoro.close()


if __name__ == "__main__":
    main()
