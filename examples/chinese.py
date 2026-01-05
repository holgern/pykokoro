#!/usr/bin/env python3
"""
Mandarin Chinese TTS example using pykokoro.

This example demonstrates text-to-speech synthesis in Mandarin Chinese
using the Kokoro model with Chinese voices.

Usage:
    python examples/chinese.py

Output:
    chinese_demo.wav - Generated Chinese speech audio

Available Chinese voices:
    - zf_xiaobei, zf_xiaoni, zf_xiaoxiao, zf_xiaoyi (female)
    - zm_yunjian, zm_yunxi, zm_yunxia, zm_yunyang (male)
"""

import soundfile as sf

import pykokoro

# Chinese proverb: "Learning is like rowing upstream; not to advance is to drop back."
TEXT = "学如逆水行舟，不进则退。知识就是力量，时间就是金钱。"

VOICE = "zf_xiaoxiao"  # Female Chinese voice
LANG = "zh"  # Mandarin Chinese


def main():
    """Generate Chinese speech audio."""
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

    output_file = "chinese_demo.wav"
    sf.write(output_file, samples, sample_rate)

    duration = len(samples) / sample_rate
    print(f"\nCreated {output_file}")
    print(f"Duration: {duration:.2f} seconds")

    kokoro.close()


if __name__ == "__main__":
    main()
