#!/usr/bin/env python3
"""
Podcast-style multi-voice conversation example using pykokoro with SSMD voice markup.

This example demonstrates how to create a podcast with multiple speakers
using SSMD voice annotations. Each speaker's dialogue is marked with SSMD
voice syntax, allowing natural inline voice switching.

Two approaches are shown:
1. Traditional approach: Generate each segment separately with different voices
2. SSMD approach: Use inline voice annotations [text](voice: name) in a single text

Usage:
    python examples/podcast.py

Output:
    podcast_demo.wav - Traditional multi-voice podcast
    podcast_ssmd_demo.wav - SSMD-based podcast with inline voice switching
"""

import random

import numpy as np
import soundfile as sf

import pykokoro

# Traditional Podcast script with separate voice segments
PODCAST_SCRIPT = [
    {
        "voice": "af_sarah",
        "text": "Welcome to Tech Talk! I'm Sarah, and today we're diving into "
        "the fascinating world of text-to-speech technology.",
    },
    {
        "voice": "am_michael",
        "text": "And I'm Michael! We've got an amazing episode lined up. "
        "The advances in neural TTS have been incredible lately.",
    },
    {
        "voice": "af_sarah",
        "text": "Absolutely! And we have a special guest with us today. "
        "Please welcome our AI researcher, Nicole!",
    },
    {
        "voice": "af_nicole",
        "text": "Thanks for having me! I'm thrilled to be here. "
        "I've been working on voice synthesis for the past five years.",
    },
    {
        "voice": "am_michael",
        "text": "Nicole, can you tell us about the latest breakthroughs "
        "in making synthetic voices sound more natural?",
    },
    {
        "voice": "af_nicole",
        "text": "Of course! The key innovation has been in capturing "
        "prosody and emotional nuance. Modern models like Kokoro "
        "can generate speech that's nearly indistinguishable from human voices.",
    },
    {
        "voice": "af_sarah",
        "text": "That's fascinating! What do you see as the main applications "
        "for this technology?",
    },
    {
        "voice": "af_nicole",
        "text": "There are so many! Audiobook production, accessibility tools, "
        "language learning, and even preserving voices of people "
        "who might lose their ability to speak.",
    },
    {
        "voice": "am_michael",
        "text": "The accessibility angle is really compelling. "
        "Imagine being able to give a voice to those who can't speak.",
    },
    {
        "voice": "af_sarah",
        "text": "Exactly! And with open-source models, this technology "
        "is becoming available to everyone.",
    },
    {
        "voice": "af_nicole",
        "text": "That's what excites me most. Democratizing access to "
        "high-quality speech synthesis opens up so many possibilities.",
    },
    {
        "voice": "am_michael",
        "text": "Well, this has been an enlightening discussion! "
        "Any final thoughts, Nicole?",
    },
    {
        "voice": "af_nicole",
        "text": "Just that we're at an inflection point. "
        "The next few years will bring even more amazing developments. "
        "Stay curious!",
    },
    {
        "voice": "af_sarah",
        "text": "Thank you so much for joining us, Nicole! "
        "And thank you to our listeners for tuning in.",
    },
    {
        "voice": "am_michael",
        "text": "Until next time, keep exploring the future of technology!",
    },
]

# SSMD-based podcast script with inline voice switching
# Using SSMD voice annotation syntax: [text](voice: name)
SSMD_PODCAST_SCRIPT = """
[Welcome to Tech Talk! I'm Sarah, and today we're diving into the fascinating world of text-to-speech technology.](voice: af_sarah)
...s

[And I'm Michael! We've got an amazing episode lined up. The advances in neural TTS have been incredible lately.](voice: am_michael)
...s

[Absolutely! And we have a special guest with us today. Please welcome our AI researcher, Nicole!](voice: af_sarah)
...s

[Thanks for having me! I'm thrilled to be here. I've been working on voice synthesis for the past five years.](voice: af_nicole)
...s

[Nicole, can you tell us about the latest breakthroughs in making synthetic voices sound more natural?](voice: am_michael)
...s

[Of course! The key innovation has been in capturing prosody and emotional nuance. Modern models like Kokoro can generate speech that's nearly indistinguishable from human voices.](voice: af_nicole)
...s

[That's fascinating! What do you see as the main applications for this technology?](voice: af_sarah)
...s

[There are so many! Audiobook production, accessibility tools, language learning, and even preserving voices of people who might lose their ability to speak.](voice: af_nicole)
...s

[The accessibility angle is really compelling. Imagine being able to give a voice to those who can't speak.](voice: am_michael)
...s

[Exactly! And with open-source models, this technology is becoming available to everyone.](voice: af_sarah)
...s

[That's what excites me most. Democratizing access to high-quality speech synthesis opens up so many possibilities.](voice: af_nicole)
...s

[Well, this has been an enlightening discussion! Any final thoughts, Nicole?](voice: am_michael)
...s

[Just that we're at an inflection point. The next few years will bring even more amazing developments. Stay curious!](voice: af_nicole)
...s

[Thank you so much for joining us, Nicole! And thank you to our listeners for tuning in.](voice: af_sarah)
...s

[Until next time, keep exploring the future of technology!](voice: am_michael)
"""

SAMPLE_RATE = 24000  # Kokoro model sample rate


def random_pause(min_duration: float = 0.3, max_duration: float = 1.0) -> np.ndarray:
    """Generate random silence between speech segments."""
    silence_duration = random.uniform(min_duration, max_duration)
    return np.zeros(int(silence_duration * SAMPLE_RATE), dtype=np.float32)


def generate_traditional_podcast(kokoro: pykokoro.Kokoro) -> tuple[np.ndarray, int]:
    """Generate podcast using traditional separate voice segments.

    Args:
        kokoro: Initialized Kokoro TTS instance

    Returns:
        Tuple of (audio_array, sample_rate)
    """
    print("\n" + "=" * 70)
    print("TRADITIONAL APPROACH: Separate voice segments")
    print("=" * 70)

    audio_parts = []
    sample_rate = SAMPLE_RATE

    print(f"\nGenerating podcast with {len(PODCAST_SCRIPT)} segments...\n")

    for i, segment in enumerate(PODCAST_SCRIPT, 1):
        voice = segment["voice"]
        text = segment["text"]

        # Show progress
        speaker = voice.split("_")[1].title() if "_" in voice else voice
        print(f"[{i:2}/{len(PODCAST_SCRIPT)}] {speaker}: {text[:50]}...")

        # Generate audio for this segment
        samples, sample_rate = kokoro.create(
            text,
            voice=voice,
            speed=1.0,
            lang="en-us",
        )
        audio_parts.append(samples)

        # Add pause after each segment (longer pause for speaker changes)
        next_voice = PODCAST_SCRIPT[i]["voice"] if i < len(PODCAST_SCRIPT) else None
        if next_voice and next_voice != voice:
            # Longer pause when speaker changes
            audio_parts.append(random_pause(0.5, 1.2))
        else:
            # Shorter pause for same speaker continuing
            audio_parts.append(random_pause(0.2, 0.5))

    # Concatenate all audio
    print("\nConcatenating audio...")
    final_audio = np.concatenate(audio_parts)

    return final_audio, sample_rate


def generate_ssmd_podcast(kokoro: pykokoro.Kokoro) -> tuple[np.ndarray, int]:
    """Generate podcast using SSMD voice annotations.

    Note: This example demonstrates the SSMD voice annotation syntax.
    Full voice switching support within a single text may require additional
    implementation in the backend to handle voice changes per segment.

    For now, this serves as a demonstration of the markup syntax and will
    use the default voice for the entire text.

    Args:
        kokoro: Initialized Kokoro TTS instance

    Returns:
        Tuple of (audio_array, sample_rate)
    """
    print("\n" + "=" * 70)
    print("SSMD APPROACH: Inline voice annotations")
    print("=" * 70)
    print("\nNote: SSMD voice annotations demonstrate the syntax.")
    print("Full per-segment voice switching may require additional backend support.")
    print("\nGenerating podcast with SSMD voice markup...")
    print("\nSSMD Script (excerpt):")
    print("-" * 70)
    # Show first few lines
    lines = SSMD_PODCAST_SCRIPT.strip().split("\n")
    for line in lines[:6]:
        if line.strip():
            print(line)
    print("...")
    print("-" * 70)

    # Generate audio with SSMD markup
    # Note: Voice switching per segment requires backend support
    # For now, this demonstrates the markup and uses a default voice
    samples, sample_rate = kokoro.create(
        SSMD_PODCAST_SCRIPT,
        voice="af_sarah",  # Default voice
        speed=1.0,
        lang="en-us",
    )

    return samples, sample_rate


def main():
    """Generate the podcast audio using both approaches."""
    print("=" * 70)
    print("PODCAST DEMO: Traditional vs SSMD Voice Markup")
    print("=" * 70)

    print("\nInitializing TTS engine...")
    kokoro = pykokoro.Kokoro()

    # Demo 1: Traditional approach
    final_audio, sample_rate = generate_traditional_podcast(kokoro)

    output_file = "podcast_demo.wav"
    sf.write(output_file, final_audio, sample_rate)

    duration = len(final_audio) / sample_rate
    print(f"\nCreated {output_file}")
    print(f"Duration: {duration:.1f} seconds ({duration / 60:.1f} minutes)")

    # Demo 2: SSMD approach
    ssmd_audio, sample_rate = generate_ssmd_podcast(kokoro)

    output_file_ssmd = "podcast_ssmd_demo.wav"
    sf.write(output_file_ssmd, ssmd_audio, sample_rate)

    duration_ssmd = len(ssmd_audio) / sample_rate
    print(f"\nCreated {output_file_ssmd}")
    print(f"Duration: {duration_ssmd:.1f} seconds ({duration_ssmd / 60:.1f} minutes)")

    # Cleanup
    kokoro.close()

    print("\n" + "=" * 70)
    print("PODCAST DEMO COMPLETE")
    print("=" * 70)
    print("\nTwo approaches demonstrated:")
    print("  1. Traditional: Separate API calls per voice")
    print("  2. SSMD: Inline voice annotations [text](voice: name)")
    print("\nSSMD voice annotation syntax:")
    print("  - Simple voice: [Hello](voice: af_sarah)")
    print("  - With language: [Bonjour](voice: fr-FR, gender: female)")
    print("  - All attributes: [Text](voice: en-GB, gender: male, variant: 1)")


if __name__ == "__main__":
    main()
