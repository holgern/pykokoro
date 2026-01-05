#!/usr/bin/env python3
"""
Podcast-style multi-voice conversation example using pykokoro.

This example demonstrates how to create a podcast with multiple speakers
using different voices. Each speaker has their own voice and the conversation
flows naturally with random pauses between lines.

Usage:
    python examples/podcast.py

Output:
    podcast_demo.wav - A multi-voice podcast conversation
"""

import random

import numpy as np
import soundfile as sf

import pykokoro

# Podcast script with different speakers
# Available voices: af_* (American Female), am_* (American Male),
#                   bf_* (British Female), bm_* (British Male), etc.
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


def random_pause(min_duration: float = 0.3, max_duration: float = 1.0) -> np.ndarray:
    """Generate random silence between speech segments."""
    silence_duration = random.uniform(min_duration, max_duration)
    return np.zeros(int(silence_duration * sample_rate), dtype=np.float32)


def main():
    """Generate the podcast audio."""
    print("Initializing TTS engine...")
    kokoro = pykokoro.Kokoro()

    audio_parts = []

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

    # Save to file
    output_file = "podcast_demo.wav"
    sf.write(output_file, final_audio, sample_rate)

    duration = len(final_audio) / sample_rate
    print(f"\nCreated {output_file}")
    print(f"Duration: {duration:.1f} seconds ({duration / 60:.1f} minutes)")

    # Cleanup
    kokoro.close()


if __name__ == "__main__":
    main()
