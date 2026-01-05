#!/usr/bin/env python3
"""
Voice mixing/blending demonstration using pykokoro.

This example shows how to blend multiple voices together to create unique
hybrid voices. Voice blending combines the characteristics of two or more
voices using weighted averaging.

Features demonstrated:
- Simple 50/50 voice blends
- Custom weight distributions (e.g., 70/30, 33/33/34)
- Blending voices with different genders
- Blending voices with different accents
- Using CLI-style blend strings

Usage:
    python examples/voices.py

Output:
    voices_demo.wav - Audio showcasing various voice blends
"""

import numpy as np
import soundfile as sf

import pykokoro

# Sample rate constant
SAMPLE_RATE = 24000


def create_silence(duration_seconds: float = 0.5) -> np.ndarray:
    """Create a silent pause."""
    return np.zeros(int(duration_seconds * SAMPLE_RATE), dtype=np.float32)


def main():
    """Generate voice blending demonstrations."""
    print("Initializing TTS engine...")
    kokoro = pykokoro.Kokoro()

    # Test sentence
    test_text = (
        "This is a demonstration of voice blending technology. "
        "You can combine multiple voices to create unique hybrid voices."
    )

    audio_parts = []

    # Demo configurations: (description, voice_blend_string)
    demos = [
        # 1. Introduction with a single voice
        (
            "Introduction - Pure voice (af_sarah)",
            None,  # Will use direct voice parameter
            "af_sarah",
        ),
        # 2. 50/50 blends - Same gender
        (
            "Fifty-fifty blend of Sarah and Nicole",
            "af_sarah:50,af_nicole:50",
            None,
        ),
        (
            "Fifty-fifty blend of Michael and Adam",
            "am_michael:50,am_adam:50",
            None,
        ),
        # 3. 50/50 blends - Cross-gender
        (
            "Fifty-fifty blend of Sarah and Michael",
            "af_sarah:50,am_michael:50",
            None,
        ),
        # 4. 70/30 blend - Dominant voice
        (
            "Seventy-thirty blend - more Sarah, less Nicole",
            "af_sarah:70,af_nicole:30",
            None,
        ),
        # 5. Three-way blend
        (
            "Three-way blend - Sarah, Nicole, and Bella",
            "af_sarah:33,af_nicole:33,af_bella:34",
            None,
        ),
        # 6. Accent mixing - American & British
        (
            "Accent blend - American Sarah and British Emma",
            "af_sarah:50,bf_emma:50",
            None,
        ),
        # 7. Subtle blend (90/10)
        (
            "Subtle blend - ninety percent Sarah, ten percent Nicole",
            "af_sarah:90,af_nicole:10",
            None,
        ),
    ]

    print(f"\nGenerating {len(demos)} voice blend demonstrations...\n")

    for i, (description, blend_str, single_voice) in enumerate(demos, 1):
        print(f"[{i}/{len(demos)}] {description}")

        # Add announcement of what's being demonstrated
        announcement_text = f"Demonstration {i}. {description}."

        # Generate announcement with a neutral voice
        announcement_audio, sample_rate = kokoro.create(
            announcement_text,
            voice="af_sarah",
            speed=1.0,
            lang="en-us",
        )
        audio_parts.append(announcement_audio)
        audio_parts.append(create_silence(0.8))

        # Generate the test text with the blended/single voice
        if single_voice:
            # Use a single voice directly
            samples, sample_rate = kokoro.create(
                test_text,
                voice=single_voice,
                speed=1.0,
                lang="en-us",
            )
        else:
            # Parse and use voice blend
            blend = pykokoro.VoiceBlend.parse(blend_str)
            samples, sample_rate = kokoro.create(
                test_text,
                voice=blend,
                speed=1.0,
                lang="en-us",
            )

        audio_parts.append(samples)
        audio_parts.append(create_silence(1.5))  # Longer pause between demos

    # Add conclusion
    print(f"[{len(demos) + 1}/{len(demos) + 1}] Conclusion")
    conclusion_text = (
        "This concludes the voice blending demonstration. "
        "You can use the --voice parameter with blend strings "
        "in the command line interface, "
        "or create VoiceBlend objects programmatically."
    )
    conclusion_audio, sample_rate = kokoro.create(
        conclusion_text,
        voice="af_sarah",
        speed=1.0,
        lang="en-us",
    )
    audio_parts.append(conclusion_audio)

    # Concatenate all audio
    print("\nConcatenating audio segments...")
    final_audio = np.concatenate(audio_parts)

    # Save to file
    output_file = "voices_demo.wav"
    sf.write(output_file, final_audio, sample_rate)

    duration = len(final_audio) / sample_rate
    print(f"\nCreated {output_file}")
    print(f"Duration: {duration:.1f} seconds ({duration / 60:.1f} minutes)")
    print("\nVoice blend format: 'voice1:weight1,voice2:weight2'")
    print("Example CLI usage:")
    print("  pykokoro sample 'Hello world' --voice 'af_sarah:50,am_michael:50'")

    # Cleanup
    kokoro.close()


if __name__ == "__main__":
    main()
