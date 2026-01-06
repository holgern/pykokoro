#!/usr/bin/env python3
"""
Demonstrate inter-word pause control using pykokoro.

This example shows how to use pause markers (.), (..), (...)
to control timing in speech synthesis.

Usage:
    python examples/pauses_demo.py

Output:
    example1_basic_pauses.wav - Basic pause demonstration
    example2_custom_durations.wav - Custom pause durations
    example3_leading_pause.wav - Leading pause example
"""

import soundfile as sf

import pykokoro


def main():
    """Generate example audio files with pauses."""
    print("Initializing TTS engine...")
    kokoro = pykokoro.Kokoro()

    # Example 1: Basic pause markers
    print("\n" + "=" * 60)
    print("Example 1: Basic Pauses")
    print("=" * 60)

    text1 = "Chapter 5 (...) I'm Klaus. (.) Welcome to the show!"

    print(f"Text: {text1}")
    print("Pause markers: (.) = 0.3s, (..) = 0.6s, (...) = 1.0s")

    samples, sample_rate = kokoro.create(
        text1,
        voice="am_michael",
        lang="en-us",
        enable_pauses=True,
    )

    output1 = "example1_basic_pauses.wav"
    sf.write(output1, samples, sample_rate)
    duration1 = len(samples) / sample_rate
    print(f"✓ Generated: {output1}")
    print(f"  Duration: {duration1:.2f}s")

    # Example 2: Custom pause durations
    print("\n" + "=" * 60)
    print("Example 2: Custom Pause Durations")
    print("=" * 60)

    text2 = "Quick pause (.) Medium pause (..) Long pause (...) Done!"

    print(f"Text: {text2}")
    print("Custom durations: (.) = 0.2s, (..) = 0.5s, (...) = 1.5s")

    samples, sample_rate = kokoro.create(
        text2,
        voice="af_sarah",
        enable_pauses=True,
        pause_short=0.2,
        pause_medium=0.5,
        pause_long=1.5,
    )

    output2 = "example2_custom_durations.wav"
    sf.write(output2, samples, sample_rate)
    duration2 = len(samples) / sample_rate
    print(f"✓ Generated: {output2}")
    print(f"  Duration: {duration2:.2f}s")

    # Example 3: Leading pause
    print("\n" + "=" * 60)
    print("Example 3: Leading Pause")
    print("=" * 60)

    text3 = "(...) After a long pause, we begin speaking."

    print(f"Text: {text3}")
    print("Note: Pause marker at start creates silence before speech")

    samples, sample_rate = kokoro.create(
        text3,
        voice="am_adam",
        enable_pauses=True,
    )

    output3 = "example3_leading_pause.wav"
    sf.write(output3, samples, sample_rate)
    duration3 = len(samples) / sample_rate
    print(f"✓ Generated: {output3}")
    print(f"  Duration: {duration3:.2f}s (includes 1.0s initial silence)")

    # Example 4: Consecutive pauses
    print("\n" + "=" * 60)
    print("Example 4: Consecutive Pauses (Additive)")
    print("=" * 60)

    text4 = "First sentence. (...) (..) Second sentence after a very long pause."

    print(f"Text: {text4}")
    print("Note: Consecutive pauses add together (1.0s + 0.6s = 1.6s)")

    samples, sample_rate = kokoro.create(
        text4,
        voice="af_bella",
        enable_pauses=True,
    )

    output4 = "example4_consecutive_pauses.wav"
    sf.write(output4, samples, sample_rate)
    duration4 = len(samples) / sample_rate
    print(f"✓ Generated: {output4}")
    print(f"  Duration: {duration4:.2f}s")

    # Example 5: Disabling pause processing
    print("\n" + "=" * 60)
    print("Example 5: Pauses Disabled (markers treated as text)")
    print("=" * 60)

    text5 = "Text with (.) pause markers that are not processed."

    print(f"Text: {text5}")
    print("Note: enable_pauses=False treats markers as regular text")

    samples, sample_rate = kokoro.create(
        text5,
        voice="am_michael",
        enable_pauses=False,  # Markers treated as text
    )

    output5 = "example5_pauses_disabled.wav"
    sf.write(output5, samples, sample_rate)
    duration5 = len(samples) / sample_rate
    print(f"✓ Generated: {output5}")
    print(f"  Duration: {duration5:.2f}s")

    kokoro.close()

    print("\n" + "=" * 60)
    print("All examples generated successfully!")
    print("=" * 60)
    print("\nTotal files created: 5")
    total_duration = duration1 + duration2 + duration3 + duration4 + duration5
    print(f"Total duration: {total_duration:.2f}s")


if __name__ == "__main__":
    main()
