#!/usr/bin/env python3
"""
Demonstrate SSMD break markers combined with pause_mode="manual".

This example shows how to use SSMD break markers together with manual
pause control for better prosody in long texts.

Usage:
    python examples/pauses_with_splitting.py

Output:
    pauses_splitting_demo.wav - Long text with pauses and sentence splitting
"""

import soundfile as sf

from pykokoro import KokoroPipeline, PipelineConfig
from pykokoro.generation_config import GenerationConfig


def main():
    """Generate example with pauses and text splitting."""
    print("Initializing TTS engine...")
    generation = GenerationConfig(
        lang="en-us",
        pause_mode="manual",
        pause_clause=0.3,
        pause_sentence=0.6,
        pause_paragraph=1.2,
        pause_variance=0.05,
        random_seed=42,
    )
    pipe = KokoroPipeline(PipelineConfig(voice="af_sarah", generation=generation))

    # Long text with SSMD pauses and natural sentence breaks
    text = """
    Welcome to our podcast!

    The Future of AI.

    Today's episode covers three main topics. ...c First, we'll explore neural
    networks and how they learn from data to make increasingly accurate predictions.
    ...s Second, we'll dive into deep learning architectures that power modern AI
    systems, including transformers and convolutional neural networks. ...p And
    third, we'll examine real-world applications transforming industries worldwide,
    from healthcare to autonomous vehicles.

    Each of these topics represents a fascinating area of research and development.
    ...c Neural networks, inspired by biological neurons, process information through
    interconnected layers. ...s Deep learning takes this further by adding many
    layers, enabling the system to learn hierarchical representations.

    Let's dive into these fascinating subjects! ...p
    """

    print("=" * 70)
    print("Generating with manual pause control and SSMD pauses...")
    print("=" * 70)
    print("\nThis combines:")
    print("  • pause_mode='manual' - PyKokoro controls pauses precisely")
    print("  • Explicit pause control using SSMD breaks (...c, ...s, ...p)")
    print("  • Automatic handling of long sentences")
    print("  • Natural pause variance for more human-like speech")
    print()

    print("Processing text...")
    print(f"Text length: {len(text)} characters")
    print()

    res = pipe.run(text)
    samples, sample_rate = res.audio, res.sample_rate

    output_file = "pauses_splitting_demo.wav"
    sf.write(output_file, samples, sample_rate)
    duration = len(samples) / sample_rate

    print("✓ Generation complete!")
    print()
    print("=" * 70)
    print(f"Generated: {output_file}")
    print(f"Duration: {duration:.2f} seconds ({duration / 60:.1f} minutes)")
    print(f"Sample rate: {sample_rate} Hz")
    print(f"Total samples: {len(samples):,}")
    print("=" * 70)
    print()

    # Show SSMD pause markers used
    pause_counts = {
        "...c": text.count("...c"),
        "...s": text.count("...s"),
        "...p": text.count("...p"),
    }

    total_pause_time = (
        pause_counts["...c"] * 0.3
        + pause_counts["...s"] * 0.6
        + pause_counts["...p"] * 1.2
    )

    print("SSMD pause statistics:")
    clause_total = pause_counts["...c"] * 0.3
    sentence_total = pause_counts["...s"] * 0.6
    paragraph_total = pause_counts["...p"] * 1.2
    print(
        f"  Clause pauses (...c):     "
        f"{pause_counts['...c']} × 0.3s = {clause_total:.1f}s"
    )
    print(
        f"  Sentence pauses (...s):   "
        f"{pause_counts['...s']} × 0.6s = {sentence_total:.1f}s"
    )
    print(
        f"  Paragraph pauses (...p):  "
        f"{pause_counts['...p']} × 1.2s = {paragraph_total:.1f}s"
    )
    print(f"  Total pause time:         ~{total_pause_time:.1f}s")
    print(f"  Estimated speech time:    ~{duration - total_pause_time:.1f}s")
    print()

    print("Pause modes:")
    print("  • pause_mode='tts' (default) - TTS generates pauses naturally")
    print("  • pause_mode='manual' - PyKokoro controls pauses with precision")
    print()
    print("SSMD break markers:")
    print("  • ...n - No pause (0ms)")
    print("  • ...w - Weak pause (150ms)")
    print("  • ...c - Clause/comma pause (300ms)")
    print("  • ...s - Sentence pause (600ms)")
    print("  • ...p - Paragraph pause (1000ms)")
    print("  • ...500ms - Custom pause (500 milliseconds)")
    print("  • ...2s - Custom pause (2 seconds)")
    print()
    print("Pause variance options:")
    print("  • pause_variance=0.0 - No variance (exact pauses)")
    print("  • pause_variance=0.05 - Default (±100ms at 95% confidence)")
    print("  • pause_variance=0.1 - More variation (±200ms at 95% confidence)")
    print("  • random_seed=42 - Reproducible results")
    print("  • random_seed=None - Different pauses each time")
    print()


if __name__ == "__main__":
    main()
