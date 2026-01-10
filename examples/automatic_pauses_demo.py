#!/usr/bin/env python3
"""
Demonstrate automatic pause insertion with trim_silence and split_mode.

This example shows how trim_silence combined with split_mode automatically
adds natural pauses between clauses, sentences, and paragraphs for more
natural-sounding speech without manual pause markers.

Usage:
    python examples/automatic_pauses_demo.py

Output:
    automatic_pauses_demo.wav - Text with automatic natural pauses
"""

import soundfile as sf

import pykokoro


def main():
    """Generate example with automatic pauses."""
    print("Initializing TTS engine...")
    kokoro = pykokoro.Kokoro()

    # Text with multiple paragraphs, sentences, and clauses
    # No manual pause markers needed - pauses are added automatically!
    text = """
    The future of artificial intelligence is rapidly evolving. Machine learning
    models are becoming more sophisticated, efficient, and accessible to developers
    worldwide. This democratization of AI technology promises to revolutionize
    industries from healthcare to transportation.

    Neural networks, the foundation of modern AI, consist of interconnected layers
    that process information hierarchically. Each layer extracts increasingly
    complex features from the input data, enabling the network to learn patterns
    and make predictions. Deep learning, a subset of machine learning, uses many
    layers to achieve remarkable results in computer vision, natural language
    processing, and speech recognition.

    As we look to the future, the integration of AI into everyday life will
    continue to accelerate. From smart homes to autonomous vehicles, AI-powered
    systems are transforming how we live, work, and interact with technology.
    """

    print("=" * 70)
    print("Generating with AUTOMATIC pauses (trim_silence + split_mode)")
    print("=" * 70)
    print("\nKey features:")
    print("  • trim_silence=True - Removes silence from segment boundaries")
    print("  • split_mode='clause' - Splits on commas and sentences")
    print("  • Automatic pause insertion:")
    print("    - Short pauses after clauses (within sentence)")
    print("    - Medium pauses after sentences (within paragraph)")
    print("    - Long pauses after paragraphs")
    print("  • Gaussian variance for natural rhythm")
    print("  • NO manual pause markers needed!")
    print()

    print("Processing text...")
    print(f"Text length: {len(text)} characters")
    print()

    # Generate with automatic pauses
    samples, sample_rate = kokoro.create(
        text,
        voice="af_sarah",
        lang="en-us",
        split_mode="clause",  # Split on commas and sentences
        trim_silence=True,  # Enable automatic pause insertion
        pause_clause=0.25,  # Clause pauses (commas)
        pause_sentence=0.5,  # Sentence pauses
        pause_paragraph=1.0,  # Paragraph pauses
        pause_variance=0.05,  # Natural variance (±100ms at 95%)
        random_seed=None,  # Different pauses each time for natural variation
    )

    output_file = "automatic_pauses_demo.wav"
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

    kokoro.close()

    print("Comparison with other approaches:")
    print()
    print("1. NO pauses (default):")
    print("   kokoro.create(text, voice='af_sarah')")
    print("   → Fast, continuous speech without breaks")
    print()
    print("2. SSMD break markers (automatically detected):")
    print("   text = 'Hello ...c world ...s How are you?'")
    print("   kokoro.create(text, voice='af_sarah')")
    print("   → SSMD breaks automatically detected and processed")
    print()
    print("3. Automatic pauses (this example):")
    print("   kokoro.create(text, voice='af_sarah',")
    print("                 split_mode='clause', trim_silence=True)")
    print("   → Natural pauses automatically added at linguistic boundaries")
    print()
    print("4. Combined approach:")
    print("   Use SSMD breaks AND split_mode + trim_silence together")
    print("   → SSMD markers for special emphasis + automatic natural pauses")
    print()

    print("Tips for best results:")
    print("  • Use split_mode='clause' for the most natural pauses")
    print("  • Use split_mode='sentence' for fewer, longer pauses")
    print("  • Use split_mode='paragraph' for minimal pauses")
    print("  • Adjust pause_clause/sentence/paragraph to match your content style")
    print("  • Set pause_variance=0.0 for consistent timing (e.g., training data)")
    print("  • Set random_seed for reproducible output")
    print()


if __name__ == "__main__":
    main()
