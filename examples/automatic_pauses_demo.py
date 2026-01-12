#!/usr/bin/env python3
"""
Demonstrate automatic pause insertion with pause_mode="manual".

This example shows how pause_mode="manual" automatically adds natural pauses
between clauses, sentences, and paragraphs for more natural-sounding speech
without manual pause markers.

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
    print("Generating with AUTOMATIC pauses (pause_mode='manual')")
    print("=" * 70)
    print("\nKey features:")
    print("  • pause_mode='manual' - PyKokoro controls all pauses precisely")
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
        pause_mode="manual",  # PyKokoro controls pauses precisely
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
    print("1. TTS-controlled pauses (default):")
    print("   kokoro.create(text, voice='af_sarah')")
    print("   → TTS generates natural pauses automatically")
    print()
    print("2. SSMD break markers:")
    print("   text = 'Hello ...c world ...s How are you?'")
    print("   kokoro.create(text, voice='af_sarah')")
    print("   → SSMD breaks automatically detected and processed")
    print()
    print("3. Manual pause control (this example):")
    print("   kokoro.create(text, voice='af_sarah', pause_mode='manual')")
    print("   → PyKokoro controls pauses precisely at linguistic boundaries")
    print()

    print("Tips for best results:")
    print("  • Use pause_mode='manual' for precise control over all pauses")
    print("  • Use pause_mode='tts' (default) to let TTS handle pauses naturally")
    print("  • Adjust pause_clause/sentence/paragraph to match your content style")
    print("  • Set pause_variance=0.0 for consistent timing (e.g., training data)")
    print("  • Set random_seed for reproducible output")
    print()


if __name__ == "__main__":
    main()
