#!/usr/bin/env python3
"""
Multi-Source Model Comparison Example for pykokoro.

This example demonstrates how to generate audio using different model sources:
1. HuggingFace (default) - Multi-language voices
   from onnx-community/Kokoro-82M-v1.0-ONNX
2. GitHub v1.0 - Multi-language voices from thewh1teagle/kokoro-onnx
3. GitHub v1.1-zh - Chinese-specific voices from thewh1teagle/kokoro-onnx

Usage:
    python examples/model_source_comparison.py

Output:
    huggingface_model_demo.wav - Audio from HuggingFace model
    github_v1.0_model_demo.wav - Audio from GitHub v1.0 model
    github_v1.1_zh_model_demo.wav - Audio from GitHub v1.1 Chinese model

The example shows how the same text is synthesized using different model sources,
allowing you to compare quality, voices, and performance.

Note:
    - HuggingFace source downloads individual voice files and combines them
    - GitHub sources download pre-combined voices.bin files
    - The v1.1-zh Chinese model has Chinese-specific voices
    - Voice names are loaded dynamically from the model's voices.bin file
"""

import soundfile as sf

import pykokoro

# Text for English models (HuggingFace and GitHub v1.0)
ENGLISH_TEXT = (
    "Hello! This is a demonstration of the PyKokoro text-to-speech library. "
    "We are comparing different model sources to show their capabilities. "
    "Technology enables us to communicate across boundaries."
)

# Text for Chinese model (GitHub v1.1-zh)
CHINESE_TEXT = (
    "你好！这是PyKokoro文本转语音库的演示。"
    "我们正在比较不同的模型来源，以展示它们的能力。"
    "技术使我们能够跨越界限进行交流。"
)


def main():
    """Generate audio using all three model sources."""

    # =========================================================================
    # Example 1: HuggingFace Model (Default)
    # =========================================================================
    print("=" * 70)
    print("Example 1: HuggingFace Model Source")
    print("=" * 70)
    print("Source: onnx-community/Kokoro-82M-v1.0-ONNX")
    print(f"Text: {ENGLISH_TEXT[:50]}...")

    print("\nInitializing TTS engine with HuggingFace model...")
    kokoro_hf = pykokoro.Kokoro(
        model_source="huggingface",  # Default source
        model_quality="fp32",  # Use fp32 quality
    )

    # Get and display available voices
    available_voices = kokoro_hf.get_voices()
    print(f"Available voices: {len(available_voices)}")
    print(f"Sample voices: {', '.join(available_voices[:5])}...")

    # Use the first available voice or default to af_sarah
    voice_to_use = "af_sarah" if "af_sarah" in available_voices else available_voices[0]
    print(f"\nGenerating audio with HuggingFace model using voice '{voice_to_use}'...")
    samples_hf, sample_rate_hf = kokoro_hf.create(
        ENGLISH_TEXT,
        voice=voice_to_use,
        speed=1.0,
        lang="en-us",
    )

    output_file_hf = "huggingface_model_demo.wav"
    sf.write(output_file_hf, samples_hf, sample_rate_hf)

    duration_hf = len(samples_hf) / sample_rate_hf
    print(f"✓ Created {output_file_hf}")
    print(f"  Duration: {duration_hf:.2f} seconds")
    print(f"  Sample rate: {sample_rate_hf} Hz")
    print(f"  Samples: {len(samples_hf):,}")

    kokoro_hf.close()

    # =========================================================================
    # Example 2: GitHub v1.0 Model
    # =========================================================================
    print("\n" + "=" * 70)
    print("Example 2: GitHub v1.0 Model Source")
    print("=" * 70)
    print("Source: github.com/thewh1teagle/kokoro-onnx (v1.0)")
    print(f"Text: {ENGLISH_TEXT[:50]}...")

    print("\nInitializing TTS engine with GitHub v1.0 model...")
    kokoro_github = pykokoro.Kokoro(
        model_source="github",  # GitHub source
        model_variant="v1.0",  # English model
        model_quality="fp16",  # Use fp16 quality (available on GitHub)
    )

    # Get and display available voices
    available_voices_github = kokoro_github.get_voices()
    print(f"Available voices: {len(available_voices_github)}")
    print(f"Sample voices: {', '.join(available_voices_github[:5])}...")

    # Use the same voice as before if available
    voice_to_use_github = (
        voice_to_use
        if voice_to_use in available_voices_github
        else available_voices_github[0]
    )
    print(
        f"\nGenerating audio with GitHub v1.0 model "
        f"using voice '{voice_to_use_github}'..."
    )
    samples_github, sample_rate_github = kokoro_github.create(
        ENGLISH_TEXT,
        voice=voice_to_use_github,
        speed=1.0,
        lang="en-us",
    )

    output_file_github = "github_v1.0_model_demo.wav"
    sf.write(output_file_github, samples_github, sample_rate_github)

    duration_github = len(samples_github) / sample_rate_github
    print(f"✓ Created {output_file_github}")
    print(f"  Duration: {duration_github:.2f} seconds")
    print(f"  Sample rate: {sample_rate_github} Hz")
    print(f"  Samples: {len(samples_github):,}")

    kokoro_github.close()

    # =========================================================================
    # Example 3: GitHub v1.1-zh Model (English voice test)
    # =========================================================================
    print("\n" + "=" * 70)
    print("Example 3: GitHub v1.1-zh Model Source")
    print("=" * 70)
    print("Source: github.com/thewh1teagle/kokoro-onnx (v1.1-zh)")
    print("Note: Testing with English text and English voice from v1.1-zh model")
    print(f"Text: {ENGLISH_TEXT[:50]}...")

    print("\nInitializing TTS engine with GitHub v1.1-zh model...")
    kokoro_v11 = pykokoro.Kokoro(
        model_source="github", model_variant="v1.1-zh", model_quality="fp32"
    )

    # Get and display available voices
    available_voices_v11 = kokoro_v11.get_voices()
    print(f"Available voices in v1.1-zh: {len(available_voices_v11)}")

    # Find English voices (af_maple, af_sol, bf_vale)
    english_voices_v11 = [
        v for v in available_voices_v11 if v in ["af_maple", "af_sol", "bf_vale"]
    ]
    print(f"English voices: {', '.join(english_voices_v11)}")

    # Use af_maple (English female voice)
    voice_to_use_v11 = "af_maple"
    print(
        f"\nGenerating audio with GitHub v1.1-zh model "
        f"using voice '{voice_to_use_v11}'..."
    )
    samples_v11, sample_rate_v11 = kokoro_v11.create(
        ENGLISH_TEXT,
        voice=voice_to_use_v11,
        speed=1.0,
        lang="en-us",
    )

    output_file_v11 = "github_v1.1_zh_model_demo.wav"
    sf.write(output_file_v11, samples_v11, sample_rate_v11)

    duration_v11 = len(samples_v11) / sample_rate_v11
    print(f"✓ Created {output_file_v11}")
    print(f"  Duration: {duration_v11:.2f} seconds")
    print(f"  Sample rate: {sample_rate_v11} Hz")
    print(f"  Samples: {len(samples_v11):,}")

    kokoro_v11.close()

    # =========================================================================
    # Summary
    # =========================================================================
    print("\n" + "=" * 70)
    print("Summary")
    print("=" * 70)
    print(f"HuggingFace model:  {duration_hf:.2f}s - {output_file_hf}")
    print(f"GitHub v1.0 model:  {duration_github:.2f}s - {output_file_github}")
    print(f"GitHub v1.1-zh:     {duration_v11:.2f}s - {output_file_v11}")
    print("\nComparison complete! You can now listen to the generated audio files.")
    print("\nModel Source Characteristics:")
    print("  • HuggingFace: Multi-language, 54 voices, 8 quality options")
    print("  • GitHub v1.0: Multi-language, 54 voices, 4 quality options")
    print("  • GitHub v1.1-zh: 103 voices (English + Chinese), fp32 only")
    print("    Note: Chinese G2P will be fixed in future update")


if __name__ == "__main__":
    main()
