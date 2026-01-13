#!/usr/bin/env python3
"""
Short Sentence Handling Demonstration.

This example demonstrates how pykokoro automatically handles short sentences
using the repeat-and-cut technique. Short phoneme sequences (like "Why?" = 3
phonemes) can produce poor audio quality when processed individually. The
repeat-and-cut technique provides more context for natural prosody.

The technique works as follows:
1. Short sentences are detected based on phoneme length
2. The sentence is repeated multiple times (e.g., "Why?" -> "Why? Why? Why?")
3. TTS generates audio for the repeated text (better quality with more context)
4. The audio is cut at the measured duration of a single instance

This happens automatically during audio generation - no manual configuration
is needed. However, you can customize the behavior using ShortSentenceConfig.

Usage:
    python examples/optimal_phoneme_length_demo.py

Output:
    short_sentence_demo.wav - Audio demonstrating short sentence handling
    Detailed console output showing how short sentences are processed
"""

import soundfile as sf

import pykokoro

# Dialogue with mix of very short and normal sentences
DIALOGUE_TEXT = """
"Why?" she asked.

"Do it!" he commanded.

"Go!" they shouted.

"I know." she whispered.

He sits quietly.

She nods slowly.

"Really?" he questioned.

"Yes." she confirmed.

"The quick brown fox jumps over the lazy dog." he said with a smile.

"That's wonderful news!" she exclaimed happily.
"""

VOICE = "af_bella"  # American Female voice
LANG = "en-us"


def print_separator(title: str) -> None:
    """Print a visual separator with title."""
    print("\n" + "=" * 70)
    print(f" {title}")
    print("=" * 70)


def analyze_segments(segments: list, title: str) -> dict:
    """Analyze and print segment statistics."""
    print_separator(title)

    phoneme_counts = []
    total_phonemes = 0
    short_count = 0

    # Default threshold for "short"
    min_threshold = 30

    for i, seg in enumerate(segments, 1):
        phoneme_length = len(seg.phonemes)
        phoneme_counts.append(phoneme_length)
        total_phonemes += phoneme_length

        is_short = phoneme_length < min_threshold
        if is_short:
            short_count += 1

        # Show first 5 and last 1 segments
        if i <= 5 or i == len(segments):
            short_marker = " [SHORT - will use repeat-and-cut]" if is_short else ""
            print(f"\nSegment {i}:{short_marker}")
            print(f"  Text: '{seg.text[:60]}{'...' if len(seg.text) > 60 else ''}'")
            print(f"  Phonemes: {phoneme_length} chars")
            if i == 5 and len(segments) > 6:
                print(f"\n  ... ({len(segments) - 6} more segments) ...")

    avg_phonemes = total_phonemes / len(segments) if segments else 0
    min_phonemes = min(phoneme_counts) if phoneme_counts else 0
    max_phonemes = max(phoneme_counts) if phoneme_counts else 0

    print("\nStatistics:")
    print(f"  Total segments: {len(segments)}")
    print(f"  Short segments (<{min_threshold} phonemes): {short_count}")
    print(f"  Average phonemes per segment: {avg_phonemes:.1f}")
    print(f"  Min/Max phonemes: {min_phonemes}/{max_phonemes}")

    return {
        "segments": len(segments),
        "short_count": short_count,
        "total_phonemes": total_phonemes,
        "avg_phonemes": avg_phonemes,
        "min_phonemes": min_phonemes,
        "max_phonemes": max_phonemes,
        "phoneme_counts": phoneme_counts,
    }


def main():
    """Generate demo showcasing automatic short sentence handling."""
    print_separator("SHORT SENTENCE HANDLING DEMONSTRATION")
    print("\nThis demo shows how pykokoro automatically handles short sentences")
    print("using the repeat-and-cut technique for improved audio quality.")
    print("\nHow it works:")
    print("  1. Short segments are detected based on phoneme length")
    print("  2. Text is repeated: 'Why?' -> 'Why? Why? Why?'")
    print("  3. TTS generates audio with more context (better prosody)")
    print("  4. Audio is trimmed to extract only the first instance")
    print("\nThis happens automatically - no configuration needed!")

    # Create Kokoro with default settings (automatic short sentence handling)
    print_separator("Creating Kokoro Instance")
    kokoro = pykokoro.Kokoro()
    print("Kokoro created with default ShortSentenceConfig:")
    print("  min_phoneme_length: 30 (segments below this use repeat-and-cut)")
    print("  target_phoneme_length: 100 (target length for repeated text)")
    print("  max_repetitions: 5 (maximum times to repeat)")

    # Analyze the text to show which segments are short
    print_separator("Analyzing Text Segments")
    segments = pykokoro.phonemes.text_to_phoneme_segments(
        text=DIALOGUE_TEXT,
        tokenizer=kokoro.tokenizer,
        lang=LANG,
        pause_mode="tts",
    )
    stats = analyze_segments(segments, "Segment Analysis (pause_mode='tts')")

    # Generate audio with automatic short sentence handling
    print_separator("Generating Audio")
    print("\nShort segments will automatically use repeat-and-cut...")
    samples, sample_rate = kokoro.create(
        DIALOGUE_TEXT,
        voice=VOICE,
        lang=LANG,
        speed=1.0,
    )

    duration = len(samples) / sample_rate
    print("\nAudio generated successfully!")
    print(f"  Duration: {duration:.2f} seconds")
    print(f"  Sample rate: {sample_rate} Hz")

    # Save the audio
    output_file = "short_sentence_demo.wav"
    sf.write(output_file, samples, sample_rate)
    print(f"\nSaved to: {output_file}")

    # Show how to customize the config
    print_separator("CUSTOMIZING SHORT SENTENCE HANDLING")
    print("\nYou can customize the behavior with ShortSentenceConfig:")
    print("""
    from pykokoro.short_sentence_handler import ShortSentenceConfig

    # More aggressive short sentence handling
    config = ShortSentenceConfig(
        min_phoneme_length=50,    # Treat segments <50 as short
        target_phoneme_length=150, # Repeat until ~150 phonemes
        max_repetitions=7,         # Allow up to 7 repetitions
    )

    kokoro = pykokoro.Kokoro(short_sentence_config=config)
    """)

    print("\nTo disable short sentence handling entirely:")
    print("""
    config = ShortSentenceConfig(
        min_phoneme_length=0,  # No segment is considered "short"
    )
    kokoro = pykokoro.Kokoro(short_sentence_config=config)
    """)

    # Summary
    print_separator("SUMMARY")
    print(f"\nProcessed {stats['segments']} segments total")
    print(f"  - {stats['short_count']} short segments (used repeat-and-cut)")
    print(f"  - {stats['segments'] - stats['short_count']} normal segments")
    print("\nBenefits of repeat-and-cut:")
    print("  - Better prosody for short utterances")
    print("  - More natural intonation")
    print("  - Consistent audio quality across all segment lengths")
    print("\nListen to the WAV file to hear the natural-sounding short sentences!")

    kokoro.close()


if __name__ == "__main__":
    main()
