#!/usr/bin/env python3
"""
Demonstrate split_and_phonemize_text with different split modes.

This example shows how to use the split_and_phonemize_text function
with paragraph, sentence, and clause modes to intelligently split text
for TTS processing.
"""

from pykokoro import Tokenizer
from pykokoro.phonemes import split_and_phonemize_text


def print_separator(title: str) -> None:
    """Print a visual separator with title."""
    print("\n" + "=" * 80)
    print(f" {title}")
    print("=" * 80)


def print_segments(segments: list, mode: str) -> None:
    """Print segment information in a readable format."""
    print_separator(f"Split Mode: {mode.upper()}")
    print(f"Total segments: {len(segments)}\n")

    for i, seg in enumerate(segments, 1):
        print(f"Segment {i}:")
        print(f"  Paragraph: {seg.paragraph}")
        print(f"  Sentence:  {seg.sentence}")
        print(f"  Text:      {seg.text!r}")
        print(f"  Phonemes:  {seg.phonemes}")
        print(f"  Tokens:    {len(seg.tokens)} tokens")
        print(f"  Language:  {seg.lang}")
        print(f"  Pause:     {seg.pause_after}s")
        print()


def main():
    """Run the split_and_phonemize_text demonstration."""

    # Sample text with 3 paragraphs, each containing 3 short sentences
    text = """The sun rises in the east. Birds begin to sing. The day starts fresh.

Coffee brews in the kitchen. Toast pops from the toaster. Breakfast is almost ready.

People walk to work. Cars fill the streets. The city comes alive."""

    print("=" * 80)
    print(" SPLIT_AND_PHONEMIZE_TEXT DEMONSTRATION")
    print("=" * 80)
    print("\nOriginal Text:")
    print("-" * 80)
    print(text)
    print("-" * 80)

    # Initialize tokenizer
    print("\nInitializing tokenizer...")
    tokenizer = Tokenizer()

    # Test all three split modes
    split_modes = ["paragraph", "sentence", "clause"]

    for mode in split_modes:
        print(f"\n{'▶' * 40}")
        print(f"Processing with split_mode='{mode}'...")
        print(f"{'▶' * 40}")

        try:
            segments = split_and_phonemize_text(
                text=text,
                tokenizer=tokenizer,
                lang="en-us",
                split_mode=mode,
                language_model="en_core_web_sm",
            )

            print_segments(segments, mode)

            # Show summary statistics
            total_chars = sum(len(seg.text) for seg in segments)
            total_phonemes = sum(len(seg.phonemes) for seg in segments)
            total_tokens = sum(len(seg.tokens) for seg in segments)

            print(f"Summary Statistics for {mode.upper()} mode:")
            print(f"  Total characters: {total_chars}")
            print(f"  Total phonemes:   {total_phonemes}")
            print(f"  Total tokens:     {total_tokens}")
            print(f"  Avg chars/seg:    {total_chars / len(segments):.1f}")
            print(f"  Avg phonemes/seg: {total_phonemes / len(segments):.1f}")

        except ImportError as e:
            print(f"\n⚠️  Error: {e}")
            print(f"   Mode '{mode}' requires spaCy.")
            print("   Install with: pip install spacy")
            print("   Then: python -m spacy download en_core_web_sm")
        except Exception as e:
            print(f"\n❌ Error processing mode '{mode}': {e}")

    # Comparison table
    print_separator("MODE COMPARISON SUMMARY")
    print(f"{'Mode':<15} {'Segments':<10} {'Behavior':<50}")
    print("-" * 80)
    newline_repr = r"\n\n"
    behavior_text = f"Split on double newlines ({newline_repr})"
    print(f"{'paragraph':<15} {'3':<10} {behavior_text:<50}")
    print(
        f"{'sentence':<15} {'9':<10} {'Split on sentence boundaries (using spaCy)':<50}"
    )
    print(
        f"{'clause':<15} {'9+':<10} {'Split on sentences + commas (using spaCy)':<50}"
    )
    print()

    print("Key Observations:")
    print("  • 'paragraph' mode: Keeps each paragraph as one segment")
    print("  • 'sentence' mode: Creates one segment per sentence")
    print("  • 'clause' mode: Can split further at commas for finer control")
    print("  • Paragraph and sentence indices track the text structure")
    print("  • All segments stay within max_phoneme_length (510 by default)")
    print()


if __name__ == "__main__":
    main()
