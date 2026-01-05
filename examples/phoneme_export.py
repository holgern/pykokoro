#!/usr/bin/env python3
"""
Phoneme export example using pykokoro.

This example demonstrates how to create a phoneme JSON file from text,
which can later be converted to audio. This is useful for:
- Pre-processing text for faster audio generation
- Inspecting and editing phonemes before synthesis
- Batch processing workflows

Uses the same punctuation-heavy text as punctuation.py to show
how punctuation affects phoneme generation.

Usage:
    python examples/phoneme_export.py

Output:
    punctuation_phonemes.json - Phoneme data in JSON format

To convert to audio later:
    pykokoro phonemes convert punctuation_phonemes.json -o punctuation_from_phonemes.wav
"""

import json

import pykokoro
from pykokoro.phonemes import PhonemeBook

# Text with heavy punctuation usage (same as punctuation.py)
TEXT = """
"Well," said the professor; "this is quite extraordinary!"

The experiment — which took years to complete — yielded surprising results:
success rates of 95%, 87%, and 72% (in that order).

"But wait..." she paused, "are you absolutely sure?"

Yes! No? Maybe... Who knows: life is full of mysteries;
that's what makes it interesting.

Consider this: the data shows (a) increased efficiency;
(b) reduced costs; and (c) improved outcomes — all remarkable achievements!

"To be, or not to be?" — that is the question.

He shouted: "Eureka!" Then whispered... "Finally."

The results were: excellent (A+), good (B), average (C);
however — and this is important — none failed!

"Why?" she asked. "Because," he replied, "science never sleeps..."
"""

VOICE = "af_bella"  # American Female voice
LANG = "en-us"  # American English


def main():
    """Export text to phoneme JSON file."""
    print("Initializing TTS engine...")
    kokoro = pykokoro.Kokoro()

    print("Text with punctuation marks: ; : , . ! ? — … \" ( ) ' '")
    print(f"Voice: {VOICE}")
    print(f"Language: {LANG}")

    # Create a PhonemeBook to store the phonemes
    book = PhonemeBook(
        title="Punctuation Demo",
        lang=LANG,
        metadata={
            "voice": VOICE,
            "description": "Punctuation stress test with various marks",
        },
    )

    # Create a chapter and add segments
    chapter = book.create_chapter("Punctuation Examples")

    print("\nTokenizing text into phonemes...")

    # Add text with sentence-level splitting for natural segments
    segments = chapter.add_text(
        TEXT,
        tokenizer=kokoro.tokenizer,
        lang=LANG,
        split_mode="sentence",  # Split on sentence boundaries
    )

    print(f"Created {len(segments)} segments")

    # Show some statistics
    print("\nPhoneme Statistics:")
    print(f"  Total segments: {book.total_segments}")
    print(f"  Total phonemes: {book.total_phonemes} characters")
    print(f"  Total tokens: {book.total_tokens}")

    # Show first few segments as examples
    print("\nFirst 3 segments:")
    for i, seg in enumerate(segments[:3]):
        print(f"\n  Segment {i + 1}:")
        print(f"    Text: {seg.text[:60]}{'...' if len(seg.text) > 60 else ''}")
        phoneme_preview = (
            f"{seg.phonemes[:60]}{'...' if len(seg.phonemes) > 60 else ''}"
        )
        print(f"    Phonemes: {phoneme_preview}")
        print(f"    Tokens: {len(seg.tokens)} tokens")

    # Save to JSON file
    output_file = "punctuation_phonemes.json"
    book.save(output_file)
    print(f"\nSaved phonemes to: {output_file}")

    # Also show a snippet of the JSON structure
    print("\nJSON structure preview:")
    data = book.to_dict()
    preview = {
        "format_version": data["format_version"],
        "title": data["title"],
        "lang": data["lang"],
        "stats": data["stats"],
        "chapters": f"[{len(data['chapters'])} chapter(s)]",
    }
    print(json.dumps(preview, indent=2))

    print("\nTo convert to audio, run:")
    print(f"  pykokoro phonemes convert {output_file} -v {VOICE}")

    kokoro.close()


if __name__ == "__main__":
    main()
