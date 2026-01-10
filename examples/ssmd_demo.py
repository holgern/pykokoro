#!/usr/bin/env python3
"""
SSMD (Speech Synthesis Markdown) Demo for pykokoro.

This example demonstrates the comprehensive SSMD markup features supported
by pykokoro, including breaks, emphasis, language switching, phonetic
pronunciation using kokorog2p format, substitution, and markers.

SSMD provides a rich, readable way to control TTS output with markup that
looks natural in text form.

Usage:
    python examples/ssmd_demo.py

Output:
    ssmd_story_demo.wav - Generated speech with SSMD markup

Features demonstrated:
    - Break markers: ...c ...s ...p ...800ms ...2s
    - Emphasis: *moderate* **strong**
    - Language switching: [Bonjour](fr)
    - Phonetic pronunciation: [creak](/kri:k/) using kokorog2p format
    - Substitution: [H2O](sub: water)
    - Markers: @location @moment
"""

import soundfile as sf

import pykokoro

# Story with comprehensive SSMD markup
# Note: Using kokorog2p format [word](/phoneme/) instead of SSMD ph: syntax
story = """
Chapter 3: The Discovery

[Emma](en-GB) stepped into the dusty library ...800ms her eyes adjusting to
the dim light. **"This is it"** she whispered to herself.

The ancient book lay on the pedestal @book_location, exactly where the
map had indicated.

She approached slowly ...c her footsteps echoing in the silence.
Each step seemed to make a [creak](/kri:k/) sound in the quiet.

The Revelation

As she opened the book ...1s a brilliant *golden light* erupted from the pages!
The mysterious symbol looked like [H2O](sub: water) but glowed with power.

[Emma](en-GB) felt a warmth spreading through her fingers.
She knew @moment_of_truth that her life would never be the same again ...2s

[Fin](fr)
"""

# Alternative: French story example
french_story = """
Chapitre 1: La Decouverte

[Marie](fr-fr) marchait dans la vieille bibliotheque ...800ms
ses pas resonnaient dans le silence.

**"C'est incroyable"** murmura-t-elle ...s admirant les livres anciens.

Elle trouva le manuscrit @manuscrit_ancien exactement ou la carte
l'avait indique ...c puis elle l'ouvrit lentement.

Une *lumiere doree* jaillit des pages! ...1s

[Marie](fr-fr) savait @moment_important que sa vie allait changer ...2s

[The End](en-us)
"""

VOICE = "af_sarah"  # American Female voice
LANG = "en-us"  # Default language

FRENCH_VOICE = "ff_siwis"  # French Female voice
FRENCH_LANG = "fr-fr"


def main():
    """Generate speech with SSMD markup."""
    print("=" * 70)
    print("SSMD (Speech Synthesis Markdown) Demo")
    print("=" * 70)

    kokoro = pykokoro.Kokoro()

    # Demo 1: English story with SSMD markup
    print("\n--- Demo 1: English Story with SSMD Markup ---")
    print(f"Voice: {VOICE}")
    print(f"Language: {LANG}")
    print("\nStory text (with SSMD markup):")
    print("-" * 70)
    print(story)
    print("-" * 70)

    print("\nGenerating audio with SSMD markup...")
    samples, sample_rate = kokoro.create(
        story,
        voice=VOICE,
        speed=1.0,
        lang=LANG,
    )

    output_file = "ssmd_story_demo.wav"
    sf.write(output_file, samples, sample_rate)

    duration = len(samples) / sample_rate
    print(f"\nCreated: {output_file}")
    print(f"Duration: {duration:.2f} seconds")

    # Demo 2: French story with SSMD markup
    print("\n--- Demo 2: French Story with SSMD Markup ---")
    print(f"Voice: {FRENCH_VOICE}")
    print(f"Language: {FRENCH_LANG}")
    print("\nStory text (with SSMD markup):")
    print("-" * 70)
    print(french_story)
    print("-" * 70)

    print("\nGenerating audio with SSMD markup...")
    samples_fr, sample_rate_fr = kokoro.create(
        french_story,
        voice=FRENCH_VOICE,
        speed=1.0,
        lang=FRENCH_LANG,
    )

    output_file_fr = "ssmd_french_story_demo.wav"
    sf.write(output_file_fr, samples_fr, sample_rate_fr)

    duration_fr = len(samples_fr) / sample_rate_fr
    print(f"\nCreated: {output_file_fr}")
    print(f"Duration: {duration_fr:.2f} seconds")

    # Demo 3: Show SSMD break markers
    print("\n--- Demo 3: SSMD Break Markers ---")
    break_examples = """
Comma pause ...c like this.
Sentence pause ...s like this.
Paragraph pause ...p like this.
Custom 500ms pause ...500ms like this.
Custom 2 second pause ...2s like this.
"""
    print("Break examples:")
    print(break_examples)

    print("\nGenerating audio with various breaks...")
    samples_breaks, _ = kokoro.create(
        break_examples,
        voice=VOICE,
        speed=1.0,
        lang=LANG,
    )

    output_file_breaks = "ssmd_breaks_demo.wav"
    sf.write(output_file_breaks, samples_breaks, sample_rate)
    print(f"Created: {output_file_breaks}")

    # Demo 4: Emphasis and substitution
    print("\n--- Demo 4: Emphasis and Substitution ---")
    emphasis_text = """
This is *moderate emphasis*.
This is **strong emphasis**.
The formula [H2O](sub: water) is essential for life.
"""
    print("Emphasis examples:")
    print(emphasis_text)

    print("\nGenerating audio with emphasis...")
    samples_emphasis, _ = kokoro.create(
        emphasis_text,
        voice=VOICE,
        speed=1.0,
        lang=LANG,
    )

    output_file_emphasis = "ssmd_emphasis_demo.wav"
    sf.write(output_file_emphasis, samples_emphasis, sample_rate)
    print(f"Created: {output_file_emphasis}")

    # Demo 5: Voice switching
    print("\n--- Demo 5: Voice Switching ---")
    voice_switching_text = """
[Hello, I'm Sarah!](voice: af_sarah) ...s
[And I'm Michael!](voice: am_michael) ...s
[Nice to meet you both!](voice: af_nicole)
"""
    print("Voice switching example:")
    print(voice_switching_text)

    print("\nGenerating audio with automatic voice switching...")
    samples_voices, _ = kokoro.create(
        voice_switching_text,
        voice=VOICE,  # Default voice
        speed=1.0,
        lang=LANG,
    )

    output_file_voices = "ssmd_voice_switching_demo.wav"
    sf.write(output_file_voices, samples_voices, sample_rate)
    print(f"Created: {output_file_voices}")
    print("Note: Each segment automatically uses its annotated voice!")

    kokoro.close()

    print("\n" + "=" * 70)
    print("SSMD Demo Complete!")
    print("=" * 70)
    print("\nSSMD Features Used:")
    print("  ✓ Break markers: ...c ...s ...p ...500ms ...2s")
    print("  ✓ Emphasis: *moderate* **strong**")
    print("  ✓ Language switching: [text](lang-code)")
    print("  ✓ Phonetic pronunciation: [word](/phoneme/) (kokorog2p format)")
    print("  ✓ Substitution: [text](sub: replacement)")
    print("  ✓ Markers: @marker_name")
    print("  ✓ Voice switching: [text](voice: name) - NEW!")
    print("\nNote: PyKokoro uses kokorog2p format [word](/phoneme/) for")
    print("      phonetic pronunciation, not SSMD's [word](ph: phoneme) syntax.")
    print("\nVoice switching happens automatically per segment!")
    print("Each [text](voice: name) annotation switches to that voice.")


if __name__ == "__main__":
    main()
