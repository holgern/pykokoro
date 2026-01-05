#!/usr/bin/env python3
"""
German text example using pykokoro.

This example demonstrates how pykokoro handles German text using the af_bella voice.
Note: The Kokoro model was not explicitly trained on German, so pronunciation may
not be perfect. The model will attempt to phonemize German text using espeak-ng.

Usage:
    python examples/german.py

Output:
    german_demo.wav - Generated German speech
"""

import logging
import os

import soundfile as sf

import pykokoro
from pykokoro import VoiceBlend

# Enable phoneme debugging to see what phonemes are generated
os.environ["PYKOKORO_DEBUG_PHONEMES"] = "1"

# Configure logging to display phoneme debug information
logging.basicConfig(level=logging.INFO, format="%(levelname)s [%(name)s] %(message)s")

# German text samples
TEXT = """
Guten Tag! Willkommen zu diesem Beispiel der deutschen Sprache.

Die deutsche Sprache hat viele besondere Eigenschaften.
Sie ist bekannt für ihre langen zusammengesetzten Wörter wie
Donaudampfschifffahrtsgesellschaft oder Kraftfahrzeughaftpflichtversicherung.

Heute ist ein schöner Tag. Die Sonne scheint, und die Vögel singen.
Ich möchte gerne einen Kaffee trinken und ein Buch lesen.

Zahlen sind auch wichtig: eins, zwei, drei, vier, fünf, sechs, sieben, acht, neun, zehn.

Umlaute sind charakteristisch für Deutsch: ä, ö, ü und das Eszett ß.
Käse, Brötchen, Müller, Straße.

Fragen Sie mich, wie es Ihnen geht?
Es geht mir sehr gut, danke schön!

Die Wissenschaft macht große Fortschritte.
Technologie verändert unsere Welt jeden Tag.

Auf Wiedersehen und vielen Dank fürs Zuhören!
"""

# Voice options - uncomment one to use:
# VOICE = "af_bella"  # American Female (English-trained, attempting German)
# VOICE = "ef_dora"   # Spanish Female (Romance language, may handle German better)
# VOICE = "ff_siwis"  # French Female (Romance language, may handle German better)
# VOICE = "ff_siwis"  # Using French voice for better German pronunciation
BLEND = "ff_siwis:50,ef_dora:50"  # Blend of French and Spanish voices

LANG = "de"  # German language code for espeak-ng phonemization


def main():
    """Generate German speech using English-trained voice."""
    print("Initializing TTS engine...")
    kokoro = pykokoro.Kokoro()

    print("=" * 60)
    print("NOTE: Kokoro was NOT explicitly trained on German.")
    print("The model will attempt German phonemization via espeak-ng.")
    print("Pronunciation may not be perfect or native-sounding.")
    print("=" * 60)
    print(f"\nVoice: {BLEND}")
    print(f"Language: {LANG}")

    print("\nGenerating audio...")
    samples, sample_rate = kokoro.create(
        TEXT,
        voice=VoiceBlend.parse(BLEND),
        speed=1.0,
        lang=LANG,
    )

    output_file = "german_demo.wav"
    sf.write(output_file, samples, sample_rate)

    duration = len(samples) / sample_rate
    print(f"\nCreated {output_file}")
    print(f"Duration: {duration:.2f} seconds")

    kokoro.close()


if __name__ == "__main__":
    main()
