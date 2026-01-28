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

from pykokoro import build_pipeline

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

VOICE = "df_eva"
# VOICE = "dm_bernd"
LANG = "de"  # German language code for espeak-ng phonemization


def main():
    """Generate German speech using English-trained voice."""
    print("Initializing TTS engine...")
    pipe = build_pipeline(
        config={
            "voice": VOICE,
            "model_source": "github",
            "model_variant": "v1.1-de",
            "generation": {
                "lang": LANG,
            },
        }
    )

    print("=" * 60)
    print("The model from Tundragoon/Kokoro-German is used for German.")
    print("Pronunciation may not be perfect or native-sounding.")
    print("=" * 60)
    print(f"\nVoice: {VOICE}")
    print(f"Language: {LANG}")

    print("\nGenerating audio...")
    res = pipe.run(TEXT)
    samples, sample_rate = res.audio, res.sample_rate

    output_file = "german_demo.wav"
    sf.write(output_file, samples, sample_rate)

    duration = len(samples) / sample_rate
    print(f"\nCreated {output_file}")
    print(f"Duration: {duration:.2f} seconds")


if __name__ == "__main__":
    main()
