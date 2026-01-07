Examples
========

This page provides practical examples for common use cases.

Hello World
-----------

The simplest example:

.. code-block:: python

   from pykokoro import Kokoro
   import soundfile as sf

   with Kokoro() as kokoro:
       audio, sr = kokoro.create("Hello, world!", voice="af_bella")
       sf.write("hello.wav", audio, sr)

Multi-Voice Demo
----------------

Generate the same text with different voices:

.. code-block:: python

   from pykokoro import Kokoro
   import soundfile as sf

   text = "This is a demonstration of different voices."

   voices = [
       ("af_bella", "American Female - Bella"),
       ("am_adam", "American Male - Adam"),
       ("bf_emma", "British Female - Emma"),
       ("bm_george", "British Male - George"),
   ]

   with Kokoro() as kokoro:
       for voice_name, description in voices:
           print(f"Generating: {description}")
           audio, sr = kokoro.create(text, voice=voice_name)
           sf.write(f"voice_{voice_name}.wav", audio, sr)

Pause Markers Demo
------------------

Demonstrate different pause durations:

.. code-block:: python

   from pykokoro import Kokoro
   import soundfile as sf

   text = """
   This is a sentence with a short pause. (.)
   Now a medium pause. (..)
   And finally a long pause. (...)
   Back to normal.
   """

   with Kokoro() as kokoro:
       audio, sr = kokoro.create(
           text,
           voice="af_bella",
           enable_pauses=True
       )
       sf.write("pauses_demo.wav", audio, sr)

Custom Pause Durations
~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   from pykokoro import Kokoro
   import soundfile as sf

   text = "Custom (.) pauses (..) here (...)"

   with Kokoro() as kokoro:
       audio, sr = kokoro.create(
           text,
           voice="af_bella",
           enable_pauses=True,
           pause_short=0.2,    # 200ms
           pause_medium=0.5,   # 500ms
           pause_long=1.0      # 1 second
       )
       sf.write("custom_pauses.wav", audio, sr)

Voice Blending
--------------

Simple Blend
~~~~~~~~~~~~

.. code-block:: python

   from pykokoro import Kokoro, VoiceBlend
   import soundfile as sf

   with Kokoro() as kokoro:
       # Equal blend
       blend = VoiceBlend.parse("af_bella + af_sarah")

       audio, sr = kokoro.create(
           "This is a blended voice",
           voice=blend
       )
       sf.write("blended.wav", audio, sr)

Weighted Blend
~~~~~~~~~~~~~~

.. code-block:: python

   from pykokoro import Kokoro, VoiceBlend
   import soundfile as sf

   with Kokoro() as kokoro:
       # 70% bella, 30% sarah
       blend = VoiceBlend.parse("af_bella*0.7 + af_sarah*0.3")

       audio, sr = kokoro.create(
           "Weighted blend example",
           voice=blend
       )
       sf.write("weighted_blend.wav", audio, sr)

Multi-Language Support
----------------------

Spanish
~~~~~~~

.. code-block:: python

   from pykokoro import Kokoro
   import soundfile as sf

   text = "Hola, ¿cómo estás? Este es un ejemplo en español."

   with Kokoro() as kokoro:
       audio, sr = kokoro.create(
           text,
           voice="af_nicole",  # Spanish voice
           lang="es"
       )
       sf.write("spanish.wav", audio, sr)

French
~~~~~~

.. code-block:: python

   from pykokoro import Kokoro
   import soundfile as sf

   text = "Bonjour! Ceci est un exemple en français."

   with Kokoro() as kokoro:
       audio, sr = kokoro.create(
           text,
           voice="af_sarah",  # French voice
           lang="fr"
       )
       sf.write("french.wav", audio, sr)

German
~~~~~~

.. code-block:: python

   from pykokoro import Kokoro
   import soundfile as sf

   text = "Guten Tag! Dies ist ein Beispiel auf Deutsch."

   with Kokoro() as kokoro:
       audio, sr = kokoro.create(
           text,
           voice="af_anna",  # German voice
           lang="de"
       )
       sf.write("german.wav", audio, sr)

Long Text Processing
--------------------

With Sentence Splitting
~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   from pykokoro import Kokoro
   import soundfile as sf

   long_text = """
   This is a long passage of text that demonstrates sentence splitting.
   Each sentence will be processed separately for better quality.

   This is a new paragraph. It will also be handled efficiently.
   The split_mode parameter controls how text is divided.
   """

   with Kokoro() as kokoro:
       audio, sr = kokoro.create(
           long_text,
           voice="af_bella",
           split_mode="sentence"
       )
       sf.write("long_text.wav", audio, sr)

With Pause Markers and Splitting
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   from pykokoro import Kokoro
   import soundfile as sf

   text = """
   Welcome to this demonstration. (..)

   This text uses both pause markers and sentence splitting. (.)
   The combination creates very natural-sounding speech. (..)

   Try it yourself!
   """

   with Kokoro() as kokoro:
       audio, sr = kokoro.create(
           text,
           voice="af_bella",
           enable_pauses=True,
           split_mode="sentence"
       )
       sf.write("combined_demo.wav", audio, sr)

Automatic Natural Pauses (NEW!)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

For the most natural-sounding speech, use automatic pause insertion:

.. code-block:: python

   from pykokoro import Kokoro
   import soundfile as sf

   text = """
   Artificial intelligence is transforming our world. Machine learning
   models are becoming more sophisticated, efficient, and accessible.

   Deep learning uses neural networks with many layers. These networks
   can learn complex patterns, enabling breakthroughs in vision and
   language processing.
   """

   with Kokoro() as kokoro:
       audio, sr = kokoro.create(
           text,
           voice="af_sarah",
           split_mode="clause",      # Split on commas and sentences
           trim_silence=True,        # Enable automatic pauses
           pause_clause=0.25,         # Clause pauses
           pause_sentence=0.5,         # Sentence pauses
           pause_paragraph=1.0,           # Paragraph pauses
           pause_variance=0.05,      # Natural variance
           random_seed=42            # For reproducibility
       )
       sf.write("automatic_pauses.wav", audio, sr)

**Key Benefits:**

* Automatic detection of clause, sentence, and paragraph boundaries
* Natural Gaussian variance prevents robotic timing
* No manual markers needed
* Combines with manual markers for maximum control

See ``examples/automatic_pauses_demo.py`` for a complete demonstration.

Speed Control
-------------

Variable Speed Example
~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   from pykokoro import Kokoro
   import soundfile as sf
   import numpy as np

   text = "This sentence demonstrates different speech speeds."

   speeds = [0.5, 0.75, 1.0, 1.25, 1.5, 2.0]

   with Kokoro() as kokoro:
       audio_parts = []

       for speed in speeds:
           audio, sr = kokoro.create(
               f"Speed {speed}x: {text}",
               voice="af_bella",
               speed=speed
           )
           audio_parts.append(audio)

           # Add silence between examples
           silence = np.zeros(int(sr * 0.5), dtype=np.float32)
           audio_parts.append(silence)

       # Concatenate all parts
       final_audio = np.concatenate(audio_parts)
       sf.write("speed_demo.wav", final_audio, sr)

Batch Processing
----------------

Process Multiple Files
~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   from pykokoro import Kokoro
   import soundfile as sf
   from pathlib import Path

   # Input: dictionary of filename -> text
   scripts = {
       "intro": "Welcome to our podcast!",
       "segment1": "This is the first segment.",
       "segment2": "This is the second segment.",
       "outro": "Thank you for listening!",
   }

   output_dir = Path("podcast_segments")
   output_dir.mkdir(exist_ok=True)

   with Kokoro() as kokoro:
       for filename, text in scripts.items():
           print(f"Generating {filename}...")

           audio, sr = kokoro.create(
               text,
               voice="af_bella",
               speed=1.0
           )

           output_path = output_dir / f"{filename}.wav"
           sf.write(output_path, audio, sr)

           print(f"  Saved to {output_path}")

Process CSV File
~~~~~~~~~~~~~~~~

.. code-block:: python

   import csv
   from pykokoro import Kokoro
   import soundfile as sf
   from pathlib import Path

   # CSV format: id,text,voice,speed
   csv_file = "scripts.csv"
   output_dir = Path("outputs")
   output_dir.mkdir(exist_ok=True)

   with Kokoro() as kokoro:
       with open(csv_file, 'r', encoding='utf-8') as f:
           reader = csv.DictReader(f)

           for row in reader:
               audio_id = row['id']
               text = row['text']
               voice = row.get('voice', 'af_bella')
               speed = float(row.get('speed', 1.0))

               print(f"Processing {audio_id}...")

               audio, sr = kokoro.create(
                   text,
                   voice=voice,
                   speed=speed
               )

               output_path = output_dir / f"{audio_id}.wav"
               sf.write(output_path, audio, sr)

Phoneme-Based Generation
-------------------------

Text to Phonemes
~~~~~~~~~~~~~~~~

.. code-block:: python

   from pykokoro import Kokoro

   with Kokoro() as kokoro:
       text = "Hello, world!"

       # Convert to phonemes
       phonemes = kokoro.tokenizer.phonemize(text, lang="en-us")
       print(f"Text: {text}")
       print(f"Phonemes: {phonemes}")

       # Generate from phonemes
       audio, sr = kokoro.create_from_phonemes(
           phonemes,
           voice="af_bella"
       )

Custom Phoneme Sequences
~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   from pykokoro import Kokoro
   import soundfile as sf

   # Custom IPA phoneme sequence
   custom_phonemes = "həˈloʊ wɝld"

   with Kokoro() as kokoro:
       audio, sr = kokoro.create_from_phonemes(
           custom_phonemes,
           voice="af_bella",
           speed=1.0
       )
       sf.write("custom_phonemes.wav", audio, sr)

Advanced Text Splitting
~~~~~~~~~~~~~~~~~~~~~~~~

The ``split_and_phonemize_text`` function provides fine-grained control over
how text is segmented for TTS processing. This is useful for:

- Long-form content (audiobooks, articles, podcasts)
- Controlling prosody and natural pauses
- Preventing phoneme buffer overflow
- Tracking paragraph and sentence structure

.. code-block:: python

   from pykokoro import Tokenizer
   from pykokoro.phonemes import split_and_phonemize_text

   text = """
   The sun rises in the east. Birds begin to sing. The day starts fresh.

   Coffee brews in the kitchen. Toast pops from the toaster. Breakfast is ready.
   """

   tokenizer = Tokenizer()

   # Split by paragraph (double newlines)
   segments = split_and_phonemize_text(
       text=text,
       tokenizer=tokenizer,
       split_mode="paragraph",
       lang="en-us"
   )
   print(f"Paragraphs: {len(segments)} segments")
   # Output: Paragraphs: 2 segments

   # Split by sentence (requires spaCy)
   segments = split_and_phonemize_text(
       text=text,
       tokenizer=tokenizer,
       split_mode="sentence",
       lang="en-us"
   )
   print(f"Sentences: {len(segments)} segments")
   # Output: Sentences: 6 segments

   # Split by clause (sentences + commas, requires spaCy)
   segments = split_and_phonemize_text(
       text=text,
       tokenizer=tokenizer,
       split_mode="clause",
       lang="en-us"
   )
   print(f"Clauses: {len(segments)} segments")

   # Each segment contains:
   for seg in segments:
       print(f"Paragraph {seg.paragraph}, Sentence {seg.sentence}")
       print(f"  Text: {seg.text}")
       print(f"  Phonemes: {seg.phonemes}")
       print(f"  Tokens: {len(seg.tokens)}")

.. note::
   For a complete working example with all split modes, see:
   ``examples/split_and_phonemize_demo.py``

Audiobook Generation
--------------------

Chapter Processing
~~~~~~~~~~~~~~~~~~

.. code-block:: python

   from pykokoro import Kokoro
   import soundfile as sf
   from pathlib import Path

   chapters = {
       "Chapter 1": """
       This is the first chapter of our story. (..)
       It was a dark and stormy night.
       """,
       "Chapter 2": """
       The second chapter continues the adventure. (..)
       Our hero embarks on a journey.
       """,
   }

   output_dir = Path("audiobook")
   output_dir.mkdir(exist_ok=True)

   with Kokoro() as kokoro:
       for i, (title, text) in enumerate(chapters.items(), 1):
           print(f"Processing {title}...")

           # Add chapter announcement
           full_text = f"{title}. (...) {text}"

           audio, sr = kokoro.create(
               full_text,
               voice="af_bella",
               speed=1.0,
               enable_pauses=True,
               split_mode="sentence"
           )

           output_file = output_dir / f"chapter_{i:02d}.wav"
           sf.write(output_file, audio, sr)
           print(f"  Saved to {output_file}")

Real-Time Streaming (Conceptual)
---------------------------------

Process and Play
~~~~~~~~~~~~~~~~

.. code-block:: python

   from pykokoro import Kokoro
   import sounddevice as sd
   import queue
   import threading

   def audio_callback(outdata, frames, time, status):
       """Callback for audio playback."""
       try:
           data = audio_queue.get_nowait()
           if len(data) < len(outdata):
               outdata[:len(data)] = data.reshape(-1, 1)
               outdata[len(data):] = 0
           else:
               outdata[:] = data[:len(outdata)].reshape(-1, 1)
       except queue.Empty:
           outdata[:] = 0

   # Queue for audio chunks
   audio_queue = queue.Queue()

   with Kokoro() as kokoro:
       # Generate audio
       audio, sr = kokoro.create(
           "This is a streaming example",
           voice="af_bella"
       )

       # Chunk and queue
       chunk_size = 1024
       for i in range(0, len(audio), chunk_size):
           chunk = audio[i:i+chunk_size]
           audio_queue.put(chunk)

       # Play
       with sd.OutputStream(
           channels=1,
           samplerate=sr,
           callback=audio_callback
       ):
           while not audio_queue.empty():
               sd.sleep(100)

Flask Web Service
-----------------

TTS API Endpoint
~~~~~~~~~~~~~~~~

.. code-block:: python

   from flask import Flask, request, send_file
   from pykokoro import Kokoro
   import soundfile as sf
   from io import BytesIO
   import numpy as np

   app = Flask(__name__)
   kokoro = Kokoro()

   @app.route('/tts', methods=['POST'])
   def text_to_speech():
       data = request.json
       text = data.get('text', '')
       voice = data.get('voice', 'af_bella')
       speed = float(data.get('speed', 1.0))

       if not text:
           return {'error': 'No text provided'}, 400

       # Generate audio
       audio, sr = kokoro.create(text, voice=voice, speed=speed)

       # Convert to bytes
       buffer = BytesIO()
       sf.write(buffer, audio, sr, format='WAV')
       buffer.seek(0)

       return send_file(
           buffer,
           mimetype='audio/wav',
           as_attachment=True,
           download_name='speech.wav'
       )

   @app.route('/voices', methods=['GET'])
   def list_voices():
       voices = kokoro.list_voices()
       return {'voices': voices}

   if __name__ == '__main__':
       app.run(host='0.0.0.0', port=5000)

Command-Line Tool
-----------------

Simple CLI
~~~~~~~~~~

.. code-block:: python

   #!/usr/bin/env python3
   """Simple command-line TTS tool."""

   import argparse
   from pykokoro import Kokoro
   import soundfile as sf

   def main():
       parser = argparse.ArgumentParser(description='Text-to-Speech CLI')
       parser.add_argument('text', help='Text to synthesize')
       parser.add_argument('-o', '--output', required=True, help='Output WAV file')
       parser.add_argument('-v', '--voice', default='af_bella', help='Voice name')
       parser.add_argument('-s', '--speed', type=float, default=1.0, help='Speech speed')
       parser.add_argument('--pauses', action='store_true', help='Enable pause markers')

       args = parser.parse_args()

       with Kokoro() as kokoro:
           audio, sr = kokoro.create(
               args.text,
               voice=args.voice,
               speed=args.speed,
               enable_pauses=args.pauses
           )
           sf.write(args.output, audio, sr)
           print(f"Generated {args.output}")

   if __name__ == '__main__':
       main()

Usage:

.. code-block:: bash

   python tts_cli.py "Hello, world!" -o output.wav -v af_bella -s 1.0
   python tts_cli.py "With (.) pauses" -o pauses.wav --pauses
