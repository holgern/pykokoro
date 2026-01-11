Quick Start Guide
=================

This guide will get you up and running with PyKokoro in just a few minutes.

First Steps
-----------

1. **Install PyKokoro**

   .. code-block:: bash

      pip install pykokoro

2. **Verify Installation**

   .. code-block:: python

      import pykokoro
      print(pykokoro.__version__)

Basic Usage
-----------

Generate Your First Audio
~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   from pykokoro import Kokoro

   # Initialize the TTS engine
   kokoro = Kokoro()

   # Generate speech from text
   audio, sample_rate = kokoro.create(
       "Hello! Welcome to PyKokoro text-to-speech.",
       voice="af_bella"  # American Female voice
   )

   # Save to a WAV file
   import soundfile as sf
   sf.write("hello.wav", audio, sample_rate)

   # Clean up
   kokoro.close()

That's it! You've generated your first audio file.

Choosing a Voice
~~~~~~~~~~~~~~~~

PyKokoro comes with 54 voices (v1.0) or 103 voices (v1.1-zh). Here are some popular ones:

.. code-block:: python

   from pykokoro import Kokoro

   kokoro = Kokoro()

   # American English
   audio1, sr = kokoro.create("American female voice", voice="af_bella")
   audio2, sr = kokoro.create("American male voice", voice="am_adam")

   # British English
   audio3, sr = kokoro.create("British female voice", voice="bf_emma")
   audio4, sr = kokoro.create("British male voice", voice="bm_george")

   # Other languages
   audio5, sr = kokoro.create("Hola, mundo", voice="af_nicole", lang="es")
   audio6, sr = kokoro.create("Bonjour le monde", voice="af_sarah", lang="fr")

   kokoro.close()

To see all available voices:

.. code-block:: python

   from pykokoro import Kokoro

   kokoro = Kokoro()
   voices = kokoro.list_voices()
   for voice in voices:
       print(voice)

Adjusting Speech Speed
~~~~~~~~~~~~~~~~~~~~~~

Control how fast or slow the speech is:

.. code-block:: python

   from pykokoro import Kokoro

   kokoro = Kokoro()

   # Normal speed (default)
   audio1, sr = kokoro.create("Normal speed", voice="af_bella", speed=1.0)

   # Slower (0.5x)
   audio2, sr = kokoro.create("Slower speech", voice="af_bella", speed=0.5)

   # Faster (1.5x)
   audio3, sr = kokoro.create("Faster speech", voice="af_bella", speed=1.5)

   kokoro.close()

Adding Pauses
~~~~~~~~~~~~~

Add natural pauses in your speech using SSMD break syntax:

.. code-block:: python

   from pykokoro import Kokoro

   kokoro = Kokoro()

   text = """
   Welcome to the tutorial ...c
   This is a short pause ...s
   And this is a longer pause ...p
   These pauses make speech sound more natural.
   """

   audio, sr = kokoro.create(
       text,
       voice="af_bella"
   )

   import soundfile as sf
   sf.write("with_pauses.wav", audio, sr)

   kokoro.close()

Pause syntax (SSMD breaks):
* ``...c`` - Short/comma pause (0.3 seconds, default)
* ``...s`` - Medium/sentence pause (0.6 seconds, default)
* ``...p`` - Long/paragraph pause (1.0 seconds, default)
* ``...500ms`` - Custom duration pause (e.g., 500 milliseconds)

Context Manager Usage
~~~~~~~~~~~~~~~~~~~~~

Use Kokoro as a context manager for automatic cleanup:

.. code-block:: python

   from pykokoro import Kokoro

   with Kokoro() as kokoro:
       audio, sr = kokoro.create("Hello, world!", voice="af_bella")

       import soundfile as sf
       sf.write("output.wav", audio, sr)
   # Automatically closed

Processing Long Text
~~~~~~~~~~~~~~~~~~~~

For long text, use ``split_mode`` to automatically split at natural boundaries:

.. code-block:: python

   from pykokoro import Kokoro

   long_text = """
   This is a long passage of text. It has multiple sentences.
   The split mode will handle this intelligently.

   This is a new paragraph. It will be processed efficiently.
   """

   with Kokoro() as kokoro:
       audio, sr = kokoro.create(
           long_text,
           voice="af_bella",
           split_mode="sentence"  # Split at sentence boundaries
       )

       import soundfile as sf
       sf.write("long_text.wav", audio, sr)

Complete Example
----------------

Here's a complete example putting it all together:

.. code-block:: python

   from pykokoro import Kokoro
   import soundfile as sf

   def text_to_speech(text, output_file, voice="af_bella", speed=1.0):
       """Convert text to speech and save to file."""
       with Kokoro() as kokoro:
           audio, sample_rate = kokoro.create(
               text,
               voice=voice,
               speed=speed,
               split_mode="sentence"
           )
           sf.write(output_file, audio, sample_rate)
           print(f"Saved audio to {output_file}")

   # Example usage
   text = """
   Welcome to PyKokoro! ...s

   This library makes text-to-speech generation simple ...c
   You can control voice, speed, and add natural pauses ...s

   Enjoy creating audio content!
   """

   text_to_speech(text, "welcome.wav", voice="af_bella", speed=1.0)

Next Steps
----------

Now that you know the basics, explore:

* :doc:`basic_usage` - Detailed usage guide
* :doc:`advanced_features` - Voice blending, phoneme control, and more
* :doc:`examples` - More examples and use cases
* :doc:`api_reference` - Complete API documentation
