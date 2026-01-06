Basic Usage
===========

This guide covers the fundamental usage patterns of PyKokoro.

Initializing Kokoro
-------------------

The main entry point is the ``Kokoro`` class:

.. code-block:: python

   from pykokoro import Kokoro

   # Initialize with defaults
   kokoro = Kokoro()

   # Or specify model quality
   kokoro = Kokoro(model_quality="q8")  # q8, q6, or fp16

   # Or specify device
   kokoro = Kokoro(device="cuda")  # cuda, cpu, rocm

   # Clean up when done
   kokoro.close()

Using Context Manager
~~~~~~~~~~~~~~~~~~~~~

The recommended way is using a context manager:

.. code-block:: python

   from pykokoro import Kokoro

   with Kokoro() as kokoro:
       audio, sr = kokoro.create("Hello!", voice="af_bella")
       # Kokoro automatically closed when exiting context

Model Quality Options
~~~~~~~~~~~~~~~~~~~~~

* ``fp16`` - Full precision (highest quality, largest size)
* ``q8`` - 8-bit quantized (default, good balance)
* ``q6`` - 6-bit quantized (smaller, slightly lower quality)

.. code-block:: python

   # High quality
   kokoro_hq = Kokoro(model_quality="fp16")

   # Balanced (default)
   kokoro = Kokoro(model_quality="q8")

   # Compact
   kokoro_small = Kokoro(model_quality="q6")

Generating Speech
-----------------

Basic Text-to-Speech
~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   from pykokoro import Kokoro

   with Kokoro() as kokoro:
       audio, sample_rate = kokoro.create(
           "Hello, world!",
           voice="af_bella"
       )

The ``create()`` method returns:

* ``audio`` - NumPy array of audio samples (float32)
* ``sample_rate`` - Sample rate in Hz (typically 24000)

Saving Audio
~~~~~~~~~~~~

Using soundfile (recommended):

.. code-block:: python

   import soundfile as sf

   with Kokoro() as kokoro:
       audio, sr = kokoro.create("Hello!", voice="af_bella")
       sf.write("output.wav", audio, sr)

Using scipy:

.. code-block:: python

   from scipy.io import wavfile

   with Kokoro() as kokoro:
       audio, sr = kokoro.create("Hello!", voice="af_bella")
       # scipy requires int16 format
       audio_int16 = (audio * 32767).astype('int16')
       wavfile.write("output.wav", sr, audio_int16)

Voice Selection
---------------

List Available Voices
~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   with Kokoro() as kokoro:
       voices = kokoro.list_voices()
       for voice in voices:
           print(voice)

Voice Name Format
~~~~~~~~~~~~~~~~~

Voice names follow the pattern: ``{accent}_{gender}_{name}``

* **Accent**: ``af`` (American Female), ``am`` (American Male), ``bf`` (British Female), ``bm`` (British Male)
* **Gender**: ``f`` (female), ``m`` (male)
* **Name**: Specific voice identifier

Common Voices
~~~~~~~~~~~~~

**American English:**

* ``af_bella`` - Female, clear and neutral
* ``af_sarah`` - Female, warm and friendly
* ``am_adam`` - Male, deep and authoritative
* ``am_michael`` - Male, clear and professional

**British English:**

* ``bf_emma`` - Female, refined accent
* ``bf_isabella`` - Female, gentle
* ``bm_george`` - Male, distinguished
* ``bm_lewis`` - Male, contemporary

**Other Languages:**

See the :doc:`api_reference` for a complete list of voices for Spanish, French, German, Italian, Portuguese, Hindi, Japanese, Korean, and Chinese.

Language Settings
-----------------

PyKokoro automatically detects language from the voice, but you can override:

.. code-block:: python

   with Kokoro() as kokoro:
       # Explicit language
       audio, sr = kokoro.create(
           "Hola, mundo",
           voice="af_nicole",
           lang="es"  # Spanish
       )

       # French
       audio, sr = kokoro.create(
           "Bonjour le monde",
           voice="af_sarah",
           lang="fr"
       )

Supported languages: ``en-us``, ``en-gb``, ``es``, ``fr``, ``de``, ``it``, ``pt``, ``hi``, ``ja``, ``ko``, ``zh``

Speech Speed Control
--------------------

Adjust the speaking rate with the ``speed`` parameter:

.. code-block:: python

   with Kokoro() as kokoro:
       # Slow (0.5x)
       audio, sr = kokoro.create(
           "Slow speech",
           voice="af_bella",
           speed=0.5
       )

       # Normal (1.0x) - default
       audio, sr = kokoro.create(
           "Normal speed",
           voice="af_bella",
           speed=1.0
       )

       # Fast (2.0x)
       audio, sr = kokoro.create(
           "Fast speech",
           voice="af_bella",
           speed=2.0
       )

Recommended range: 0.5 to 2.0

Pause Control
-------------

Add pauses using simple markers in your text:

Pause Markers
~~~~~~~~~~~~~

* ``(.)`` - Short pause (0.3 seconds by default)
* ``(..)`` - Medium pause (0.6 seconds by default)
* ``(...)`` - Long pause (1.0 seconds by default)

.. code-block:: python

   with Kokoro() as kokoro:
       text = """
       Hello! (.) This is a short pause.
       Now a medium pause. (..)
       And finally a long pause. (...)
       Back to normal speech.
       """

       audio, sr = kokoro.create(
           text,
           voice="af_bella",
           enable_pauses=True  # Must enable pause processing
       )

Custom Pause Durations
~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   with Kokoro() as kokoro:
       audio, sr = kokoro.create(
           "Custom pauses (.) here (..) and here (...)",
           voice="af_bella",
           enable_pauses=True,
           pause_short=0.2,    # Short pause (.)
           pause_medium=0.5,   # Medium pause (..)
           pause_long=0.8      # Long pause (...)
       )

Text Splitting
--------------

For long text, use ``split_mode`` to automatically split at natural boundaries:

Split Modes
~~~~~~~~~~~

* ``"paragraph"`` - Split on double newlines
* ``"sentence"`` - Split on sentence boundaries (requires spaCy)
* ``"clause"`` - Split on sentences and commas (requires spaCy)

.. code-block:: python

   with Kokoro() as kokoro:
       long_text = """
       This is the first sentence. This is the second sentence.
       
       This is a new paragraph with more content.
       """

       # Split by sentences
       audio, sr = kokoro.create(
           long_text,
           voice="af_bella",
           split_mode="sentence"
       )

Sentence splitting requires spaCy:

.. code-block:: bash

   pip install spacy
   python -m spacy download en_core_web_sm

Trimming Silence
~~~~~~~~~~~~~~~~

Remove leading/trailing silence from each segment:

.. code-block:: python

   with Kokoro() as kokoro:
       audio, sr = kokoro.create(
           long_text,
           voice="af_bella",
           split_mode="sentence",
           trim_silence=True  # Remove silence between segments
       )

Error Handling
--------------

Handle common errors:

.. code-block:: python

   from pykokoro import Kokoro

   try:
       with Kokoro() as kokoro:
           audio, sr = kokoro.create(
               "Hello!",
               voice="invalid_voice"
           )
   except ValueError as e:
       print(f"Invalid voice: {e}")
   except RuntimeError as e:
       print(f"Runtime error: {e}")

Batch Processing
----------------

Process multiple texts efficiently:

.. code-block:: python

   from pykokoro import Kokoro
   import soundfile as sf

   texts = [
       ("Welcome", "welcome.wav"),
       ("Thank you", "thanks.wav"),
       ("Goodbye", "goodbye.wav"),
   ]

   with Kokoro() as kokoro:
       for text, filename in texts:
           audio, sr = kokoro.create(text, voice="af_bella")
           sf.write(filename, audio, sr)
           print(f"Generated {filename}")

Next Steps
----------

* :doc:`advanced_features` - Voice blending, phoneme control, and more
* :doc:`examples` - Real-world examples
* :doc:`api_reference` - Complete API documentation
