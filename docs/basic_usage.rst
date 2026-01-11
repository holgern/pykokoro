Basic Usage
===========

This guide covers the fundamental usage patterns of PyKokoro.

Initializing Kokoro
-------------------

The main entry point is the ``Kokoro`` class:

.. code-block:: python

   from pykokoro import Kokoro

   # Initialize with defaults (HuggingFace v1.0, q8 quality)
   kokoro = Kokoro()

   # Or specify model source and variant
   kokoro = Kokoro(model_source="huggingface")  # Default
   kokoro = Kokoro(model_source="huggingface", model_variant="v1.0")  # Default variant
   kokoro = Kokoro(model_source="huggingface", model_variant="v1.1-zh")  # 103 voices
   kokoro = Kokoro(model_source="github", model_variant="v1.0")
   kokoro = Kokoro(model_source="github", model_variant="v1.1-zh")

   # Or specify model quality
   kokoro = Kokoro(model_quality="q8")  # Default: q8

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

Available quality options vary by model source and variant:

**HuggingFace (Default Source):**

Both v1.0 and v1.1-zh variants support:

* ``fp32`` - Full precision (highest quality, largest size)
* ``fp16`` - Half precision (good balance)
* ``q8`` - 8-bit quantized (default, good balance)
* ``q8f16`` - 8-bit with fp16
* ``q4`` - 4-bit quantized (smallest, faster)
* ``q4f16`` - 4-bit with fp16
* ``uint8`` - Unsigned 8-bit
* ``uint8f16`` - Unsigned 8-bit with fp16

**GitHub v1.0:**

* ``fp32`` - Full precision
* ``fp16`` - Half precision
* ``fp16-gpu`` - GPU-optimized fp16
* ``q8`` - 8-bit quantized

**GitHub v1.1-zh:**

* ``fp32`` - Full precision only

.. code-block:: python

   # HuggingFace v1.0 with fp16 (default source)
   kokoro_hq = Kokoro(model_quality="fp16")

   # HuggingFace v1.1-zh with q8 (103 voices)
   kokoro_zh = Kokoro(model_variant="v1.1-zh", model_quality="q8")

   # GitHub v1.0 with GPU optimization
   kokoro_gpu = Kokoro(model_source="github", model_variant="v1.0", model_quality="fp16-gpu")

   # GitHub v1.1-zh (103 voices, fp32 only)
   kokoro_gh_zh = Kokoro(model_source="github", model_variant="v1.1-zh", model_quality="fp32")

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

The number of available voices depends on the model variant:

* **v1.0 (Default)**: 54 multi-language voices
* **v1.1-zh**: 103 voices (including Chinese)

.. code-block:: python

   # Default: HuggingFace v1.0 (54 voices)
   with Kokoro() as kokoro:
       voices = kokoro.list_voices()
       for voice in voices:
           print(voice)

   # Using v1.1-zh for more voices (103 voices from HuggingFace)
   with Kokoro(model_variant="v1.1-zh") as kokoro:
       voices = kokoro.list_voices()
       print(f"Total voices: {len(voices)}")  # 103 voices

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

**GitHub v1.1-zh Additional Voices:**

The v1.1-zh model (available from both HuggingFace and GitHub) includes additional English and Chinese voices:

* **English**: ``af_maple``, ``af_sol``, ``bf_vale``
* **Chinese**: ``zf_001`` through ``zf_099``, ``zm_009`` through ``zm_100``

.. code-block:: python

   # Using v1.1-zh English voices from HuggingFace (default source)
   with Kokoro(model_variant="v1.1-zh") as kokoro:
       audio, sr = kokoro.create(
           "Hello from the v1.1-zh model!",
           voice="af_maple",
           lang="en-us"
       )

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

PyKokoro provides two powerful ways to control pauses in your generated speech:

1. Manual Pause Markers
~~~~~~~~~~~~~~~~~~~~~~~~

Add explicit pauses using simple markers in your text:

**SSMD Break Syntax:**

* ``...c`` - Short/comma pause (0.3 seconds by default)
* ``...s`` - Medium/sentence pause (0.6 seconds by default)
* ``...p`` - Long/paragraph pause (1.0 seconds by default)
* ``...500ms`` - Custom duration pause (e.g., 500 milliseconds)

.. code-block:: python

   with Kokoro() as kokoro:
       text = """
       Hello! ...c This is a short pause.
       Now a medium pause ...s
       And finally a long pause ...p
       Back to normal speech.
       """

       audio, sr = kokoro.create(
           text,
           voice="af_bella"
       )

**Custom Pause Durations:**

You can customize the default pause durations:

.. code-block:: python

   with Kokoro() as kokoro:
       audio, sr = kokoro.create(
           "Custom pauses ...c here ...s and here ...p",
           voice="af_bella",
           pause_clause=0.2,      # Comma pause ...c
           pause_sentence=0.5,    # Sentence pause ...s
           pause_paragraph=0.8    # Paragraph pause ...p
       )

2. Automatic Natural Pauses (NEW!)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

For more natural speech, enable automatic pause insertion at linguistic boundaries:

.. code-block:: python

   with Kokoro() as kokoro:
       text = """
       Artificial intelligence is transforming our world. Machine learning
       models are becoming more sophisticated, efficient, and accessible.

       Deep learning uses neural networks with many layers. These networks
       can learn complex patterns, enabling breakthroughs in vision and
       language processing.
       """

       audio, sr = kokoro.create(
           text,
           voice="af_sarah",
           split_mode="clause",     # Split on commas and sentences
           trim_silence=True,       # Enable automatic pause insertion
           pause_clause=0.25,        # Pause after clauses (commas)
           pause_sentence=0.5,        # Pause after sentences
           pause_paragraph=1.0,          # Pause after paragraphs
           pause_variance=0.05,     # Add natural variance
           random_seed=42           # For reproducibility (optional)
       )

**How It Works:**

When ``trim_silence=True`` and ``split_mode`` is set, PyKokoro automatically:

* Detects linguistic boundaries (clauses, sentences, paragraphs)
* Inserts appropriate pauses based on boundary type:

  - **Clause boundary** (comma within sentence) → ``pause_short``
  - **Sentence boundary** (within paragraph) → ``pause_medium``
  - **Paragraph boundary** → ``pause_long``

* Adds Gaussian variance to prevent robotic timing

**Pause Variance:**

* ``pause_variance=0.0`` - No variance (exact pauses)
* ``pause_variance=0.05`` - Default (±100ms at 95% confidence)
* ``pause_variance=0.1`` - More variation (±200ms at 95% confidence)

**Combining Both Approaches:**

Use manual markers for special emphasis and automatic pauses for natural rhythm:

.. code-block:: python

   with Kokoro() as kokoro:
       text = "Welcome! ...p Let's discuss AI ...c deep learning ...c and robotics."

       audio, sr = kokoro.create(
           text,
           voice="af_sarah",
           split_mode="clause",     # Automatic pauses at commas
           trim_silence=True,
           pause_variance=0.05
       )

Text Splitting
--------------

For long text, use ``split_mode`` to automatically split at natural boundaries.

This improves prosody and handles phoneme length limits automatically.

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

Automatic Pauses with Text Splitting
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

When combining ``split_mode`` with ``trim_silence=True``, PyKokoro automatically
adds natural pauses between segments:

.. code-block:: python

   with Kokoro() as kokoro:
       audio, sr = kokoro.create(
           long_text,
           voice="af_bella",
           split_mode="clause",      # Best for natural pauses
           trim_silence=True,        # Enable automatic pauses
           pause_short=0.25,         # Clause pauses
           pause_medium=0.5,         # Sentence pauses
           pause_long=1.0,           # Paragraph pauses
           pause_variance=0.05       # Natural variance
       )

This creates more natural-sounding speech with appropriate pauses at linguistic
boundaries without requiring manual pause markers.

Text Normalization (Say-As)
----------------------------

PyKokoro supports automatic text normalization using SSMD (Speech Synthesis Markdown)
syntax. This feature converts numbers, dates, phone numbers, and other special formats
into speakable text.

Basic Usage
~~~~~~~~~~~

Use the ``[text](as: type)`` syntax to specify how text should be normalized:

.. code-block:: python

   with Kokoro() as kokoro:
       # Cardinal numbers
       text = "I have [123](as: cardinal) apples"
       audio, sr = kokoro.create(text, voice="af_sarah")
       # TTS says: "I have one hundred twenty-three apples"

       # Ordinal numbers
       text = "I came in [3](as: ordinal) place"
       audio, sr = kokoro.create(text, voice="af_sarah")
       # TTS says: "I came in third place"

       # Digits (spell out)
       text = "My PIN is [1234](as: digits)"
       audio, sr = kokoro.create(text, voice="af_sarah")
       # TTS says: "My PIN is one two three four"

Supported Say-As Types
~~~~~~~~~~~~~~~~~~~~~~~

**Numbers:**

* ``cardinal`` - Numbers as cardinals: "123" → "one hundred twenty-three"
* ``ordinal`` - Numbers as ordinals: "3" → "third"
* ``digits`` - Spell out digits: "123" → "one two three"
* ``number`` - Alias for cardinal
* ``fraction`` - Fractions: "1/2" → "one half"

**Text:**

* ``characters`` - Spell out text: "ABC" → "A B C"
* ``expletive`` - Censors to "beep"

**Date and Time:**

* ``date`` - Dates with format support (mdy, dmy, ymd, ym, my, md, dm, d, m, y)
* ``time`` - Time in 12h or 24h format

**Other:**

* ``telephone`` - Phone numbers: "+1-555-0123" → "plus one five five five oh one two three"
* ``unit`` - Units: "5kg" → "five kilograms"

Examples
~~~~~~~~

.. code-block:: python

   with Kokoro() as kokoro:
       # Telephone numbers
       text = "Call [+1-555-0123](as: telephone)"
       audio, sr = kokoro.create(text, voice="af_sarah")
       # TTS says: "Call plus one five five five oh one two three"

       # Dates with custom formatting
       text = "Today is [12/31/2024](as: date, format: mdy)"
       audio, sr = kokoro.create(text, voice="af_sarah")
       # TTS says: "Today is December thirty-first, two thousand twenty-four"

       # Time (12-hour or 24-hour)
       text = "The time is [14:30](as: time)"
       audio, sr = kokoro.create(text, voice="af_sarah")
       # TTS says: "The time is two thirty PM"

       # Characters (spell out)
       text = "The code is [ABC](as: characters)"
       audio, sr = kokoro.create(text, voice="af_sarah")
       # TTS says: "The code is A B C"

       # Fractions
       text = "Add [1/2](as: fraction) cup of sugar"
       audio, sr = kokoro.create(text, voice="af_sarah")
       # TTS says: "Add one half cup of sugar"

       # Units
       text = "The package weighs [5kg](as: unit)"
       audio, sr = kokoro.create(text, voice="af_sarah")
       # TTS says: "The package weighs five kilograms"

Multi-language Support
~~~~~~~~~~~~~~~~~~~~~~

Say-as works with multiple languages (English, French, German, Spanish, and more):

.. code-block:: python

   with Kokoro() as kokoro:
       # French cardinal
       text = "[123](as: cardinal)"
       audio, sr = kokoro.create(text, voice="ff_siwis", lang="fr-fr")
       # TTS says: "cent vingt-trois"

       # German ordinal
       text = "[3](as: ordinal)"
       audio, sr = kokoro.create(text, voice="gf_maria", lang="de-de")
       # TTS says: "dritte"

Combining with Other Features
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Say-as works seamlessly with all SSMD features:

.. code-block:: python

   with Kokoro() as kokoro:
       # With prosody
       text = "[100](as: cardinal) +loud+ dollars!"
       audio, sr = kokoro.create(text, voice="af_sarah")

       # With pauses
       text = "[First](as: ordinal) ...s [second](as: ordinal) ...s [third](as: ordinal)!"
       audio, sr = kokoro.create(text, voice="af_sarah")

       # With emphasis
       text = "The winner is *[1](as: ordinal)*!"
       audio, sr = kokoro.create(text, voice="af_sarah")

Trimming Silence Only
~~~~~~~~~~~~~~~~~~~~~~

To remove silence without adding automatic pauses, use ``trim_silence=True``
without ``split_mode``:

.. code-block:: python

   with Kokoro() as kokoro:
       audio, sr = kokoro.create(
           text,
           voice="af_bella",
           trim_silence=True  # Just trim, no automatic pauses
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
