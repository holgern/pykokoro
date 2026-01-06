Advanced Features
=================

This guide covers advanced features of PyKokoro for power users.

Voice Blending
--------------

Create custom voices by blending multiple voices together.

Basic Voice Blending
~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   from pykokoro import Kokoro, VoiceBlend

   with Kokoro() as kokoro:
       # Blend two voices equally
       blend = VoiceBlend.parse("af_bella + af_sarah")
       
       audio, sr = kokoro.create(
           "This is a blended voice",
           voice=blend
       )

Weighted Blending
~~~~~~~~~~~~~~~~~

Control the contribution of each voice:

.. code-block:: python

   from pykokoro import Kokoro, VoiceBlend

   with Kokoro() as kokoro:
       # 70% bella, 30% sarah
       blend = VoiceBlend.parse("af_bella*0.7 + af_sarah*0.3")
       
       audio, sr = kokoro.create(
           "Weighted blend",
           voice=blend
       )

       # Percentage notation (normalized automatically)
       blend2 = VoiceBlend.parse("af_bella*70% + af_sarah*30%")

Multiple Voice Blending
~~~~~~~~~~~~~~~~~~~~~~~~

Blend more than two voices:

.. code-block:: python

   from pykokoro import Kokoro, VoiceBlend

   with Kokoro() as kokoro:
       # Three-way blend
       blend = VoiceBlend.parse(
           "af_bella*0.5 + af_sarah*0.3 + af_nicole*0.2"
       )
       
       audio, sr = kokoro.create(
           "Complex blend",
           voice=blend
       )

Creating Blended Voice Programmatically
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   from pykokoro import Kokoro, VoiceBlend

   # Create blend object directly
   blend = VoiceBlend(
       voices=["af_bella", "af_sarah", "am_adam"],
       weights=[0.4, 0.4, 0.2]
   )

   with Kokoro() as kokoro:
       audio, sr = kokoro.create(
           "Custom blend",
           voice=blend
       )

Phoneme-Based Generation
-------------------------

For precise control, generate speech directly from phonemes.

Using create_from_phonemes()
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   from pykokoro import Kokoro

   with Kokoro() as kokoro:
       # Get phonemes for text
       phonemes = kokoro.tokenizer.phonemize("Hello, world!")
       
       # Generate from phonemes
       audio, sr = kokoro.create_from_phonemes(
           phonemes,
           voice="af_bella",
           speed=1.0
       )

Text to Phonemes
~~~~~~~~~~~~~~~~

Convert text to phonemes:

.. code-block:: python

   from pykokoro import Kokoro

   with Kokoro() as kokoro:
       # Get phonemes
       phonemes = kokoro.tokenizer.phonemize(
           "Hello, world!",
           lang="en-us"
       )
       print(f"Phonemes: {phonemes}")
       
       # Get detailed phoneme info
       result = kokoro.tokenizer.text_to_phonemes(
           "Hello",
           lang="en-us",
           with_words=True
       )
       print(result)

PhonemeSegment Processing
~~~~~~~~~~~~~~~~~~~~~~~~~~

Work with phoneme segments for batch processing:

.. code-block:: python

   from pykokoro import phonemize_text_list, create_tokenizer

   tokenizer = create_tokenizer()
   
   texts = ["Hello", "World", "How are you?"]
   segments = phonemize_text_list(texts, tokenizer, lang="en-us")
   
   for segment in segments:
       print(f"Text: {segment.text}")
       print(f"Phonemes: {segment.phonemes}")
       print(f"Tokens: {segment.tokens}")

Advanced Text Splitting
------------------------

Split and Phonemize in One Step
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

For advanced text processing with splitting and phoneme generation:

.. code-block:: python

   from pykokoro import split_and_phonemize_text, create_tokenizer

   tokenizer = create_tokenizer()
   
   long_text = """
   This is the first sentence. This is the second.
   
   This is a new paragraph.
   """
   
   segments = split_and_phonemize_text(
       long_text,
       tokenizer,
       lang="en-us",
       split_mode="sentence",
       max_chars=300,
       max_phoneme_length=510
   )
   
   for segment in segments:
       print(f"Paragraph {segment.paragraph}, Sentence {segment.sentence}")
       print(f"Text: {segment.text}")
       print(f"Phonemes: {segment.phonemes[:50]}...")

Split Modes in Detail
~~~~~~~~~~~~~~~~~~~~~

**Paragraph Mode:**

.. code-block:: python

   segments = split_and_phonemize_text(
       text,
       tokenizer,
       split_mode="paragraph"  # Splits on double newlines
   )

**Sentence Mode:**

Requires spaCy for sentence boundary detection:

.. code-block:: python

   segments = split_and_phonemize_text(
       text,
       tokenizer,
       split_mode="sentence"  # Splits on sentence boundaries
   )

**Clause Mode:**

Splits on both sentences and commas for finer control:

.. code-block:: python

   segments = split_and_phonemize_text(
       text,
       tokenizer,
       split_mode="clause"  # Splits on sentences and commas
   )

Custom Warning Callbacks
~~~~~~~~~~~~~~~~~~~~~~~~~

Handle warnings during phoneme generation:

.. code-block:: python

   from pykokoro import split_and_phonemize_text, create_tokenizer

   def my_warning_handler(message):
       print(f"WARNING: {message}")

   tokenizer = create_tokenizer()
   segments = split_and_phonemize_text(
       very_long_text,
       tokenizer,
       warn_callback=my_warning_handler
   )

GPU Acceleration
----------------

Automatic GPU Detection
~~~~~~~~~~~~~~~~~~~~~~~

PyKokoro automatically uses GPU if available:

.. code-block:: python

   from pykokoro import Kokoro, get_device

   # Check available device
   device = get_device()
   print(f"Using device: {device}")

   # Kokoro will use GPU automatically
   with Kokoro() as kokoro:
       audio, sr = kokoro.create("Hello!", voice="af_bella")

Forcing Specific Device
~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   from pykokoro import Kokoro

   # Force CPU
   kokoro_cpu = Kokoro(device="cpu")

   # Force CUDA (NVIDIA)
   kokoro_gpu = Kokoro(device="cuda")

   # Force ROCm (AMD)
   kokoro_rocm = Kokoro(device="rocm")

GPU Information
~~~~~~~~~~~~~~~

.. code-block:: python

   from pykokoro import get_gpu_info

   info = get_gpu_info()
   print(f"Device: {info['device']}")
   print(f"Providers: {info['providers']}")

Custom Model Paths
------------------

Use Custom Model Files
~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   from pykokoro import Kokoro

   kokoro = Kokoro(
       model_path="/path/to/custom/model.onnx",
       voices_path="/path/to/voices.bin"
   )

Download Models Manually
~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   from pykokoro import download_model, download_voice, download_all_models

   # Download specific model quality
   download_model(quality="q8")

   # Download specific voice
   download_voice("af_bella")

   # Download all models
   download_all_models()

Get Model Paths
~~~~~~~~~~~~~~~

.. code-block:: python

   from pykokoro import get_model_path, get_voice_path

   model_path = get_model_path(quality="q8")
   voice_path = get_voice_path()
   
   print(f"Model: {model_path}")
   print(f"Voices: {voice_path}")

Advanced Tokenizer Configuration
---------------------------------

Custom Tokenizer Settings
~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   from pykokoro import create_tokenizer, TokenizerConfig, EspeakConfig

   # Custom espeak settings
   espeak_config = EspeakConfig(
       backend="espeak-ng",
       voice="en-us"
   )

   # Custom tokenizer config
   tokenizer_config = TokenizerConfig(
       vocab_path="/path/to/vocab.txt",
       espeak_config=espeak_config
   )

   tokenizer = create_tokenizer(config=tokenizer_config)

Mixed Language Support
~~~~~~~~~~~~~~~~~~~~~~

For text with multiple languages:

.. code-block:: python

   from pykokoro import create_tokenizer, TokenizerConfig

   config = TokenizerConfig(
       enable_mixed_language=True,
       primary_language="en-us",
       allowed_languages=["en-us", "es", "fr"],
       language_confidence_threshold=0.7
   )

   tokenizer = create_tokenizer(config=config)

Audio Trimming
--------------

Trim Silence from Audio
~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   from pykokoro import trim

   # Generate audio with silence
   with Kokoro() as kokoro:
       audio, sr = kokoro.create("Hello!", voice="af_bella")
   
   # Trim silence
   trimmed_audio, trim_info = trim(audio)
   
   print(f"Original: {len(audio)} samples")
   print(f"Trimmed: {len(trimmed_audio)} samples")
   print(f"Trim info: {trim_info}")

Configuration Management
------------------------

Save and Load Configuration
~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   from pykokoro import save_config, load_config

   # Save configuration
   config = {
       "default_voice": "af_bella",
       "default_speed": 1.0,
       "model_quality": "q8"
   }
   save_config(config, "my_config.json")

   # Load configuration
   loaded_config = load_config("my_config.json")

Get Cache Paths
~~~~~~~~~~~~~~~

.. code-block:: python

   from pykokoro import get_user_cache_path, get_user_config_path

   cache_path = get_user_cache_path()
   config_path = get_user_config_path()
   
   print(f"Cache: {cache_path}")
   print(f"Config: {config_path}")

Performance Tips
----------------

1. **Reuse Kokoro Instance**

   Don't create a new ``Kokoro()`` for each request - initialize once and reuse.

2. **Use GPU When Available**

   GPU acceleration provides 3-10x speedup.

3. **Batch Processing**

   Process multiple texts in one session to avoid initialization overhead.

4. **Choose Appropriate Model Quality**

   Use ``q6`` or ``q8`` for production; ``fp16`` only when quality is critical.

5. **Use split_mode for Long Text**

   Splitting long text improves quality and reduces memory usage.

Next Steps
----------

* :doc:`examples` - Real-world usage examples
* :doc:`api_reference` - Complete API documentation
