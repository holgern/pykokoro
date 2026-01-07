API Reference
=============

This page provides detailed API documentation for all public classes and functions in PyKokoro.

Main Classes
------------

Kokoro
~~~~~~

.. autoclass:: pykokoro.Kokoro
   :members:
   :undoc-members:
   :show-inheritance:

The ``Kokoro`` class is the main entry point for text-to-speech generation.

**Initialization Examples:**

.. code-block:: python

   from pykokoro import Kokoro

   # Default: HuggingFace v1.0, q8 quality, CPU
   kokoro = Kokoro()

   # High quality with GPU
   kokoro = Kokoro(model_quality="fp16", device="cuda")

   # Use v1.1-zh variant for 103 voices
   kokoro = Kokoro(model_variant="v1.1-zh", model_quality="q8")

   # GitHub source with GPU-optimized model
   kokoro = Kokoro(
       model_source="github",
       model_variant="v1.0",
       model_quality="fp16-gpu"
   )

   # Custom cache directory
   kokoro = Kokoro(cache_dir="/custom/path")

   # Context manager (recommended)
   with Kokoro() as kokoro:
       audio, sr = kokoro.create("Hello!", voice="af_bella")

**Key Methods:**

* ``create()`` - Main text-to-speech method with support for:

  - Manual pause markers (``enable_pauses=True``)
  - Automatic natural pauses (``split_mode`` + ``trim_silence=True``)
  - Pause variance control (``pause_variance``, ``random_seed``)

* ``create_from_phonemes()`` - Generate from IPA phonemes
* ``create_from_tokens()`` - Generate from token IDs
* ``create_stream()`` - Async streaming generation
* ``create_stream_sync()`` - Sync streaming generation

**create() Method Examples:**

.. code-block:: python

   # Basic usage
   audio, sr = kokoro.create("Hello, world!", voice="af_bella")

   # With speed control
   audio, sr = kokoro.create(
       "Fast speech",
       voice="af_sarah",
       speed=1.5  # 1.5x speed
   )

   # With manual pause markers
   text = "First part. (...) Second part after long pause."
   audio, sr = kokoro.create(
       text,
       voice="am_adam",
       enable_pauses=True,
       pause_long=1.5  # 1.5 second pause for (...)
   )

   # With automatic natural pauses
   long_text = """
   Artificial intelligence is transforming our world. Machine learning
   models are becoming more sophisticated, efficient, and accessible.

   Deep learning uses neural networks with many layers.
   """
   audio, sr = kokoro.create(
       long_text,
       voice="af_bella",
       split_mode="clause",     # Split on commas and sentences
       trim_silence=True,       # Enable automatic pauses
       pause_clause=0.25,       # 250ms after commas
       pause_sentence=0.5,      # 500ms after sentences
       pause_paragraph=1.0,     # 1s after paragraphs
       pause_variance=0.05,     # Natural variance
       random_seed=42           # Reproducible
   )

   # With language override
   audio, sr = kokoro.create(
       "Bonjour le monde",
       voice="af_sarah",
       lang="fr"
   )

   # Trim silence only
   audio, sr = kokoro.create(
       "Clean audio",
       voice="bf_emma",
       trim_silence=True
   )

**create_from_phonemes() Examples:**

.. code-block:: python

   from pykokoro import Kokoro

   with Kokoro() as kokoro:
       # Get phonemes for text
       phonemes = kokoro.tokenizer.phonemize("Hello world", lang="en-us")
       print(phonemes)  # həˈloʊ wɜːld

       # Generate from phonemes
       audio, sr = kokoro.create_from_phonemes(
           phonemes,
           voice="af_bella",
           speed=1.0
       )

       # Custom phonemes for precise control
       custom_phonemes = "həˈloʊ ˈwɜːld"
       audio, sr = kokoro.create_from_phonemes(
           custom_phonemes,
           voice="am_adam"
       )

**create_stream_sync() Examples:**

.. code-block:: python

   from pykokoro import Kokoro
   import soundfile as sf

   with Kokoro() as kokoro:
       long_text = """
       This is a long text that will be streamed.
       Each sentence will be generated and returned separately.
       This allows for lower latency in applications.
       """

       # Synchronous streaming
       chunks = []
       for audio_chunk, sample_rate, text_segment in kokoro.create_stream_sync(
           long_text,
           voice="af_sarah",
           split_mode="sentence"
       ):
           print(f"Generated: {text_segment}")
           chunks.append(audio_chunk)
           # You can play or process each chunk immediately

       # Combine all chunks
       import numpy as np
       full_audio = np.concatenate(chunks)
       sf.write("streamed_output.wav", full_audio, sample_rate)

**create_stream() Async Examples:**

.. code-block:: python

   import asyncio
   from pykokoro import Kokoro

   async def stream_speech():
       with Kokoro() as kokoro:
           text = "Async streaming allows concurrent operations."

           async for audio_chunk, sr, segment in kokoro.create_stream(
               text,
               voice="bf_emma",
               split_mode="sentence"
           ):
               print(f"Chunk: {segment}")
               # Process chunk asynchronously
               await process_audio_async(audio_chunk, sr)

   asyncio.run(stream_speech())

**Voice Management Examples:**

.. code-block:: python

   with Kokoro() as kokoro:
       # List all available voices
       voices = kokoro.list_voices()
       print(f"Available voices: {len(voices)}")
       for voice in voices[:5]:
           print(voice)

       # Get specific voice embedding
       voice_embedding = kokoro.get_voice("af_bella")
       print(voice_embedding.shape)  # (1, 1, 256)

**Utility Method Examples:**

.. code-block:: python

   with Kokoro() as kokoro:
       # Split text into chunks
       chunks = kokoro.split_text("Long text here...", max_length=100)

       # Check if model is downloaded
       is_ready = kokoro.is_model_ready()

       # Get model info
       info = kokoro.get_model_info()
       print(f"Model: {info['variant']}, Quality: {info['quality']}")

**Advanced: Phoneme Segment Generation:**

.. code-block:: python

   from pykokoro import Kokoro, PhonemeSegment

   with Kokoro() as kokoro:
       # Create phoneme segments with metadata
       segments = [
           PhonemeSegment(phonemes="həˈloʊ", lang="en-us", pause_after=0.3),
           PhonemeSegment(phonemes="wɜːld", lang="en-us", pause_after=0.0),
       ]

       # Generate from segments
       audio, sr = kokoro.create_from_segments(segments, voice="af_bella")

VoiceBlend
~~~~~~~~~~

.. autoclass:: pykokoro.VoiceBlend
   :members:
   :undoc-members:
   :show-inheritance:

The ``VoiceBlend`` class allows you to create custom voices by blending multiple voices together.

**Basic Examples:**

.. code-block:: python

   from pykokoro import Kokoro, VoiceBlend

   with Kokoro() as kokoro:
       # Equal blend of two voices (50% each)
       blend = VoiceBlend.parse("af_bella + af_sarah")
       audio, sr = kokoro.create("Blended voice", voice=blend)

       # Weighted blend (70% bella, 30% sarah)
       blend = VoiceBlend.parse("af_bella*0.7 + af_sarah*0.3")
       audio, sr = kokoro.create("Weighted blend", voice=blend)

       # Percentage notation (normalized automatically)
       blend = VoiceBlend.parse("af_bella*70% + af_sarah*30%")
       audio, sr = kokoro.create("Percentage blend", voice=blend)

**Multiple Voice Blending:**

.. code-block:: python

   from pykokoro import Kokoro, VoiceBlend

   with Kokoro() as kokoro:
       # Three-way blend
       blend = VoiceBlend.parse("af_bella*50% + af_sarah*30% + am_adam*20%")
       audio, sr = kokoro.create("Three voices mixed", voice=blend)

       # Complex blend with four voices
       blend = VoiceBlend.parse(
           "af_bella*0.4 + af_sarah*0.3 + bf_emma*0.2 + am_michael*0.1"
       )
       audio, sr = kokoro.create("Complex blend", voice=blend)

**Programmatic Blend Creation:**

.. code-block:: python

   from pykokoro import Kokoro, VoiceBlend

   with Kokoro() as kokoro:
       # Create blend from lists
       voices = ["af_bella", "af_sarah", "am_adam"]
       weights = [0.5, 0.3, 0.2]
       blend = VoiceBlend(voices=voices, weights=weights)

       audio, sr = kokoro.create("Programmatic blend", voice=blend)

       # Dynamic weight calculation
       import numpy as np
       num_voices = 5
       voices = ["af_bella", "af_sarah", "bf_emma", "am_adam", "am_michael"]
       weights = np.random.dirichlet(np.ones(num_voices))  # Random normalized weights
       blend = VoiceBlend(voices=voices, weights=weights.tolist())

**Accessing Blend Properties:**

.. code-block:: python

   from pykokoro import VoiceBlend

   blend = VoiceBlend.parse("af_bella*60% + af_sarah*40%")

   print(blend.voices)   # ['af_bella', 'af_sarah']
   print(blend.weights)  # [0.6, 0.4]
   print(str(blend))     # 'af_bella*0.6 + af_sarah*0.4'

**Gender and Accent Blending:**

.. code-block:: python

   from pykokoro import Kokoro, VoiceBlend

   with Kokoro() as kokoro:
       # Mix American and British accents
       blend = VoiceBlend.parse("af_bella*50% + bf_emma*50%")
       audio, sr = kokoro.create("Trans-Atlantic accent", voice=blend)

       # Mix male and female voices for androgynous effect
       blend = VoiceBlend.parse("af_sarah*50% + am_adam*50%")
       audio, sr = kokoro.create("Androgynous voice", voice=blend)

**Error Handling:**

.. code-block:: python

   from pykokoro import VoiceBlend

   try:
       # Invalid syntax
       blend = VoiceBlend.parse("invalid syntax here")
   except ValueError as e:
       print(f"Parse error: {e}")

   try:
       # Weights don't sum to 1 (will be auto-normalized)
       blend = VoiceBlend(voices=["af_bella"], weights=[0.5])
       # Weights are normalized: [1.0]
   except ValueError as e:
       print(f"Weight error: {e}")

ModelQuality
~~~~~~~~~~~~

.. autoclass:: pykokoro.ModelQuality
   :members:
   :undoc-members:
   :show-inheritance:

The ``ModelQuality`` class represents available model quality options.

**Available Qualities:**

.. code-block:: python

   from pykokoro import ModelQuality, Kokoro

   # HuggingFace qualities (both v1.0 and v1.1-zh)
   ModelQuality.FP32      # Full precision (highest quality, largest)
   ModelQuality.FP16      # Half precision (good balance)
   ModelQuality.Q8        # 8-bit quantized (default, recommended)
   ModelQuality.Q8F16     # 8-bit with fp16
   ModelQuality.Q4        # 4-bit quantized (smallest, fastest)
   ModelQuality.Q4F16     # 4-bit with fp16
   ModelQuality.UINT8     # Unsigned 8-bit
   ModelQuality.UINT8F16  # Unsigned 8-bit with fp16

   # GitHub v1.0 additional quality
   ModelQuality.FP16_GPU  # GPU-optimized fp16 (GitHub v1.0 only)

**Usage Examples:**

.. code-block:: python

   from pykokoro import Kokoro, ModelQuality

   # Using enum
   kokoro = Kokoro(model_quality=ModelQuality.FP16)

   # Using string (recommended)
   kokoro = Kokoro(model_quality="fp16")

   # Quality comparison
   qualities = ["fp32", "fp16", "q8", "q4"]
   for quality in qualities:
       kokoro = Kokoro(model_quality=quality)
       # Benchmark each quality...

**Quality Recommendations:**

.. code-block:: python

   from pykokoro import Kokoro

   # For production (best balance)
   kokoro = Kokoro(model_quality="q8")  # Default, recommended

   # For highest quality (larger size)
   kokoro = Kokoro(model_quality="fp16")

   # For fastest inference (some quality loss)
   kokoro = Kokoro(model_quality="q4")

   # For GPU inference (GitHub v1.0 only)
   kokoro = Kokoro(
       model_source="github",
       model_variant="v1.0",
       model_quality="fp16-gpu"
   )

Tokenizer
---------

.. autoclass:: pykokoro.Tokenizer
   :members:
   :undoc-members:
   :show-inheritance:

The ``Tokenizer`` class handles text-to-phoneme conversion and tokenization.

**Basic Tokenization:**

.. code-block:: python

   from pykokoro import Tokenizer

   tokenizer = Tokenizer()

   # Convert text to phonemes
   phonemes = tokenizer.phonemize("Hello, world!", lang="en-us")
   print(phonemes)  # həˈloʊ, wɜːld!

   # Convert phonemes to token IDs
   tokens = tokenizer.encode(phonemes)
   print(tokens)  # [12, 45, 78, ...]

   # Decode tokens back to phonemes
   decoded = tokenizer.decode(tokens)
   print(decoded)  # həˈloʊ, wɜːld!

**Language-Specific Phonemization:**

.. code-block:: python

   from pykokoro import Tokenizer

   tokenizer = Tokenizer()

   # English (US)
   phonemes_us = tokenizer.phonemize("Hello", lang="en-us")

   # English (GB)
   phonemes_gb = tokenizer.phonemize("Hello", lang="en-gb")

   # Spanish
   phonemes_es = tokenizer.phonemize("Hola", lang="es")

   # French
   phonemes_fr = tokenizer.phonemize("Bonjour", lang="fr")

   # German
   phonemes_de = tokenizer.phonemize("Hallo", lang="de")

   # Japanese
   phonemes_ja = tokenizer.phonemize("こんにちは", lang="ja")

**Batch Processing:**

.. code-block:: python

   from pykokoro import Tokenizer

   tokenizer = Tokenizer()

   # Phonemize multiple texts
   texts = [
       "First sentence.",
       "Second sentence.",
       "Third sentence."
   ]

   phoneme_list = [
       tokenizer.phonemize(text, lang="en-us")
       for text in texts
   ]

**Custom Configuration:**

.. code-block:: python

   from pykokoro import Tokenizer, TokenizerConfig

   # Create custom config
   config = TokenizerConfig(
       preserve_punctuation=True,
       use_mixed_language=False
   )

   tokenizer = Tokenizer(config=config)
   phonemes = tokenizer.phonemize("Hello, world!")

**Advanced: Direct Token Generation:**

.. code-block:: python

   from pykokoro import Tokenizer, Kokoro

   tokenizer = Tokenizer()

   # Get phonemes
   phonemes = tokenizer.phonemize("Hello", lang="en-us")

   # Convert to tokens
   tokens = tokenizer.encode(phonemes)

   # Use tokens directly with Kokoro
   with Kokoro() as kokoro:
       audio, sr = kokoro.create_from_tokens(
           tokens,
           voice="af_bella"
       )

.. autofunction:: pykokoro.create_tokenizer

**create_tokenizer() Examples:**

.. code-block:: python

   from pykokoro import create_tokenizer

   # Create tokenizer with variant
   tokenizer_v1_0 = create_tokenizer(variant="v1.0")
   tokenizer_v1_1_zh = create_tokenizer(variant="v1.1-zh")

   # Use different vocabularies
   phonemes_v1_0 = tokenizer_v1_0.phonemize("Hello", lang="en-us")
   phonemes_v1_1_zh = tokenizer_v1_1_zh.phonemize("Hello", lang="en-us")

Configuration Classes
~~~~~~~~~~~~~~~~~~~~~

.. autoclass:: pykokoro.TokenizerConfig
   :members:
   :undoc-members:
   :show-inheritance:

**TokenizerConfig Examples:**

.. code-block:: python

   from pykokoro import TokenizerConfig, Tokenizer

   # Basic configuration
   config = TokenizerConfig(
       preserve_punctuation=True,
       use_mixed_language=False
   )
   tokenizer = Tokenizer(config=config)

   # Mixed language configuration
   config = TokenizerConfig(
       use_mixed_language=True,
       mixed_language_primary="en-us",
       mixed_language_allowed=["en-us", "de", "fr", "es"]
   )
   tokenizer = Tokenizer(config=config)

   # Now can handle mixed text
   phonemes = tokenizer.phonemize(
       "Hello, Bonjour, Hola, Guten Tag!",
       lang="en-us"
   )

   # Custom phoneme dictionary
   config = TokenizerConfig(
       phoneme_dictionary_path="/path/to/custom_dict.json"
   )
   tokenizer = Tokenizer(config=config)

   # Preserve all punctuation
   config = TokenizerConfig(preserve_punctuation=True)
   tokenizer = Tokenizer(config=config)
   phonemes = tokenizer.phonemize("Hello, world! How are you?")

.. autoclass:: pykokoro.EspeakConfig
   :members:
   :undoc-members:
   :show-inheritance:

**EspeakConfig Examples:**

.. code-block:: python

   from pykokoro import EspeakConfig

   # Custom espeak configuration
   config = EspeakConfig(
       voice="en-us",
       speed=1.0,
       pitch=50,
       volume=100
   )

   # Adjust speaking parameters
   config = EspeakConfig(
       voice="en-gb",
       speed=1.2,   # Faster speech for phonemization
       pitch=60     # Higher pitch
   )

.. autoclass:: pykokoro.PhonemeResult
   :members:
   :undoc-members:
   :show-inheritance:

**PhonemeResult Examples:**

.. code-block:: python

   from pykokoro import Tokenizer

   tokenizer = Tokenizer()

   # Get phoneme result with metadata
   result = tokenizer.phonemize_detailed("Hello", lang="en-us")

   print(result.phonemes)      # The phoneme string
   print(result.language)      # Language used
   print(result.is_truncated)  # Whether text was truncated

Phoneme Classes
---------------

.. autoclass:: pykokoro.PhonemeSegment
   :members:
   :undoc-members:
   :show-inheritance:

**PhonemeSegment Examples:**

.. code-block:: python

   from pykokoro import PhonemeSegment, Kokoro

   # Create phoneme segment
   segment = PhonemeSegment(
       phonemes="həˈloʊ",
       lang="en-us",
       pause_after=0.5  # 500ms pause after this segment
   )

   # Create multiple segments
   segments = [
       PhonemeSegment(phonemes="həˈloʊ", lang="en-us", pause_after=0.3),
       PhonemeSegment(phonemes="wɜːld", lang="en-us", pause_after=0.0),
   ]

   # Use with Kokoro
   with Kokoro() as kokoro:
       audio, sr = kokoro.create_from_segments(
           segments,
           voice="af_bella"
       )

   # Serialize to dict
   segment_dict = segment.to_dict()
   print(segment_dict)
   # {'phonemes': 'həˈloʊ', 'lang': 'en-us', 'pause_after': 0.5}

   # Deserialize from dict
   restored = PhonemeSegment.from_dict(segment_dict)

   # Format for display
   print(segment.format_readable())
   # həˈloʊ [en-us] (pause: 0.5s)

**Advanced Segment Manipulation:**

.. code-block:: python

   from pykokoro import PhonemeSegment
   import json

   # Create segments with metadata
   segments = [
       PhonemeSegment(
           phonemes="ðɪs ɪz ə",
           lang="en-us",
           pause_after=0.2,
           metadata={"paragraph": 1, "sentence": 1}
       ),
       PhonemeSegment(
           phonemes="ˈsɛntəns",
           lang="en-us",
           pause_after=0.5,
           metadata={"paragraph": 1, "sentence": 1}
       ),
   ]

   # Save to JSON
   segments_json = json.dumps([s.to_dict() for s in segments], indent=2)

   # Load from JSON
   loaded_dicts = json.loads(segments_json)
   restored_segments = [PhonemeSegment.from_dict(d) for d in loaded_dicts]

.. autofunction:: pykokoro.phonemize_text_list

**phonemize_text_list() Examples:**

.. code-block:: python

   from pykokoro import phonemize_text_list

   # Phonemize list of texts
   texts = ["Hello", "World", "Goodbye"]
   phoneme_list = phonemize_text_list(texts, lang="en-us")

   for text, phonemes in zip(texts, phoneme_list):
       print(f"{text} -> {phonemes}")
   # Hello -> həˈloʊ
   # World -> wɜːld
   # Goodbye -> ɡʊdˈbaɪ

   # With different languages
   mixed_texts = [
       ("Hello", "en-us"),
       ("Bonjour", "fr"),
       ("Hola", "es"),
   ]

   phonemes = []
   for text, lang in mixed_texts:
       result = phonemize_text_list([text], lang=lang)
       phonemes.append(result[0])

.. autofunction:: pykokoro.split_and_phonemize_text

**split_and_phonemize_text() Examples:**

.. code-block:: python

   from pykokoro import split_and_phonemize_text

   # Basic splitting and phonemization
   text = "This is sentence one. This is sentence two."
   segments = split_and_phonemize_text(
       text,
       lang="en-us",
       split_mode="sentence",
       max_phoneme_length=510
   )

   for segment in segments:
       print(segment.format_readable())

   # With clause splitting
   text = "AI is transforming our world, making technology more accessible."
   segments = split_and_phonemize_text(
       text,
       lang="en-us",
       split_mode="clause",  # Split on commas too
       max_phoneme_length=510
   )

   # Paragraph splitting
   text = """
   First paragraph here.

   Second paragraph here.
   """
   segments = split_and_phonemize_text(
       text,
       lang="en-us",
       split_mode="paragraph"
   )

   # With warning callback
   def on_truncation(original_text, truncated_text):
       print(f"Warning: Text truncated from {len(original_text)} to {len(truncated_text)}")

   segments = split_and_phonemize_text(
       "Very long text here" * 100,
       lang="en-us",
       split_mode="sentence",
       warning_callback=on_truncation
   )

Internal Manager Classes
-------------------------

These classes are used internally by Kokoro but are available for advanced use cases.

OnnxSessionManager
~~~~~~~~~~~~~~~~~~

.. autoclass:: pykokoro.OnnxSessionManager
   :members:
   :undoc-members:
   :show-inheritance:

Manages ONNX Runtime session creation with automatic provider selection (CUDA, ROCm, CPU).

**Advanced Usage Examples:**

.. code-block:: python

   from pykokoro.onnx_session import OnnxSessionManager
   from pathlib import Path

   # Create session manager
   manager = OnnxSessionManager()

   # Create session with automatic provider selection
   model_path = Path("~/.cache/pykokoro/models/huggingface/v1.0/onnx/model_quantized.onnx")
   session = manager.create_session(
       model_path.expanduser(),
       providers=["CUDAExecutionProvider", "CPUExecutionProvider"]
   )

   # Get active provider
   provider = manager.get_active_provider()
   print(f"Using provider: {provider}")

   # Clean up
   manager.close()

VoiceManager
~~~~~~~~~~~~

.. autoclass:: pykokoro.VoiceManager
   :members:
   :undoc-members:
   :show-inheritance:

Handles voice loading and blending operations.

**Advanced Usage Examples:**

.. code-block:: python

   from pykokoro.voice_manager import VoiceManager
   from pykokoro import VoiceBlend

   # Create voice manager
   manager = VoiceManager(
       source="huggingface",
       variant="v1.0"
   )

   # Load single voice
   voice_embedding = manager.get_voice("af_bella")
   print(voice_embedding.shape)  # (1, 1, 256)

   # Load blended voice
   blend = VoiceBlend.parse("af_bella*0.6 + af_sarah*0.4")
   blended_embedding = manager.get_voice(blend)

   # List available voices
   voices = manager.list_voices()
   print(f"Available voices: {len(voices)}")

   # Clean up
   manager.close()

AudioGenerator
~~~~~~~~~~~~~~

.. autoclass:: pykokoro.AudioGenerator
   :members:
   :undoc-members:
   :show-inheritance:

Manages audio generation from phonemes and tokens.

**Advanced Usage Examples:**

.. code-block:: python

   from pykokoro.audio_generator import AudioGenerator
   from pykokoro import Kokoro

   # Create audio generator (typically used internally)
   with Kokoro() as kokoro:
       generator = kokoro._audio_generator

       # Access generation parameters
       print(f"Sample rate: {generator.sample_rate}")
       print(f"Max phoneme length: {generator.max_phoneme_length}")

MixedLanguageHandler
~~~~~~~~~~~~~~~~~~~~

.. autoclass:: pykokoro.MixedLanguageHandler
   :members:
   :undoc-members:
   :show-inheritance:

Handles automatic language detection and mixed-language text processing.

**Advanced Usage Examples:**

.. code-block:: python

   from pykokoro.mixed_language import MixedLanguageHandler

   # Create handler with allowed languages
   handler = MixedLanguageHandler(
       primary_lang="en-us",
       allowed_langs=["en-us", "fr", "de", "es"]
   )

   # Detect language
   text = "Bonjour le monde"
   detected = handler.detect_language(text)
   print(f"Detected: {detected}")

   # Process mixed language text
   mixed = "Hello, Bonjour, Hola"
   result = handler.process(mixed)

PhonemeDictionary
~~~~~~~~~~~~~~~~~

.. autoclass:: pykokoro.PhonemeDictionary
   :members:
   :undoc-members:
   :show-inheritance:

Manages custom word-to-phoneme mappings for pronunciation control.

**Advanced Usage Examples:**

.. code-block:: python

   from pykokoro.phoneme_dictionary import PhonemeDictionary
   import json

   # Create empty dictionary
   phoneme_dict = PhonemeDictionary()

   # Add custom pronunciations
   phoneme_dict.add("PyKokoro", "paɪ kəˈkɔːɹoʊ")
   phoneme_dict.add("ONNX", "ɑːn ɛks")
   phoneme_dict.add("API", "eɪ piː aɪ")

   # Lookup pronunciation
   pronunciation = phoneme_dict.lookup("PyKokoro")
   print(pronunciation)  # paɪ kəˈkɔːɹoʊ

   # Check if word exists
   if phoneme_dict.has("ONNX"):
       print(f"ONNX pronunciation: {phoneme_dict.lookup('ONNX')}")

   # Save to file
   phoneme_dict.save("custom_dict.json")

   # Load from file
   loaded_dict = PhonemeDictionary.load("custom_dict.json")

   # Use with Tokenizer
   from pykokoro import Tokenizer, TokenizerConfig

   config = TokenizerConfig(
       phoneme_dictionary_path="custom_dict.json"
   )
   tokenizer = Tokenizer(config=config)

   # Now "PyKokoro" will use custom pronunciation
   phonemes = tokenizer.phonemize("Welcome to PyKokoro!")

   # Export/Import as dict
   dict_data = phoneme_dict.to_dict()
   new_dict = PhonemeDictionary.from_dict(dict_data)

Model and Voice Management
---------------------------

Download Functions
~~~~~~~~~~~~~~~~~~

.. autofunction:: pykokoro.download_model

**download_model() Examples:**

.. code-block:: python

   from pykokoro import download_model

   # Download HuggingFace v1.0 model (default)
   path = download_model(
       source="huggingface",
       variant="v1.0",
       quality="q8"
   )
   print(f"Downloaded to: {path}")

   # Download HuggingFace v1.1-zh with high quality
   path = download_model(
       source="huggingface",
       variant="v1.1-zh",
       quality="fp16"
   )

   # Download GitHub v1.0 with GPU optimization
   path = download_model(
       source="github",
       variant="v1.0",
       quality="fp16-gpu"
   )

   # Download GitHub v1.1-zh
   path = download_model(
       source="github",
       variant="v1.1-zh",
       quality="fp32"  # Only fp32 available for GitHub v1.1-zh
   )

   # Force re-download
   path = download_model(
       source="huggingface",
       variant="v1.0",
       quality="q8",
       force=True  # Re-download even if exists
   )

.. autofunction:: pykokoro.download_voice

**download_voice() Examples:**

.. code-block:: python

   from pykokoro import download_voice

   # Download single voice from HuggingFace v1.0
   path = download_voice(
       voice_name="af_bella",
       source="huggingface",
       variant="v1.0"
   )

   # Download v1.1-zh voice
   path = download_voice(
       voice_name="af_maple",  # v1.1-zh English voice
       source="huggingface",
       variant="v1.1-zh"
   )

   # Download from GitHub
   path = download_voice(
       voice_name="af_sarah",
       source="github",
       variant="v1.0"
   )

   # Force re-download
   path = download_voice(
       voice_name="am_adam",
       source="huggingface",
       variant="v1.0",
       force=True
   )

.. autofunction:: pykokoro.download_all_models

**download_all_models() Examples:**

.. code-block:: python

   from pykokoro import download_all_models

   # Download all HuggingFace v1.0 models
   paths = download_all_models(
       source="huggingface",
       variant="v1.0"
   )
   print(f"Downloaded {len(paths)} models")
   # Downloads: fp32, fp16, q8, q8f16, q4, q4f16, uint8, uint8f16

   # Download all GitHub v1.0 models
   paths = download_all_models(
       source="github",
       variant="v1.0"
   )
   # Downloads: fp32, fp16, fp16-gpu, q8

   # With progress callback
   def progress(model_name, current, total):
       print(f"Downloading {model_name} ({current}/{total})")

   paths = download_all_models(
       source="huggingface",
       variant="v1.1-zh",
       progress_callback=progress
   )

.. autofunction:: pykokoro.download_all_voices

**download_all_voices() Examples:**

.. code-block:: python

   from pykokoro import download_all_voices

   # Download all HuggingFace v1.0 voices (54 voices)
   path = download_all_voices(
       source="huggingface",
       variant="v1.0"
   )
   print(f"Voices saved to: {path}")

   # Download all v1.1-zh voices (103 voices)
   path = download_all_voices(
       source="huggingface",
       variant="v1.1-zh"
   )

   # With progress callback
   def progress(voice_name, current, total):
       percent = (current / total) * 100
       print(f"Downloading {voice_name} ({current}/{total}) - {percent:.1f}%")

   path = download_all_voices(
       source="huggingface",
       variant="v1.0",
       progress_callback=progress
   )

   # Force re-download all voices
   path = download_all_voices(
       source="github",
       variant="v1.0",
       force=True
   )

.. autofunction:: pykokoro.download_config

**download_config() Examples:**

.. code-block:: python

   from pykokoro import download_config

   # Download v1.0 config
   path = download_config(variant="v1.0")
   print(f"Config saved to: {path}")

   # Download v1.1-zh config
   path = download_config(variant="v1.1-zh")

   # Force re-download
   path = download_config(variant="v1.0", force=True)

Path Functions
~~~~~~~~~~~~~~

.. autofunction:: pykokoro.get_model_path

**get_model_path() Examples:**

.. code-block:: python

   from pykokoro import get_model_path
   from pathlib import Path

   # Get HuggingFace v1.0 model path
   path = get_model_path(
       source="huggingface",
       variant="v1.0",
       quality="q8"
   )
   print(path)
   # ~/.cache/pykokoro/models/huggingface/v1.0/onnx/model_quantized.onnx

   # Check if model exists
   if path.exists():
       print(f"Model size: {path.stat().st_size / (1024**2):.1f} MB")

   # Get GitHub v1.0 GPU model path
   path = get_model_path(
       source="github",
       variant="v1.0",
       quality="fp16-gpu"
   )

   # Get v1.1-zh model path
   path = get_model_path(
       source="huggingface",
       variant="v1.1-zh",
       quality="fp16"
   )

.. autofunction:: pykokoro.get_voice_path

**get_voice_path() Examples:**

.. code-block:: python

   from pykokoro import get_voice_path

   # Get HuggingFace voice path
   path = get_voice_path(
       voice_name="af_bella",
       source="huggingface",
       variant="v1.0"
   )
   print(path)
   # ~/.cache/pykokoro/voices/huggingface/v1.0/af_bella.bin

   # Check if voice exists
   if path.exists():
       print("Voice is downloaded")

   # Get v1.1-zh voice path
   path = get_voice_path(
       voice_name="af_maple",
       source="huggingface",
       variant="v1.1-zh"
   )

   # Get GitHub voice path
   path = get_voice_path(
       voice_name="am_adam",
       source="github",
       variant="v1.0"
   )

Utility Functions
-----------------

Configuration
~~~~~~~~~~~~~

.. autofunction:: pykokoro.load_config

**load_config() Examples:**

.. code-block:: python

   from pykokoro import load_config

   # Load current configuration
   config = load_config()
   print(config)
   # {
   #     'model_quality': 'q8',
   #     'use_gpu': False,
   #     'vocab_version': 'v1.0',
   #     'model_source': 'huggingface',
   #     'model_variant': 'v1.0'
   # }

   # Access specific settings
   model_quality = config.get('model_quality', 'q8')
   use_gpu = config.get('use_gpu', False)

   # Check if config exists
   if config:
       print("Configuration loaded successfully")

.. autofunction:: pykokoro.save_config

**save_config() Examples:**

.. code-block:: python

   from pykokoro import save_config, load_config

   # Modify and save configuration
   config = load_config()
   config['model_quality'] = 'fp16'
   config['use_gpu'] = True
   config['model_source'] = 'huggingface'
   config['model_variant'] = 'v1.1-zh'

   success = save_config(config)
   if success:
       print("Configuration saved successfully")

   # Create custom configuration
   custom_config = {
       'model_quality': 'q8',
       'use_gpu': True,
       'vocab_version': 'v1.1-zh',
       'model_source': 'github',
       'model_variant': 'v1.0',
       'custom_setting': 'my_value'
   }
   save_config(custom_config)

   # Reset to defaults
   default_config = {
       'model_quality': 'q8',
       'use_gpu': False,
       'vocab_version': 'v1.0',
       'model_source': 'huggingface',
       'model_variant': 'v1.0'
   }
   save_config(default_config)

.. autofunction:: pykokoro.get_user_cache_path

**get_user_cache_path() Examples:**

.. code-block:: python

   from pykokoro import get_user_cache_path

   # Get base cache directory
   cache_dir = get_user_cache_path()
   print(cache_dir)
   # Linux: ~/.cache/pykokoro/
   # macOS: ~/Library/Caches/pykokoro/
   # Windows: %LOCALAPPDATA%\pykokoro\cache\

   # Get cache subdirectory
   models_cache = get_user_cache_path("models")
   voices_cache = get_user_cache_path("voices")
   temp_cache = get_user_cache_path("temp")

   # Create cache directory if it doesn't exist
   cache_dir.mkdir(parents=True, exist_ok=True)

   # Check cache size
   import os
   total_size = 0
   for dirpath, dirnames, filenames in os.walk(cache_dir):
       for filename in filenames:
           filepath = os.path.join(dirpath, filename)
           total_size += os.path.getsize(filepath)

   print(f"Cache size: {total_size / (1024**3):.2f} GB")

   # Clear specific cache folder
   import shutil
   temp_dir = get_user_cache_path("temp")
   if temp_dir.exists():
       shutil.rmtree(temp_dir)
       print("Temp cache cleared")

.. autofunction:: pykokoro.get_user_config_path

**get_user_config_path() Examples:**

.. code-block:: python

   from pykokoro import get_user_config_path

   # Get config file path
   config_path = get_user_config_path()
   print(config_path)
   # Linux: ~/.config/pykokoro/config.json
   # macOS: ~/Library/Application Support/pykokoro/config.json
   # Windows: %APPDATA%\pykokoro\config.json

   # Check if config exists
   if config_path.exists():
       print("Config file exists")
       print(f"Config size: {config_path.stat().st_size} bytes")

   # Read config directly
   import json
   if config_path.exists():
       with open(config_path, 'r') as f:
           config = json.load(f)
           print(json.dumps(config, indent=2))

   # Backup config
   import shutil
   if config_path.exists():
       backup_path = config_path.with_suffix('.json.backup')
       shutil.copy(config_path, backup_path)
       print(f"Config backed up to: {backup_path}")

Device Management
~~~~~~~~~~~~~~~~~

.. autofunction:: pykokoro.get_device

**get_device() Examples:**

.. code-block:: python

   from pykokoro import get_device

   # Get default device (CPU)
   device = get_device(use_gpu=False)
   print(device)  # 'CPUExecutionProvider'

   # Try to get GPU device
   device = get_device(use_gpu=True)
   print(device)
   # 'CUDAExecutionProvider' (NVIDIA GPU)
   # 'ROCMExecutionProvider' (AMD GPU)
   # 'DmlExecutionProvider' (DirectML on Windows)
   # 'CoreMLExecutionProvider' (Apple Silicon)
   # 'CPUExecutionProvider' (fallback if no GPU)

   # Use with Kokoro
   from pykokoro import Kokoro

   device = get_device(use_gpu=True)
   if 'CUDA' in device:
       print("Using NVIDIA GPU")
       kokoro = Kokoro(device="cuda")
   elif 'ROCM' in device:
       print("Using AMD GPU")
       kokoro = Kokoro(device="rocm")
   else:
       print("Using CPU")
       kokoro = Kokoro(device="cpu")

.. autofunction:: pykokoro.get_gpu_info

**get_gpu_info() Examples:**

.. code-block:: python

   from pykokoro import get_gpu_info

   # Check GPU availability
   message, is_available = get_gpu_info(enabled=True)
   print(message)
   print(f"GPU available: {is_available}")

   # When GPU is disabled
   message, is_available = get_gpu_info(enabled=False)
   print(message)  # "GPU disabled by user"

   # Use in application
   from pykokoro import Kokoro, get_gpu_info

   gpu_msg, gpu_available = get_gpu_info(enabled=True)
   if gpu_available:
       print(f"Initializing with GPU: {gpu_msg}")
       kokoro = Kokoro(device="cuda")
   else:
       print(f"Initializing with CPU: {gpu_msg}")
       kokoro = Kokoro(device="cpu")

   # System diagnostics
   def check_system():
       gpu_msg, gpu_available = get_gpu_info(enabled=True)

       print("System Information:")
       print(f"  GPU Status: {gpu_msg}")
       print(f"  GPU Available: {gpu_available}")

       if gpu_available:
           print("  Recommendation: Use GPU for faster inference")
       else:
           print("  Recommendation: CPU mode (consider GPU for better performance)")

   check_system()

Audio Processing
~~~~~~~~~~~~~~~~

.. autofunction:: pykokoro.trim

**trim() Examples:**

.. code-block:: python

   from pykokoro import Kokoro, trim
   import soundfile as sf

   with Kokoro() as kokoro:
       # Generate audio
       audio, sr = kokoro.create("Hello, world!", voice="af_bella")

       # Trim silence from audio
       trimmed_audio = trim(audio)

       # Save trimmed audio
       sf.write("trimmed.wav", trimmed_audio, sr)

       print(f"Original length: {len(audio)} samples")
       print(f"Trimmed length: {len(trimmed_audio)} samples")

   # Trim with custom threshold
   from pykokoro import Kokoro, trim
   import numpy as np

   with Kokoro() as kokoro:
       audio, sr = kokoro.create("Test audio", voice="af_sarah")

       # More aggressive trimming (higher threshold)
       trimmed = trim(audio, threshold=0.01)

       # Less aggressive trimming (lower threshold)
       gentle_trimmed = trim(audio, threshold=0.001)

   # Batch trimming
   from pykokoro import Kokoro, trim

   with Kokoro() as kokoro:
       texts = ["First", "Second", "Third"]

       for i, text in enumerate(texts):
           audio, sr = kokoro.create(text, voice="am_adam")
           trimmed = trim(audio)
           sf.write(f"output_{i}.wav", trimmed, sr)

   # Compare trimmed vs untrimmed
   import matplotlib.pyplot as plt
   from pykokoro import Kokoro, trim

   with Kokoro() as kokoro:
       audio, sr = kokoro.create("Hello!", voice="bf_emma")
       trimmed = trim(audio)

       # Plot comparison
       plt.figure(figsize=(12, 4))

       plt.subplot(2, 1, 1)
       plt.plot(audio)
       plt.title(f"Original ({len(audio)} samples)")

       plt.subplot(2, 1, 2)
       plt.plot(trimmed)
       plt.title(f"Trimmed ({len(trimmed)} samples)")

       plt.tight_layout()
       plt.savefig("trim_comparison.png")

Constants
---------

.. autodata:: pykokoro.PROGRAM_NAME
   :annotation:

**PROGRAM_NAME Examples:**

.. code-block:: python

   from pykokoro import PROGRAM_NAME

   print(PROGRAM_NAME)  # 'pykokoro'

   # Use in logging
   import logging
   logger = logging.getLogger(PROGRAM_NAME)
   logger.info("PyKokoro initialized")

   # Use in cache paths
   from pathlib import Path
   cache_dir = Path.home() / ".cache" / PROGRAM_NAME
   cache_dir.mkdir(parents=True, exist_ok=True)

.. autodata:: pykokoro.DEFAULT_CONFIG
   :annotation:

**DEFAULT_CONFIG Examples:**

.. code-block:: python

   from pykokoro import DEFAULT_CONFIG
   import json

   print(json.dumps(DEFAULT_CONFIG, indent=2))
   # {
   #   "model_quality": "q8",
   #   "use_gpu": false,
   #   "vocab_version": "v1.0",
   #   "model_source": "huggingface",
   #   "model_variant": "v1.0"
   # }

   # Use as template for custom config
   custom_config = DEFAULT_CONFIG.copy()
   custom_config['model_quality'] = 'fp16'
   custom_config['use_gpu'] = True

   # Reset config to defaults
   from pykokoro import save_config
   save_config(DEFAULT_CONFIG)

Complete Usage Examples
------------------------

Basic Text-to-Speech
~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   from pykokoro import Kokoro
   import soundfile as sf

   # Simple TTS
   with Kokoro() as kokoro:
       audio, sr = kokoro.create("Hello, world!", voice="af_bella")
       sf.write("output.wav", audio, sr)

Multi-Language Speech
~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   from pykokoro import Kokoro
   import soundfile as sf

   languages = [
       ("Hello, world!", "en-us", "af_bella"),
       ("Bonjour le monde!", "fr", "ff_siwis"),
       ("Hola, mundo!", "es", "ef_dora"),
       ("Hallo, Welt!", "de", "gf_vera"),
   ]

   with Kokoro() as kokoro:
       for i, (text, lang, voice) in enumerate(languages):
           audio, sr = kokoro.create(text, voice=voice, lang=lang)
           sf.write(f"output_{lang}.wav", audio, sr)
           print(f"Generated {lang}: {text}")

Streaming for Low Latency
~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   from pykokoro import Kokoro
   import soundfile as sf
   import numpy as np

   long_text = """
   This is a demonstration of streaming text-to-speech.
   Each sentence is generated independently.
   This allows for lower latency in real-time applications.
   You can start playing audio before all text is generated.
   """

   with Kokoro() as kokoro:
       chunks = []
       for audio_chunk, sr, segment in kokoro.create_stream_sync(
           long_text,
           voice="af_sarah",
           split_mode="sentence"
       ):
           print(f"Generated: {segment}")
           chunks.append(audio_chunk)
           # Play chunk immediately for lowest latency

       # Save combined result
       full_audio = np.concatenate(chunks)
       sf.write("streamed.wav", full_audio, sr)

Voice Blending and Customization
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   from pykokoro import Kokoro, VoiceBlend
   import soundfile as sf

   with Kokoro() as kokoro:
       # Create unique blended voices
       blends = [
           ("af_bella*50% + af_sarah*50%", "blend_female_mix"),
           ("af_bella*70% + am_adam*30%", "blend_feminine"),
           ("am_adam*70% + af_bella*30%", "blend_masculine"),
           ("bf_emma*50% + af_bella*50%", "blend_transatlantic"),
       ]

       text = "This is a custom blended voice."

       for blend_str, name in blends:
           blend = VoiceBlend.parse(blend_str)
           audio, sr = kokoro.create(text, voice=blend)
           sf.write(f"{name}.wav", audio, sr)
           print(f"Created: {name}")

Batch Processing with Progress
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   from pykokoro import Kokoro
   import soundfile as sf
   from pathlib import Path

   # Read texts from file
   texts = Path("texts.txt").read_text().strip().split("\n")

   with Kokoro() as kokoro:
       output_dir = Path("output")
       output_dir.mkdir(exist_ok=True)

       for i, text in enumerate(texts, 1):
           print(f"Processing {i}/{len(texts)}: {text[:50]}...")

           audio, sr = kokoro.create(
               text,
               voice="af_bella",
               split_mode="sentence",
               trim_silence=True
           )

           output_path = output_dir / f"audio_{i:03d}.wav"
           sf.write(output_path, audio, sr)

       print(f"Generated {len(texts)} audio files")

Custom Phoneme Generation
~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   from pykokoro import Kokoro, Tokenizer
   import soundfile as sf

   with Kokoro() as kokoro:
       tokenizer = Tokenizer()

       # Get phonemes for custom pronunciation
       text = "PyKokoro is awesome!"
       phonemes = tokenizer.phonemize(text, lang="en-us")

       # Modify phonemes for effect
       modified_phonemes = phonemes.replace("ɔː", "oʊ")

       # Generate from modified phonemes
       audio, sr = kokoro.create_from_phonemes(
           modified_phonemes,
           voice="am_adam"
       )

       sf.write("custom_pronunciation.wav", audio, sr)

Natural Pauses and Prosody
~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   from pykokoro import Kokoro
   import soundfile as sf

   narration = """
   In the year 2045, artificial intelligence had transformed society.
   Cities were clean, efficient, and sustainable. Transportation was
   seamless, powered by autonomous vehicles and advanced logistics.

   But not everyone was happy. Some longed for the old ways, the chaos
   and unpredictability of human-driven systems. They formed communities
   on the outskirts, living as their ancestors had.

   The future, it seemed, was not one-size-fits-all.
   """

   with Kokoro() as kokoro:
       audio, sr = kokoro.create(
           narration,
           voice="am_michael",
           split_mode="clause",      # Natural pauses at commas
           trim_silence=True,
           pause_clause=0.3,         # 300ms after clauses
           pause_sentence=0.6,       # 600ms after sentences
           pause_paragraph=1.2,      # 1.2s between paragraphs
           pause_variance=0.08,      # Natural variation
           speed=0.95,               # Slightly slower for narration
           random_seed=42            # Reproducible
       )

       sf.write("narration.wav", audio, sr)
       print(f"Generated narration: {len(audio)/sr:.1f} seconds")

Download Management
~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   from pykokoro import (
       download_model,
       download_all_voices,
       get_model_path,
       Kokoro
   )

   # Pre-download models and voices
   print("Downloading models...")

   # Download specific model
   download_model(
       source="huggingface",
       variant="v1.0",
       quality="fp16"
   )

   # Download all voices
   print("Downloading voices...")
   download_all_voices(
       source="huggingface",
       variant="v1.0",
       progress_callback=lambda v, c, t: print(f"  {v} ({c}/{t})")
   )

   print("Downloads complete!")

   # Now use offline
   with Kokoro(model_quality="fp16") as kokoro:
       audio, sr = kokoro.create(
           "Ready for offline use!",
           voice="af_bella"
       )

See Also
--------

* :doc:`basic_usage` - Fundamental usage patterns
* :doc:`advanced_features` - Advanced features and techniques
* :doc:`examples` - Real-world example scripts
* :doc:`installation` - Installation and setup guide
