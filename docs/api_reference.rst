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

**Key Methods:**

* ``create()`` - Main text-to-speech method with support for:
  
  - Manual pause markers (``enable_pauses=True``)
  - Automatic natural pauses (``split_mode`` + ``trim_silence=True``)
  - Pause variance control (``pause_variance``, ``random_seed``)
  
* ``create_from_phonemes()`` - Generate from IPA phonemes
* ``create_from_tokens()`` - Generate from token IDs
* ``create_stream()`` - Async streaming generation
* ``create_stream_sync()`` - Sync streaming generation

VoiceBlend
~~~~~~~~~~

.. autoclass:: pykokoro.VoiceBlend
   :members:
   :undoc-members:
   :show-inheritance:

ModelQuality
~~~~~~~~~~~~

.. autoclass:: pykokoro.ModelQuality
   :members:
   :undoc-members:
   :show-inheritance:

Tokenizer
---------

.. autoclass:: pykokoro.Tokenizer
   :members:
   :undoc-members:
   :show-inheritance:

.. autofunction:: pykokoro.create_tokenizer

Configuration Classes
~~~~~~~~~~~~~~~~~~~~~

.. autoclass:: pykokoro.TokenizerConfig
   :members:
   :undoc-members:
   :show-inheritance:

.. autoclass:: pykokoro.EspeakConfig
   :members:
   :undoc-members:
   :show-inheritance:

.. autoclass:: pykokoro.PhonemeResult
   :members:
   :undoc-members:
   :show-inheritance:

Phoneme Classes
---------------

.. autoclass:: pykokoro.PhonemeSegment
   :members:
   :undoc-members:
   :show-inheritance:

.. autofunction:: pykokoro.phonemize_text_list

.. autofunction:: pykokoro.split_and_phonemize_text

Model and Voice Management
---------------------------

Download Functions
~~~~~~~~~~~~~~~~~~

.. autofunction:: pykokoro.download_model

.. autofunction:: pykokoro.download_voice

.. autofunction:: pykokoro.download_all_models

.. autofunction:: pykokoro.download_all_voices

.. autofunction:: pykokoro.download_config

Path Functions
~~~~~~~~~~~~~~

.. autofunction:: pykokoro.get_model_path

.. autofunction:: pykokoro.get_voice_path

Utility Functions
-----------------

Configuration
~~~~~~~~~~~~~

.. autofunction:: pykokoro.load_config

.. autofunction:: pykokoro.save_config

.. autofunction:: pykokoro.get_user_cache_path

.. autofunction:: pykokoro.get_user_config_path

Device Management
~~~~~~~~~~~~~~~~~

.. autofunction:: pykokoro.get_device

.. autofunction:: pykokoro.get_gpu_info

Audio Processing
~~~~~~~~~~~~~~~~

.. autofunction:: pykokoro.trim

Constants
---------

.. autodata:: pykokoro.PROGRAM_NAME
   :annotation:

.. autodata:: pykokoro.DEFAULT_CONFIG
   :annotation:
