PyKokoro Documentation
======================

PyKokoro is a Python library for text-to-speech synthesis using the Kokoro TTS model.
It provides high-quality, natural-sounding speech generation with support for multiple
languages, voices, and advanced features like pause control and text splitting.

.. toctree::
   :maxdepth: 2
   :caption: Contents:

   quickstart
   installation
   basic_usage
   advanced_features
   api_reference
   examples
   changelog

Features
--------

* **High-Quality TTS**: Natural-sounding speech synthesis using the Kokoro model
* **Multiple Languages**: Support for English (US/GB), Spanish, French, German, Italian, Portuguese, Hindi, Japanese, Korean, and Chinese
* **Voice Selection**: Choose from 54 voices (v1.0) or 103 voices (v1.1-zh) with various styles and accents
* **Voice Blending**: Create custom voice styles by blending multiple voices
* **Pause Control**: Add precise pauses using simple markers: `(.)`, `(..)`, `(...)`
* **Smart Text Splitting**: Automatically split long text at sentence, paragraph, or clause boundaries
* **Phoneme-Based Generation**: Generate speech directly from phonemes for precise control
* **GPU Acceleration**: Optional GPU support for faster generation
* **Flexible Audio Processing**: Trim silence, adjust speed, and more

Quick Example
-------------

.. code-block:: python

   from pykokoro import Kokoro

   # Initialize TTS engine
   kokoro = Kokoro()

   # Generate speech
   audio, sample_rate = kokoro.create(
       "Hello, world! This is a test.",
       voice="af_bella",
       speed=1.0
   )

   # Save to file
   import soundfile as sf
   sf.write("output.wav", audio, sample_rate)

   kokoro.close()

Installation
------------

Install via pip:

.. code-block:: bash

   pip install pykokoro

Or with GPU support:

.. code-block:: bash

   pip install pykokoro[gpu]

Requirements
------------

* Python 3.9 or higher
* NumPy
* ONNX Runtime
* espeak-ng (for phonemization)
* Optional: GPU with CUDA/ROCm for acceleration

Getting Help
------------

* **GitHub Issues**: https://github.com/remixer-dec/pykokoro/issues
* **Documentation**: https://pykokoro.readthedocs.io
* **Examples**: See the ``examples/`` directory in the repository

License
-------

PyKokoro is released under the MIT License. The Kokoro model itself is subject
to its own license terms.

Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
