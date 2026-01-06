Changelog
=========

Version 0.2.0 (2025-01-06)
--------------------------

**Breaking Changes:**

* Removed ``PhonemeBook`` class - moved to separate ebook package
* Removed ``PhonemeChapter`` class - moved to separate ebook package
* Removed ``create_phoneme_book_from_chapters()`` function
* Removed ``FORMAT_VERSION`` constant
* Deleted ``examples/phoneme_export.py`` example

**New Features:**

* Added ``split_and_phonemize_text()`` function for standalone text processing
* Added ``enable_pauses`` parameter to ``create()`` method for pause marker support
* Added pause markers: ``(.)``, ``(..)``, ``(...)`` for controlling speech pauses
* Added ``pause_short``, ``pause_medium``, ``pause_long`` parameters for custom pause durations
* Added ``split_mode`` parameter to ``create()`` for intelligent text splitting
* Added ``trim_silence`` parameter for removing silence between segments
* Added ``pause_after`` field to ``PhonemeSegment`` class

**Improvements:**

* Refactored ``_process_with_split_mode()`` to use standalone function
* Improved phoneme-based generation with automatic length checking
* Enhanced documentation with comprehensive examples
* Better error handling and validation
* Optimized text splitting for long passages

**Bug Fixes:**

* Fixed floating point precision in pause duration tests
* Improved backward compatibility for PhonemeSegment serialization
* Better handling of empty and whitespace-only text

**Documentation:**

* Added complete Sphinx documentation
* Added quick start guide
* Added installation guide
* Added basic usage guide
* Added advanced features guide
* Added comprehensive examples
* Added API reference

Version 0.1.0 (Initial Release)
-------------------------------

**Initial Features:**

* Text-to-speech synthesis using Kokoro model
* Support for 70+ voices across multiple languages
* Support for English (US/GB), Spanish, French, German, Italian, Portuguese, Hindi, Japanese, Korean, Chinese
* Voice blending capabilities
* Phoneme-based generation
* GPU acceleration support (CUDA/ROCm)
* Model quality options (fp16, q8, q6)
* Speed control
* Basic tokenizer functionality
* Audio trimming utilities
* Configuration management
* Model and voice downloading
* PhonemeBook and PhonemeChapter classes for document processing
* spaCy integration for sentence splitting
* Mixed language support
