Changelog
=========

Version 0.3.1 (TBD)
-------------------

**New Features:**

* Added support for HuggingFace Kokoro v1.1-zh model (``onnx-community/Kokoro-82M-v1.1-zh-ONNX``)
* New model variant ``v1.1-zh-hf`` with 103 voices and full quantization support
* Added ``download_model_hf_v11zh()`` for downloading v1.1-zh models with quantization
* Added ``download_voices_hf_v11zh()`` for downloading all 103 voices
* Added ``download_all_models_hf_v11zh()`` for complete v1.1-zh setup

**Improvements:**

* All quantization levels supported for v1.1-zh: fp32, fp16, q8, q8f16, q4, q4f16, uint8, uint8f16
* Voice files automatically combined into efficient .npz format
* Progress callbacks for voice downloads

**Documentation:**

* Added examples and documentation for HuggingFace v1.1-zh usage
* Updated advanced features guide with v1.1-zh-hf variant
* Added ``examples/hf_v11zh_demo.py`` demonstration script

Version 0.3.0 (2026-01-07)
--------------------------

**Major Refactoring:**

* Extracted internal manager classes for better code organization
* Reduced codebase complexity by ~706 lines (12% reduction)
* Improved maintainability with better separation of concerns
* 100% backward compatibility maintained - no breaking changes

**New Internal Classes:**

* Added ``OnnxSessionManager`` class for ONNX Runtime session management
* Added ``VoiceManager`` class for voice loading and blending operations
* Added ``AudioGenerator`` class for audio generation pipeline
* Added ``MixedLanguageHandler`` class for automatic language detection
* Added ``PhonemeDictionary`` class for custom word-to-phoneme mappings

**Code Quality:**

* Reduced ``onnx_backend.py`` by 436 lines
* Reduced ``tokenizer.py`` by 270 lines
* Added comprehensive test coverage for new manager classes
* All pre-commit hooks passing (ruff, ruff-format)
* 98.7% test pass rate (312/316 tests)

**Architecture Improvements:**

* Delegate pattern implementation for backward compatibility
* Better separation of session management, voice handling, and audio generation
* Improved modularity for easier testing and maintenance
* Enhanced error handling and validation

**Documentation:**

* Added API documentation for new internal manager classes
* Added internal architecture section to advanced features guide
* Updated changelog with refactoring details

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
