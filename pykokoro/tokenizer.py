"""Tokenizer for pykokoro - converts text to phonemes and tokens.

This module provides text-to-phoneme and phoneme-to-token conversion using
kokorog2p (dictionary + espeak fallback) as the phonemizer backend.
"""

from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING

from kokorog2p import (
    N_TOKENS,
    GToken,
    filter_for_kokoro,
    get_g2p,
    get_kokoro_vocab,
    ids_to_phonemes,
    phonemes_to_ids,
    phonemize_with_markdown,
    validate_for_kokoro,
)
from kokorog2p.base import G2PBase
from kokorog2p.mixed_language_g2p import MixedLanguageG2P

if TYPE_CHECKING:
    pass

logger = logging.getLogger(__name__)

# Constants
MAX_PHONEME_LENGTH = 510
SAMPLE_RATE = 24000

# Supported languages for phonemization
# Format: language code -> kokorog2p language code
# All languages now fully supported by kokorog2p with dictionary + espeak fallback
SUPPORTED_LANGUAGES = {
    "en-us": "en-us",
    "en-gb": "en-gb",
    "en": "en-us",  # Default English to US
    "es": "es",
    "fr-fr": "fr-fr",
    "fr": "fr-fr",  # Accept both fr and fr-fr
    "de": "de",
    "it": "it",
    "pt": "pt",
    "pl": "pl",
    "tr": "tr",
    "ru": "ru",
    "ko": "ko",
    "ja": "ja",
    "zh": "zh",  # Mandarin Chinese
    "cmn": "cmn",  # Accept both zh and cmn
}


@dataclass
class TokenizerConfig:
    """Configuration for the tokenizer.

    Attributes:
        use_espeak_fallback: Whether to use espeak for OOV words (default: True)
        use_spacy: Whether to use spaCy for POS tagging (default: True)
        use_dictionary: Whether to use dictionary lookup (default: True)
        use_mixed_language: Enable automatic language detection for mixed-language
            text (default: False). Requires mixed_language_allowed to be set.
        mixed_language_primary: Primary language code for mixed-language mode
            (e.g., 'de', 'en-us'). If None, uses the language passed to phonemize().
        mixed_language_allowed: List of language codes to detect and support in
            mixed-language mode (e.g., ['de', 'en-us', 'fr']). Required when
            use_mixed_language is True.
        mixed_language_confidence: Minimum confidence threshold (0.0-1.0) for
            accepting language detection results. Words below this threshold
            fall back to primary_language (default: 0.7).
        phoneme_dictionary_path: Path to custom phoneme dictionary JSON file.
            Format: {"word": "/phoneme/"} where phonemes are in IPA format.
        phoneme_dict_case_sensitive: Whether phoneme dictionary matching should
            be case-sensitive (default: False).
    """

    use_espeak_fallback: bool = True
    use_spacy: bool = True
    use_dictionary: bool = True
    use_mixed_language: bool = False
    mixed_language_primary: str | None = None
    mixed_language_allowed: list[str] | None = None
    mixed_language_confidence: float = 0.7
    phoneme_dictionary_path: str | None = None
    phoneme_dict_case_sensitive: bool = False


# Backward compatibility alias
@dataclass
class EspeakConfig:
    """Configuration for espeak-ng backend (deprecated, use TokenizerConfig).

    Kept for backward compatibility. The lib_path and data_path are now
    managed by kokorog2p internally.

    Attributes:
        lib_path: Path to the espeak-ng shared library (ignored)
        data_path: Path to the espeak-ng data directory (ignored)
    """

    lib_path: str | None = None
    data_path: str | None = None


@dataclass
class PhonemeResult:
    """Result of phonemization with quality metadata.

    Attributes:
        phonemes: The phoneme string
        tokens: List of GToken objects with per-word phonemes
        low_confidence_words: Words that used espeak fallback
    """

    phonemes: str
    tokens: list[GToken] = field(default_factory=list)
    low_confidence_words: list[str] = field(default_factory=list)


class Tokenizer:
    """Text-to-phoneme-to-token converter using kokorog2p.

    This class handles:
    1. Text normalization
    2. Text to phoneme conversion (via kokorog2p dictionary + espeak fallback)
    3. Phoneme to token conversion (via Kokoro vocabulary)
    4. Token to phoneme conversion (reverse lookup)
    5. Optional mixed-language support for automatic language detection

    Mixed-Language Support:
        Enable automatic language detection for text containing multiple languages
        by setting TokenizerConfig.use_mixed_language=True and specifying
        allowed_languages. This uses kokorog2p's MixedLanguageG2P backend.

    Args:
        espeak_config: Deprecated, kept for backward compatibility
        vocab_version: Ignored (uses kokorog2p's embedded vocabulary)
        vocab: Optional custom vocabulary dict (overrides default)
        config: Optional TokenizerConfig for phonemization settings

    Example:
        >>> # Single-language usage (default)
        >>> tokenizer = Tokenizer()
        >>> phonemes = tokenizer.phonemize("Hello world")
        >>> tokens = tokenizer.tokenize(phonemes)

        >>> # Mixed-language usage
        >>> config = TokenizerConfig(
        ...     use_mixed_language=True,
        ...     mixed_language_primary="de",
        ...     mixed_language_allowed=["de", "en-us"]
        ... )
        >>> tokenizer = Tokenizer(config=config)
        >>> phonemes = tokenizer.phonemize("Ich gehe zum Meeting")
    """

    def __init__(
        self,
        espeak_config: EspeakConfig | None = None,
        vocab_version: str = "v1.0",
        vocab: dict[str, int] | None = None,
        config: TokenizerConfig | None = None,
    ):
        """Initialize the tokenizer.

        Args:
            espeak_config: Deprecated, kept for backward compatibility
            vocab_version: Model variant/version (e.g., 'v1.0', 'v1.1-zh') for filtering
            vocab: Optional custom vocabulary (overrides default)
            config: Optional TokenizerConfig for phonemization settings
        """
        self.vocab_version = vocab_version
        # Determine kokorog2p model parameter from vocab_version
        # Map variant names to kokorog2p model names
        if vocab_version == "v1.1-zh":
            self._kokorog2p_model = "1.1"
        else:
            self._kokorog2p_model = "1.0"

        self.vocab = (
            vocab
            if vocab is not None
            else get_kokoro_vocab(model=self._kokorog2p_model)
        )
        self._reverse_vocab: dict[int, str] | None = None
        self.config = config or TokenizerConfig()

        # G2P instances cache (lazy loaded per language)
        self._g2p_cache: dict[str, G2PBase] = {}

        # Phoneme dictionary cache (lazy loaded)
        self._phoneme_dictionary: dict[str, str] | None = None

        # Log if espeak_config was provided (deprecated)
        if espeak_config is not None and (
            espeak_config.lib_path or espeak_config.data_path
        ):
            logger.warning(
                "EspeakConfig is deprecated. kokorog2p manages espeak internally."
            )

        # Load phoneme dictionary if specified
        if self.config.phoneme_dictionary_path:
            try:
                self._phoneme_dictionary = self._load_phoneme_dictionary(
                    self.config.phoneme_dictionary_path
                )
            except Exception as e:
                logger.warning(
                    f"Failed to load phoneme dictionary from "
                    f"'{self.config.phoneme_dictionary_path}': {e}. "
                    f"Continuing without custom phoneme dictionary."
                )

    def _validate_mixed_language_config(self) -> None:
        """Validate mixed-language configuration.

        Raises:
            ValueError: If mixed-language is enabled but configuration is invalid
        """
        if not self.config.use_mixed_language:
            return

        # Require allowed_languages to be explicitly set
        if not self.config.mixed_language_allowed:
            raise ValueError(
                "use_mixed_language is enabled but mixed_language_allowed is not set. "
                "You must explicitly specify which languages to detect, e.g., "
                "mixed_language_allowed=['de', 'en-us', 'fr']"
            )

        # Validate all allowed languages are supported
        for lang in self.config.mixed_language_allowed:
            # Map to kokorog2p format for validation
            kokorog2p_lang = SUPPORTED_LANGUAGES.get(lang, lang)
            if kokorog2p_lang not in SUPPORTED_LANGUAGES.values():
                supported = sorted(set(SUPPORTED_LANGUAGES.keys()))
                raise ValueError(
                    f"Language '{lang}' in mixed_language_allowed is not supported. "
                    f"Supported languages: {supported}"
                )

        # Validate primary language if set
        if self.config.mixed_language_primary:
            primary = self.config.mixed_language_primary
            kokorog2p_primary = SUPPORTED_LANGUAGES.get(primary, primary)
            if kokorog2p_primary not in SUPPORTED_LANGUAGES.values():
                supported = sorted(set(SUPPORTED_LANGUAGES.keys()))
                raise ValueError(
                    f"Primary language '{primary}' is not supported. "
                    f"Supported languages: {supported}"
                )

            # Primary MUST be in allowed languages
            if primary not in self.config.mixed_language_allowed:
                raise ValueError(
                    f"Primary language '{primary}' must be in allowed_languages. "
                    f"Got primary='{primary}' but "
                    f"allowed={self.config.mixed_language_allowed}"
                )

        # Validate confidence threshold
        if not 0.0 <= self.config.mixed_language_confidence <= 1.0:
            raise ValueError(
                f"mixed_language_confidence must be between 0.0 and 1.0, "
                f"got {self.config.mixed_language_confidence}"
            )

    def _get_mixed_language_cache_key(self) -> str:
        """Generate cache key for mixed-language G2P instance.

        Returns:
            String key representing the current mixed-language configuration
        """
        if not self.config.use_mixed_language:
            return ""

        # Include all relevant config parameters in the key
        allowed = tuple(sorted(self.config.mixed_language_allowed or []))
        primary = self.config.mixed_language_primary or ""
        confidence = self.config.mixed_language_confidence

        return f"mixed:{primary}:{allowed}:{confidence}"

    def invalidate_mixed_language_cache(self) -> None:
        """Invalidate cached mixed-language G2P instance.

        Call this after changing mixed-language configuration to force
        recreation of the MixedLanguageG2P instance with new settings.
        """
        cache_key = self._get_mixed_language_cache_key()
        if cache_key and cache_key in self._g2p_cache:
            del self._g2p_cache[cache_key]
            logger.debug(f"Invalidated mixed-language G2P cache: {cache_key}")

    def _get_g2p(self, lang: str) -> G2PBase:
        """Get or create a G2P instance for the given language.

        If mixed-language mode is enabled, returns a MixedLanguageG2P instance.
        Otherwise, returns a standard single-language G2P instance.

        Args:
            lang: Language code (e.g., 'en-us', 'en-gb', 'de', 'fr-fr')

        Returns:
            G2P instance for the language (or MixedLanguageG2P if enabled)

        Raises:
            ValueError: If mixed-language config is invalid
        """
        # Validate mixed-language configuration if enabled
        self._validate_mixed_language_config()

        # If mixed-language mode is enabled, use MixedLanguageG2P
        if self.config.use_mixed_language and self.config.mixed_language_allowed:
            cache_key = self._get_mixed_language_cache_key()

            if cache_key not in self._g2p_cache:
                # Map primary language to kokorog2p format
                primary_lang = self.config.mixed_language_primary or lang
                kokorog2p_primary = SUPPORTED_LANGUAGES.get(primary_lang, primary_lang)

                # Map all allowed languages to kokorog2p format
                allowed_langs = [
                    SUPPORTED_LANGUAGES.get(lang_code, lang_code)
                    for lang_code in self.config.mixed_language_allowed
                ]

                try:
                    # Create MixedLanguageG2P instance
                    self._g2p_cache[cache_key] = MixedLanguageG2P(
                        primary_language=kokorog2p_primary,
                        allowed_languages=allowed_langs,
                        confidence_threshold=self.config.mixed_language_confidence,
                        enable_detection=True,
                        use_espeak_fallback=self.config.use_espeak_fallback,
                        use_spacy=self.config.use_spacy,
                        version=self._kokorog2p_model,
                    )
                    logger.info(
                        f"Created MixedLanguageG2P: primary={kokorog2p_primary}, "
                        f"allowed={allowed_langs}, "
                        f"confidence={self.config.mixed_language_confidence}"
                    )
                except ImportError as e:
                    # lingua-language-detector not available,
                    # fall back to single-language
                    logger.warning(
                        f"Mixed-language mode requested but "
                        f"lingua-language-detector is not available: {e}. "
                        f"Falling back to single-language mode."
                    )
                    # Disable mixed-language mode for this session
                    self.config.use_mixed_language = False
                    # Fall through to single-language G2P creation below

            if cache_key in self._g2p_cache:
                return self._g2p_cache[cache_key]

        # Standard single-language G2P
        if lang not in self._g2p_cache:
            # Map language to kokorog2p format
            kokorog2p_lang = SUPPORTED_LANGUAGES.get(lang, lang)

            # All languages are now fully supported by kokorog2p
            # kokorog2p uses dictionary + espeak fallback for all languages
            self._g2p_cache[lang] = get_g2p(
                language=kokorog2p_lang,
                use_espeak_fallback=self.config.use_espeak_fallback,
                use_spacy=self.config.use_spacy,
                version=self._kokorog2p_model,
            )

        return self._g2p_cache[lang]

    def _load_phoneme_dictionary(self, path: str | Path) -> dict[str, str]:
        """Load custom phoneme dictionary from JSON file.

        Args:
            path: Path to JSON file containing phoneme mappings

        Returns:
            Dictionary mapping words to phoneme strings

        Raises:
            FileNotFoundError: If file doesn't exist
            ValueError: If JSON format is invalid
        """
        file_path = Path(path)
        if not file_path.exists():
            raise FileNotFoundError(f"Phoneme dictionary not found: {path}")

        with open(file_path, encoding="utf-8") as f:
            data = json.load(f)

        # Support both simple format and metadata format
        if isinstance(data, dict):
            # Check if it's metadata format with "entries" key
            if "entries" in data and isinstance(data["entries"], dict):
                entries = data["entries"]
                # Support both simple string format and dict format with "phoneme" key
                phoneme_dict = {}
                for word, value in entries.items():
                    if isinstance(value, str):
                        phoneme_dict[word] = value
                    elif isinstance(value, dict) and "phoneme" in value:
                        phoneme_dict[word] = value["phoneme"]
                    else:
                        raise ValueError(
                            f"Invalid entry format for '{word}': {value}. "
                            f"Expected string or dict with 'phoneme' key."
                        )
            else:
                # Simple format: {word: phoneme}
                phoneme_dict = data
        else:
            raise ValueError(
                f"Phoneme dictionary must be a JSON object, got {type(data)}"
            )

        # Validate all phoneme values
        for word, phoneme in phoneme_dict.items():
            if not isinstance(phoneme, str):
                raise ValueError(
                    f"Phoneme for '{word}' must be a string, got {type(phoneme)}"
                )
            # Phonemes should be in /.../ format for markdown
            if not phoneme.startswith("/") or not phoneme.endswith("/"):
                raise ValueError(
                    f"Invalid phoneme format for '{word}': '{phoneme}'. "
                    f"Expected format: '/phoneme/' (e.g., '/mɪsˈɑki/')"
                )

        logger.info(f"Loaded {len(phoneme_dict)} custom phoneme entries from {path}")
        return phoneme_dict

    def filter_phonemes(self, text: str, replacement: str = "") -> str:
        """Filter text to only include characters in the tokenizer's vocabulary.

        This method uses the instance's vocabulary (which may be variant-specific)
        instead of the global kokorog2p vocabulary. This is important for models
        like v1.1-zh which have extended vocabularies with Bopomofo characters.

        Args:
            text: Input text/phonemes to filter
            replacement: Character to use for filtered characters (default: remove)

        Returns:
            Filtered text with only vocabulary characters
        """
        # Use instance vocabulary for filtering
        return "".join(c if c in self.vocab else replacement for c in text)

    def _apply_phoneme_dictionary(self, text: str) -> str:
        """Apply custom phoneme dictionary to text.

        Replaces words with markdown phoneme notation: [word](/phoneme/)

        Args:
            text: Input text

        Returns:
            Text with phoneme dictionary applied
        """
        if not self._phoneme_dictionary:
            return text

        result = text
        flags = 0 if self.config.phoneme_dict_case_sensitive else re.IGNORECASE

        # Sort by length (longest first) to handle multi-word entries correctly
        sorted_words = sorted(
            self._phoneme_dictionary.items(), key=lambda x: len(x[0]), reverse=True
        )

        for word, phoneme in sorted_words:
            # Create regex pattern with word boundaries
            # Use re.escape to handle special characters in the word
            pattern = r"\b" + re.escape(word) + r"\b"

            # Replace with markdown format: [word](/phoneme/)
            # Keep the slashes - kokorog2p requires them to recognize custom phonemes
            # Use a replacement function to preserve the original case
            def replace_func(match: re.Match, p=phoneme) -> str:
                matched_word = match.group(0)
                return f"[{matched_word}]({p})"

            result = re.sub(pattern, replace_func, result, flags=flags)

        return result

    @property
    def reverse_vocab(self) -> dict[int, str]:
        """Get the reverse vocabulary (token ID -> phoneme).

        Lazily constructed on first access.
        """
        if self._reverse_vocab is None:
            self._reverse_vocab = {v: k for k, v in self.vocab.items()}
        return self._reverse_vocab

    @staticmethod
    def normalize_text(text: str) -> str:
        """Normalize text before phonemization.

        Args:
            text: Input text

        Returns:
            Normalized text
        """
        return text.strip()

    def phonemize(
        self,
        text: str,
        lang: str = "en-us",
        normalize: bool = True,
    ) -> str:
        """Convert text to phonemes.

        If a custom phoneme dictionary is configured, words in the dictionary
        will be replaced with their custom pronunciations before phonemization.

        Args:
            text: Input text
            lang: Language code (e.g., 'en-us', 'en-gb')
            normalize: Whether to normalize text first

        Returns:
            Phoneme string (Kokoro format)

        Raises:
            ValueError: If language is not supported
        """
        if normalize:
            text = self.normalize_text(text)

        if not text:
            return ""

        # Apply custom phoneme dictionary first
        processed_text = self._apply_phoneme_dictionary(text)

        # Use phonemize_with_markdown if we have custom phonemes
        # Otherwise use standard phonemization
        if processed_text != text and "[" in processed_text:
            # Text contains markdown phoneme annotations
            try:
                phonemes = phonemize_with_markdown(processed_text, lang)
            except Exception as e:
                logger.warning(
                    f"Failed to use phonemize_with_markdown, "
                    f"falling back to standard phonemization: {e}"
                )
                # Fallback to standard phonemization
                g2p = self._get_g2p(lang)
                phonemes = g2p.phonemize(text)
        else:
            # Standard phonemization
            g2p = self._get_g2p(lang)
            phonemes = g2p.phonemize(text)

        # Filter to only characters in vocabulary using variant-specific model
        phonemes = filter_for_kokoro(phonemes, model=self._kokorog2p_model)

        return phonemes.strip()

    def phonemize_detailed(
        self,
        text: str,
        lang: str = "en-us",
        normalize: bool = True,
    ) -> PhonemeResult:
        """Convert text to phonemes with detailed token information.

        Args:
            text: Input text
            lang: Language code (e.g., 'en-us', 'en-gb')
            normalize: Whether to normalize text first

        Returns:
            PhonemeResult with phonemes, tokens, and quality metadata
        """
        if normalize:
            text = self.normalize_text(text)

        if not text:
            return PhonemeResult(phonemes="", tokens=[], low_confidence_words=[])

        # Get G2P instance for language
        g2p = self._get_g2p(lang)

        # Get tokens with per-word phonemes
        tokens = g2p(text)

        # Build phoneme string and identify low-confidence words
        phoneme_parts = []
        low_confidence = []

        for token in tokens:
            if token.phonemes:
                phoneme_parts.append(token.phonemes)
                # Check rating (1 = espeak fallback, 3-4 = dictionary)
                rating = token.get("rating", 4)
                if rating is not None and rating < 2:
                    low_confidence.append(token.text)
            if token.whitespace:
                phoneme_parts.append(" ")

        phonemes = "".join(phoneme_parts)
        phonemes = filter_for_kokoro(phonemes, model=self._kokorog2p_model)

        return PhonemeResult(
            phonemes=phonemes.strip(),
            tokens=tokens,
            low_confidence_words=low_confidence,
        )

    def tokenize(self, phonemes: str) -> list[int]:
        """Convert phonemes to token IDs.

        Args:
            phonemes: Phoneme string (Kokoro format)

        Returns:
            List of token IDs

        Raises:
            ValueError: If phoneme string exceeds MAX_PHONEME_LENGTH
        """
        if len(phonemes) > MAX_PHONEME_LENGTH:
            raise ValueError(
                f"Phoneme string too long ({len(phonemes)} chars). "
                f"Maximum is {MAX_PHONEME_LENGTH} phonemes."
            )

        return phonemes_to_ids(phonemes, model=self._kokorog2p_model)

    def detokenize(self, tokens: list[int]) -> str:
        """Convert token IDs back to phonemes.

        Args:
            tokens: List of token IDs

        Returns:
            Phoneme string
        """
        return ids_to_phonemes(tokens, model=self._kokorog2p_model)

    def text_to_tokens(
        self,
        text: str,
        lang: str = "en-us",
        normalize: bool = True,
    ) -> list[int]:
        """Convert text directly to tokens.

        Convenience method combining phonemize() and tokenize().

        Args:
            text: Input text
            lang: Language code
            normalize: Whether to normalize text first

        Returns:
            List of token IDs
        """
        phonemes = self.phonemize(text, lang=lang, normalize=normalize)
        return self.tokenize(phonemes)

    def text_to_phonemes_with_words(
        self,
        text: str,
        lang: str = "en-us",
    ) -> list[tuple[str, str]]:
        """Convert text to phonemes, preserving word boundaries.

        Useful for creating readable phoneme exports.

        Args:
            text: Input text
            lang: Language code

        Returns:
            List of (word, phonemes) tuples
        """
        g2p = self._get_g2p(lang)
        tokens = g2p(text)

        result = []
        for token in tokens:
            if token.phonemes and token.text.strip():
                # Filter phonemes for Kokoro vocabulary
                filtered_phonemes = filter_for_kokoro(
                    token.phonemes, model=self._kokorog2p_model
                )
                result.append((token.text, filtered_phonemes))

        return result

    def format_readable(
        self,
        text: str,
        lang: str = "en-us",
    ) -> str:
        """Format text with phonemes in a human-readable way.

        Args:
            text: Input text
            lang: Language code

        Returns:
            Formatted string like "Hello [həˈloʊ] world [wɜːld]"
        """
        word_phonemes = self.text_to_phonemes_with_words(text, lang=lang)
        return " ".join(f"{word} [{phonemes}]" for word, phonemes in word_phonemes)

    def split_with_pauses(
        self,
        text: str,
        pause_short: float = 0.3,
        pause_medium: float = 0.6,
        pause_long: float = 1.0,
    ) -> tuple[float, list[tuple[str, float]]]:
        """Split text at pause markers and return segments with pause durations.

        Detects and splits on pause markers: (.), (..), (...)
        Removes the markers from the text segments.
        Consecutive pause markers have their durations added together.

        Args:
            text: Input text with optional pause markers
            pause_short: Duration for (.) in seconds
            pause_medium: Duration for (..) in seconds
            pause_long: Duration for (...) in seconds

        Returns:
            Tuple of (initial_pause_duration, segments_list) where segments_list
            is a list of (text_segment, pause_after_seconds) tuples

        Example:
            >>> split_with_pauses("Hello (.) World (...) Foo")
            (0.0, [("Hello", 0.3), ("World", 1.0), ("Foo", 0.0)])

            >>> split_with_pauses("Start (...) (..) End")
            (0.0, [("Start", 1.6), ("End", 0.0)])  # 1.0 + 0.6 = 1.6

            >>> split_with_pauses("(...) Hello")
            (1.0, [("Hello", 0.0)])  # Leading pause
        """
        import re

        # Pattern to match pause markers: (.), (..), (...)
        # Use lookbehind and lookahead to split while preserving markers
        pattern = r"\(\.\.\.\)|\(\.\.\)|\(\.\)"

        # Split text and capture markers
        parts = re.split(f"({pattern})", text)

        # Process parts into segments with pauses
        segments = []
        current_text = ""
        accumulated_pause = 0.0
        initial_pause = 0.0
        found_first_text = False

        for part in parts:
            if not part:
                continue

            # Check if this is a pause marker
            if re.match(pattern, part):
                # Determine pause duration
                if part == "(...)":
                    pause_duration = pause_long
                elif part == "(..)":
                    pause_duration = pause_medium
                else:  # '(.)'
                    pause_duration = pause_short

                # If we haven't found any text yet, this is initial pause
                if not found_first_text and not current_text.strip():
                    initial_pause += pause_duration
                else:
                    # Accumulate pause for current segment
                    accumulated_pause += pause_duration
            else:
                # This is text
                stripped = part.strip()
                if stripped:
                    found_first_text = True

                    # If we have accumulated text, save it as a segment
                    if current_text.strip():
                        segments.append((current_text.strip(), accumulated_pause))
                        accumulated_pause = 0.0
                        current_text = ""

                    current_text = stripped
                elif current_text.strip():
                    # Whitespace part after we have text, keep accumulating
                    current_text += part

        # Add final segment if any
        if current_text.strip():
            segments.append((current_text.strip(), accumulated_pause))

        return initial_pause, segments

    def get_vocab_info(self) -> dict:
        """Get information about the current vocabulary.

        Returns:
            Dictionary with vocabulary metadata
        """
        return {
            "version": self.vocab_version,
            "num_tokens": len(self.vocab),
            "max_token_id": max(self.vocab.values()) if self.vocab else 0,
            "max_phoneme_length": MAX_PHONEME_LENGTH,
            "n_tokens": N_TOKENS,
            "backend": "kokorog2p",
        }

    def validate_phonemes(self, phonemes: str) -> tuple[bool, list[str]]:
        """Validate that all characters are in the Kokoro vocabulary.

        Args:
            phonemes: Phoneme string to validate

        Returns:
            Tuple of (is_valid, list_of_invalid_chars)
        """
        return validate_for_kokoro(phonemes)


# Convenience function for simple usage
def create_tokenizer(
    use_espeak_fallback: bool = True,
    use_spacy: bool = True,
) -> Tokenizer:
    """Create a tokenizer with the specified configuration.

    Args:
        use_espeak_fallback: Whether to use espeak for OOV words
        use_spacy: Whether to use spaCy for POS tagging

    Returns:
        Configured Tokenizer instance
    """
    config = TokenizerConfig(
        use_espeak_fallback=use_espeak_fallback,
        use_spacy=use_spacy,
    )
    return Tokenizer(config=config)
