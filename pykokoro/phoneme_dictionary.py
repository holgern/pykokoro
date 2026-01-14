"""Custom phoneme dictionary support for pykokoro.

This module provides utilities for loading and applying custom phoneme dictionaries
that override default G2P conversions with user-specified phoneme representations.

The phoneme dictionary uses kokorog2p's markdown-style syntax: [word](/phoneme/)
This format is recognized by kokorog2p during phonemization.

Dictionary entries are applied to text BEFORE SSMD parsing and phonemization,
so SSMD markup in the input text is preserved and processed normally.
"""

from __future__ import annotations

import json
import logging
import re
from pathlib import Path

logger = logging.getLogger(__name__)


class PhonemeDictionary:
    """Manages custom phoneme dictionary loading and application.

    A phoneme dictionary allows users to specify custom phoneme representations
    for specific words, overriding the default G2P output. Entries are applied
    using kokorog2p's markdown format: [word](/phoneme/)

    Example dictionary JSON format:
    {
        "entries": {
            "Misaki": "mɪsˈɑki",
            "complex_word": {
                "phoneme": "kəmˈplɛks wɜːrd",
                "description": "optional description"
            }
        }
    }

    Or simple format:
    {
        "Misaki": "mɪsˈɑki",
        "complex_word": "kəmˈplɛks wɜːrd"
    }

    Note: Phonemes in JSON should NOT include /slashes/. The slashes are
    added automatically when applying to text in kokorog2p format.
    """

    def __init__(
        self,
        dictionary_path: str | Path | None = None,
        case_sensitive: bool = False,
    ):
        """Initialize phoneme dictionary.

        Args:
            dictionary_path: Optional path to JSON phoneme dictionary file
            case_sensitive: Whether word matching should be case-sensitive
        """
        self.case_sensitive = case_sensitive
        self._dictionary: dict[str, str] = {}

        if dictionary_path:
            self._dictionary = self.load(dictionary_path)

    def load(self, path: str | Path) -> dict[str, str]:
        """Load custom phoneme dictionary from JSON file.

        Phonemes can be specified with or without /slashes/. The slashes
        are automatically stripped and the appropriate format is applied
        during text processing based on use_ssmd_format setting.

        Args:
            path: Path to JSON file containing phoneme mappings

        Returns:
            Dictionary mapping words to phoneme strings (without slashes)

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

        # Validate and normalize all phoneme values (strip slashes if present)
        normalized_dict = {}
        for word, phoneme in phoneme_dict.items():
            if not isinstance(phoneme, str):
                raise ValueError(
                    f"Phoneme for '{word}' must be a string, got {type(phoneme)}"
                )

            # Strip /slashes/ if present (support legacy format)
            cleaned_phoneme = phoneme.strip()
            if cleaned_phoneme.startswith("/") and cleaned_phoneme.endswith("/"):
                cleaned_phoneme = cleaned_phoneme[1:-1]

            if not cleaned_phoneme:
                raise ValueError(f"Empty phoneme for '{word}' after cleaning")

            normalized_dict[word] = cleaned_phoneme

        logger.info(f"Loaded {len(normalized_dict)} custom phoneme entries from {path}")
        self._dictionary = normalized_dict
        return normalized_dict

    def apply(self, text: str) -> str:
        """Apply custom phoneme dictionary to text.

        Replaces words with markdown phoneme notation: [word](/phoneme/)

        Args:
            text: Input text

        Returns:
            Text with phoneme dictionary applied
        """
        if not self._dictionary:
            return text

        result = text
        flags = 0 if self.case_sensitive else re.IGNORECASE

        # Sort by length (longest first) to handle multi-word entries correctly
        sorted_words = sorted(
            self._dictionary.items(), key=lambda x: len(x[0]), reverse=True
        )

        for word, phoneme in sorted_words:
            # Create regex pattern with word boundaries
            # Use re.escape to handle special characters in the word
            pattern = r"\b" + re.escape(word) + r"\b"

            # Replace with markdown format: [word](/phoneme/)
            # Keep the slashes - kokorog2p requires them to recognize custom phonemes
            # Use a replacement function to preserve the original case
            def replace_func(match: re.Match[str], p: str = phoneme) -> str:
                matched_word = match.group(0)
                return f"[{matched_word}](/{p}/)"

            result = re.sub(pattern, replace_func, result, flags=flags)

        return result

    def has_entries(self) -> bool:
        """Check if dictionary has any entries.

        Returns:
            True if dictionary has entries, False otherwise
        """
        return bool(self._dictionary)

    def get_phoneme(self, word: str) -> str | None:
        """Get phoneme for a specific word.

        Args:
            word: Word to look up

        Returns:
            Phoneme string if found, None otherwise
        """
        if self.case_sensitive:
            return self._dictionary.get(word)
        else:
            # Case-insensitive lookup
            word_lower = word.lower()
            for key, value in self._dictionary.items():
                if key.lower() == word_lower:
                    return value
            return None

    def __len__(self) -> int:
        """Return number of entries in dictionary."""
        return len(self._dictionary)

    def __bool__(self) -> bool:
        """Return True if dictionary has entries."""
        return bool(self._dictionary)
