"""Mixed-language phonemization support for pykokoro.

This module handles automatic language detection and mixed-language text-to-phoneme
conversion using kokorog2p's preprocess_multilang capability.
"""

from __future__ import annotations

import logging
import re
from typing import TYPE_CHECKING

import kokorog2p as _kokorog2p
from kokorog2p.multilang import preprocess_multilang

from .constants import SUPPORTED_LANGUAGES

if TYPE_CHECKING:
    from .tokenizer import TokenizerConfig

logger = logging.getLogger(__name__)
ANNOTATION_REGEX = getattr(
    _kokorog2p, "ANNOTATION_REGEX", re.compile(r"\[[^\]]+\]\{[^}]+\}")
)


class MixedLanguageHandler:
    """Handles mixed-language G2P configuration and preprocessing.

    This class manages:
    - Mixed-language configuration validation
    - Text preprocessing with language detection
    """

    def __init__(self, config: TokenizerConfig, kokorog2p_model: str | None = None):
        """Initialize mixed-language handler.

        Args:
            config: TokenizerConfig instance with mixed-language settings
            kokorog2p_model: Optional kokorog2p model version (e.g., 'v0.1', 'v1.0')
        """
        self.config = config
        self._kokorog2p_model = kokorog2p_model

    def validate_config(self) -> None:
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

    def preprocess_text(self, text: str, default_language: str) -> str:
        """Preprocess text for mixed-language phonemization.

        Uses kokorog2p's preprocess_multilang to add language annotations.
        Respects existing language annotations (doesn't re-preprocess).

        Args:
            text: Input text to preprocess
            default_language: Default language for unannotated words

        Returns:
            Text with language annotations in SSMD format

        Raises:
            ValueError: If mixed-language config is invalid
        """
        if not self.config.use_mixed_language:
            return text

        # Validate configuration first
        self.validate_config()

        # Skip if text already has language annotations
        if ANNOTATION_REGEX.search(text):
            return text

        # Map primary language to kokorog2p format
        primary_lang = self.config.mixed_language_primary or default_language
        kokorog2p_primary = SUPPORTED_LANGUAGES.get(primary_lang, primary_lang)

        # Map allowed languages to kokorog2p format
        mixed_allowed = self.config.mixed_language_allowed or []
        allowed_langs = [
            SUPPORTED_LANGUAGES.get(lang_code, lang_code) for lang_code in mixed_allowed
        ]

        try:
            kwargs = {
                "text": text,
                "default_language": kokorog2p_primary,
                "allowed_languages": allowed_langs,
                "confidence_threshold": self.config.mixed_language_confidence,
            }
            try:
                import inspect

                params = inspect.signature(preprocess_multilang).parameters
                if "markdown_syntax" in params:
                    kwargs["markdown_syntax"] = "ssmd"
            except (TypeError, ValueError):
                pass

            overrides = preprocess_multilang(**kwargs)
            if isinstance(overrides, str):
                return overrides
            return self._apply_overrides(text, overrides)
        except ImportError:
            logger.warning(
                "Mixed-language mode requested but lingua-language-detector "
                "not available. Returning text without preprocessing."
            )
            return text

    def _apply_overrides(self, text: str, overrides: list[object]) -> str:
        if not overrides:
            return text

        try:
            from kokorog2p.types import OverrideSpan
        except Exception:
            return text

        spans = [s for s in overrides if isinstance(s, OverrideSpan)]
        spans.sort(key=lambda s: s.char_start)
        if not spans:
            return text

        out: list[str] = []
        cursor = 0
        for span in spans:
            if span.char_start < cursor:
                continue
            out.append(text[cursor : span.char_start])
            chunk = text[span.char_start : span.char_end]
            lang = span.attrs.get("lang")
            if lang:
                out.append(f'[{chunk}]{{lang="{lang}"}}')
            else:
                out.append(chunk)
            cursor = span.char_end
        out.append(text[cursor:])
        return "".join(out)
