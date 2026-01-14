"""Mixed-language phonemization support for pykokoro.

This module handles automatic language detection and mixed-language text-to-phoneme
conversion using kokorog2p's MixedLanguageG2P capability.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

from kokorog2p.base import G2PBase
from kokorog2p.mixed_language_g2p import MixedLanguageG2P

from .constants import SUPPORTED_LANGUAGES

if TYPE_CHECKING:
    from .tokenizer import KokoroConfig

logger = logging.getLogger(__name__)


class MixedLanguageHandler:
    """Handles mixed-language G2P configuration and caching.

    This class manages:
    - Mixed-language configuration validation
    - MixedLanguageG2P instance creation and caching
    - Cache invalidation when configuration changes
    """

    def __init__(self, config: KokoroConfig, kokorog2p_model: str | None = None):
        """Initialize mixed-language handler.

        Args:
            config: KokoroConfig instance with mixed-language settings
            kokorog2p_model: Optional kokorog2p model version (e.g., 'v0.1', 'v1.0')
        """
        self.config = config
        self._kokorog2p_model = kokorog2p_model
        self._g2p_cache: dict[str, G2PBase] = {}

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

    def get_cache_key(self) -> str:
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

    def invalidate_cache(self) -> None:
        """Invalidate cached mixed-language G2P instance.

        Call this after changing mixed-language configuration to force
        recreation of the MixedLanguageG2P instance with new settings.
        """
        cache_key = self.get_cache_key()
        if cache_key and cache_key in self._g2p_cache:
            del self._g2p_cache[cache_key]
            logger.debug(f"Invalidated mixed-language G2P cache: {cache_key}")

    def get_or_create_g2p(
        self,
        lang: str,
        use_espeak_fallback: bool,
        use_goruut_fallback: bool,
        use_spacy: bool,
    ) -> G2PBase | None:
        """Get or create a MixedLanguageG2P instance if mixed-language mode is enabled.

        Args:
            lang: Default language code (e.g., 'en-us', 'en-gb', 'de', 'fr-fr')
            use_espeak_fallback: Whether to use espeak fallback
            use_goruut_fallback: Whether to use goruut fallback
            use_spacy: Whether to use spaCy

        Returns:
            MixedLanguageG2P instance if mixed-language mode is enabled and configured,
            None otherwise (caller should fall back to single-language G2P)

        Raises:
            ValueError: If mixed-language config is invalid
        """
        # Validate mixed-language configuration if enabled
        self.validate_config()

        # If mixed-language mode is not enabled, return None
        if not self.config.use_mixed_language or not self.config.mixed_language_allowed:
            return None

        cache_key = self.get_cache_key()

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
                    use_espeak_fallback=use_espeak_fallback,
                    use_goruut_fallback=use_goruut_fallback,
                    use_spacy=use_spacy,
                    backend=self.config.backend,
                    load_gold=self.config.load_gold,
                    load_silver=self.config.load_silver,
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
                return None

        return self._g2p_cache.get(cache_key)
