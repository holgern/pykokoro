class KokoroError(Exception):
    """Base exception for pykokoro."""


class ConfigurationError(KokoroError):
    """Invalid or inconsistent configuration."""


class CapabilityError(KokoroError):
    """Requested SSMD/SSML feature is unsupported by selected backend/profile."""


class AlignmentError(KokoroError):
    """Annotation/token alignment failed in a way that can't be recovered."""


class BackendError(KokoroError):
    """Synthesis backend failed (onnx runtime, model I/O, etc.)."""
