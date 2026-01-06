"""Constants for pykokoro - default configuration and program metadata."""

# Program metadata
PROGRAM_NAME = "pykokoro"

# Default configuration
DEFAULT_CONFIG = {
    # Options: fp32, fp16, q8, q8f16, q4, q4f16, uint8, uint8f16
    "model_quality": "fp32",
    # Whether to use GPU acceleration
    "use_gpu": False,
    # Vocabulary version
    "vocab_version": "v1.0",
}
"""Default configuration dictionary for PyKokoro.

Contains default settings for model quality, GPU usage, and vocabulary version.

Dictionary structure:
    {"model_quality": "fp32", "use_gpu": False, "vocab_version": "v1.0"}

Keys are: model_quality (quantization level), use_gpu (boolean), vocab_version (string)
"""
