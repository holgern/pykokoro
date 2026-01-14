"""Constants for pykokoro - default configuration and program metadata."""

# Program metadata
PROGRAM_NAME = "pykokoro"

# Default configuration
# Structure: {"model_quality": "fp32", "use_gpu": False, "vocab_version": "v1.0"}
# Keys: model_quality (quantization), use_gpu (bool), vocab_version (str)
DEFAULT_CONFIG = {
    # Options: fp32, fp16, q8, q8f16, q4, q4f16, uint8, uint8f16
    "model_quality": "fp32",
    # Whether to use GPU acceleration
    "use_gpu": False,
    # Vocabulary version
    "vocab_version": "v1.0",
}

# Model constants
MAX_PHONEME_LENGTH = 510
SAMPLE_RATE = 24000
