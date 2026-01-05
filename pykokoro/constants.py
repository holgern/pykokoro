"""Constants for pykokoro - default configuration and program metadata."""

# Program metadata
PROGRAM_NAME = "pykokoro"

# Default configuration
DEFAULT_CONFIG = {
    "model_quality": "fp32",  # Options: fp32, fp16, q8, q8f16, q4, q4f16, uint8, uint8f16
    "use_gpu": False,  # Whether to use GPU acceleration
    "vocab_version": "v1.0",  # Vocabulary version
}
