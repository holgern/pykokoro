# PyKokoro

A Python library for Kokoro TTS (Text-to-Speech) using ONNX runtime.

## Features

- **ONNX-based TTS**: Fast, efficient text-to-speech using the Kokoro-82M model
- **Multiple Languages**: Support for English, Spanish, French, German, Italian,
  Portuguese, and more
- **Multiple Voices**: 50+ built-in voices with different accents and genders
- **Voice Blending**: Create custom voices by blending multiple voices
- **Model Quality Options**: Choose from fp32, fp16, q8, q4, and uint8 quantization
  levels
- **GPU Acceleration**: Optional CUDA, CoreML, or DirectML support
- **Phoneme Support**: Advanced phoneme-based generation with kokorog2p
- **Hugging Face Integration**: Automatic model downloading from Hugging Face Hub

## Installation

```bash
pip install pykokoro
```

### Optional Dependencies

For GPU acceleration:

```bash
# CUDA
pip install onnxruntime-gpu

# macOS (CoreML)
# Already included in onnxruntime on macOS

# Windows (DirectML)
pip install onnxruntime-directml
```

## Quick Start

```python
import pykokoro
import soundfile as sf

# Initialize the TTS engine
tts = pykokoro.Kokoro()

# Generate speech
text = "Hello, world! This is Kokoro speaking."
audio, sample_rate = tts.create(text, voice="af_sarah", speed=1.0, lang="en-us")

# Save to file
sf.write("output.wav", audio, sample_rate)
```

## Usage Examples

### Basic Text-to-Speech

```python
import pykokoro

# Create TTS instance
tts = pykokoro.Kokoro(use_gpu=True, model_quality="fp16")

# Generate audio
audio, sr = tts.create("Hello world", voice="af_nicole", lang="en-us")
```

### Voice Blending

```python
# Blend two voices (50% each)
blend = pykokoro.VoiceBlend.parse("af_nicole:50,am_michael:50")
audio, sr = tts.create("Mixed voice", voice=blend)
```

### Streaming Generation

```python
# Synchronous streaming
for chunk, sr, text_chunk in tts.create_stream_sync("Long text here...", voice="af_sarah"):
    # Process audio chunk in real-time
    play_audio(chunk, sr)

# Async streaming
async for chunk, sr, text_chunk in tts.create_stream("Long text here...", voice="af_sarah"):
    await process_audio(chunk, sr)
```

### Phoneme-Based Generation

```python
from pykokoro import Tokenizer

# Create tokenizer
tokenizer = Tokenizer()

# Convert text to phonemes
phonemes = tokenizer.phonemize("Hello world", lang="en-us")
print(phonemes)  # hə'loʊ wɜːld

# Generate from phonemes
audio, sr = tts.create_from_phonemes(phonemes, voice="af_sarah")
```

## Available Voices

The library includes 50+ voices across different languages and accents:

- **American English**: af_alloy, af_bella, af_sarah, am_adam, am_michael, etc.
- **British English**: bf_alice, bf_emma, bm_george, bm_lewis
- **Spanish**: ef_dora, em_alex
- **French**: ff_siwis
- **Japanese**: jf_alpha, jm_kumo
- **Chinese**: zf_xiaobei, zm_yunxi
- And many more...

List all available voices:

```python
voices = tts.get_voices()
print(voices)
```

## Model Quality Options

Choose the model quality based on your needs:

- `fp32`: Full precision (highest quality, largest size)
- `fp16`: Half precision (good quality, smaller size)
- `q8`: 8-bit quantized (fast, small)
- `q4`: 4-bit quantized (fastest, smallest)
- `uint8`: Unsigned 8-bit (compatible)

```python
tts = pykokoro.Kokoro(model_quality="q8")
```

## Configuration

Configuration is stored in a platform-specific directory:

- Linux: `~/.config/pykokoro/config.json`
- macOS: `~/Library/Application Support/pykokoro/config.json`
- Windows: `%APPDATA%\pykokoro\config.json`

```python
import pykokoro

# Load config
config = pykokoro.load_config()

# Modify config
config["model_quality"] = "fp16"
config["use_gpu"] = True

# Save config
pykokoro.save_config(config)
```

## Advanced Features

### Custom Phoneme Dictionary

```python
from pykokoro import Tokenizer, TokenizerConfig

# Create config with custom phoneme dictionary
config = TokenizerConfig(
    phoneme_dictionary_path="my_pronunciations.json"
)

tokenizer = Tokenizer(config=config)
```

### Mixed Language Support

```python
from pykokoro import TokenizerConfig

config = TokenizerConfig(
    use_mixed_language=True,
    mixed_language_primary="en-us",
    mixed_language_allowed=["en-us", "de", "fr"]
)

tokenizer = Tokenizer(config=config)
```

## License

This library is licensed under the Apache License 2.0.

## Credits

- **Kokoro Model**: [hexgrad/Kokoro-82M](https://huggingface.co/hexgrad/Kokoro-82M)
- **ONNX Models**:
  [onnx-community/Kokoro-82M-v1.0-ONNX](https://huggingface.co/onnx-community/Kokoro-82M-v1.0-ONNX)
- **Phonemizer**: [kokorog2p](https://github.com/remyxai/kokorog2p)

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## Links

- **GitHub**: https://github.com/holgern/pykokoro
- **PyPI**: https://pypi.org/project/pykokoro/
- **Documentation**: https://pykokoro.readthedocs.io/
