[![PyPI - Version](https://img.shields.io/pypi/v/pykokoro)](https://pypi.org/project/pykokoro/)
![PyPI - Python Version](https://img.shields.io/pypi/pyversions/pykokoro)
![PyPI - Downloads](https://img.shields.io/pypi/dm/pykokoro)
[![codecov](https://codecov.io/gh/holgern/pykokoro/graph/badge.svg?token=iCHXwbjAXG)](https://codecov.io/gh/holgern/pykokoro)

# PyKokoro

A Python library for Kokoro TTS (Text-to-Speech) using ONNX runtime.

## Features

- **ONNX-based TTS**: Fast, efficient text-to-speech using the Kokoro-82M model
- **Multiple Languages**: Support for English, Spanish, French, German, Italian,
  Portuguese, and more
- **Multiple Voices**: 54+ built-in voices (or 103 voices with v1.1-zh model)
- **Voice Blending**: Create custom voices by blending multiple voices
- **Multiple Model Sources**: Download models from HuggingFace or GitHub (v1.0/v1.1-zh)
- **Model Quality Options**: Choose from fp32, fp16, q8, q4, and uint8 quantization
  levels
- **GPU Acceleration**: Optional CUDA, CoreML, or DirectML support
- **Phoneme Support**: Advanced phoneme-based generation with kokorog2p
- **Hugging Face Integration**: Automatic model downloading from Hugging Face Hub
- **Text Normalization**: Automatic say-as support for numbers, dates, phone numbers,
  and more using SSMD markup

## Installation

### Basic Installation (CPU only)

```bash
pip install pykokoro
```

### GPU and Accelerator Support

PyKokoro supports multiple hardware accelerators for faster inference:

#### NVIDIA CUDA GPU

```bash
pip install pykokoro[gpu]
```

#### Intel OpenVINO

**Note:** OpenVINO is currently incompatible with Kokoro models due to dynamic rank
tensor requirements. The provider will automatically fall back to CPU if OpenVINO fails.

```bash
pip install pykokoro[openvino]
```

#### DirectML (Windows - AMD/Intel/NVIDIA GPUs)

```bash
pip install pykokoro[directml]
```

#### Apple CoreML (macOS)

```bash
pip install pykokoro[coreml]
```

#### All Accelerators

```bash
pip install pykokoro[all]
```

### Performance Comparison

To find the best provider for your system, run the benchmark:

```bash
python examples/gpu_benchmark.py
```

## Quick Start

```python
import pykokoro
import soundfile as sf

# Initialize the TTS engine (auto-selects best provider)
tts = pykokoro.Kokoro(provider="auto")

# Generate speech
text = "Hello, world! This is Kokoro speaking."
audio, sample_rate = tts.create(text, voice="af_sarah", speed=1.0, lang="en-us")

# Save to file
sf.write("output.wav", audio, sample_rate)
```

## Hardware Acceleration

### Automatic Provider Selection (Recommended)

```python
import pykokoro

# Auto-select best available provider (CUDA > CoreML > DirectML > CPU)
# Note: OpenVINO is attempted but will fall back to next priority if incompatible
tts = pykokoro.Kokoro(provider="auto")
```

### Explicit Provider Selection

```python
# Force specific provider
tts = pykokoro.Kokoro(provider="cuda")      # NVIDIA CUDA
tts = pykokoro.Kokoro(provider="openvino")  # Intel OpenVINO (currently incompatible, will raise error)
tts = pykokoro.Kokoro(provider="directml")  # Windows DirectML
tts = pykokoro.Kokoro(provider="coreml")    # Apple CoreML
tts = pykokoro.Kokoro(provider="cpu")       # CPU only
```

### Check Available Providers

```bash
# See all available providers on your system
python examples/provider_info.py

# Benchmark all providers
python examples/gpu_benchmark.py
```

### Environment Variable Override

```bash
# Force a specific provider via environment variable
export ONNX_PROVIDER="OpenVINOExecutionProvider"
python your_script.py
```

## Usage Examples

### Basic Text-to-Speech

```python
import pykokoro

# Create TTS instance with GPU acceleration and fp16 model
tts = pykokoro.Kokoro(provider="cuda", model_quality="fp16")

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

### Pause Control

PyKokoro uses SSMD (Speech Synthesis Markdown) syntax for controlling pauses in
generated speech:

#### 1. SSMD Break Markers

Add explicit pauses using SSMD break syntax in your text:

```python
# Use SSMD break markers in your text
text = "Chapter 5 ...p I'm Klaus. ...c Welcome to the show!"

# Breaks are processed automatically
audio, sr = tts.create(
    text,
    voice="am_michael"
)
```

**SSMD Break Markers:**

- `...n` - No pause (0ms)
- `...w` - Weak pause (150ms by default)
- `...c` - Clause/comma pause (300ms by default)
- `...s` - Sentence pause (600ms by default)
- `...p` - Paragraph pause (1000ms by default)
- `...500ms` - Custom pause (500 milliseconds)
- `...2s` - Custom pause (2 seconds)

**Note:** Bare `...` (ellipsis) is NOT treated as a pause and will be phonemized
normally.

**Custom Pause Durations:**

```python
audio, sr = tts.create(
    text,
    voice="am_michael",
    pause_clause=0.2,      # ...c = 200ms
    pause_sentence=0.5,    # ...s = 500ms
    pause_paragraph=1.5    # ...p = 1500ms
)
```

#### 2. Automatic Natural Pauses

For more natural speech, enable automatic pause insertion at linguistic boundaries:

```python
text = """
Artificial intelligence is transforming our world. Machine learning models
are becoming more sophisticated, efficient, and accessible.

Deep learning, a subset of AI, uses neural networks with many layers. These
networks can learn complex patterns from data, enabling breakthroughs in
computer vision, natural language processing, and speech recognition.
"""

# Automatic pauses at clause, sentence, and paragraph boundaries
audio, sr = tts.create(
    text,
    voice="af_sarah",
    split_mode="clause",      # Split on commas and sentences
    trim_silence=True,        # Enable automatic pause insertion
    pause_clause=0.25,        # Pause after clauses (commas)
    pause_sentence=0.5,       # Pause after sentences
    pause_paragraph=1.0,      # Pause after paragraphs
    pause_variance=0.05,      # Add natural variance (default)
    random_seed=42            # For reproducible results (optional)
)
```

**Key Features:**

- **Natural boundaries**: Automatically detects clauses, sentences, and paragraphs
- **Variance**: Gaussian variance prevents robotic timing (±100ms by default)
- **Reproducible**: Use `random_seed` for consistent output
- **Composable**: Works with SSMD break markers

**Split Modes:**

- `None` (default) - Automatic phoneme-based splitting, no automatic pauses
- `"paragraph"` - Split on double newlines
- `"sentence"` - Split on sentence boundaries (requires spaCy)
- `"clause"` - Split on sentences + commas (requires spaCy, recommended)

**Pause Variance Options:**

- `pause_variance=0.0` - No variance (exact pauses)
- `pause_variance=0.05` - Default (±100ms at 95% confidence)
- `pause_variance=0.1` - More variation (±200ms at 95% confidence)

**Note:** For `split_mode="sentence"` or `split_mode="clause"`, install spaCy:

```bash
pip install spacy
python -m spacy download en_core_web_sm
```

**Combining Both Approaches:**

Use SSMD markers for special emphasis and automatic pauses for natural rhythm:

```python
text = "Welcome! ...p Let's discuss AI, machine learning, and deep learning."

audio, sr = tts.create(
    text,
    voice="af_sarah",
    split_mode="clause",     # Automatic pauses at commas
    trim_silence=True,
    pause_variance=0.05
)
```

See `examples/pauses_demo.py`, `examples/pauses_with_splitting.py`, and
`examples/automatic_pauses_demo.py` for complete examples.

### Text Normalization (Say-As)

PyKokoro supports automatic text normalization using SSMD (Speech Synthesis Markdown)
syntax. Convert numbers, dates, phone numbers, and more into speakable text:

```python
# Cardinal numbers
text = "I have [123](as: cardinal) apples"
audio, sr = tts.create(text, voice="af_sarah")
# TTS says: "I have one hundred twenty-three apples"

# Ordinal numbers
text = "I came in [3](as: ordinal) place"
# TTS says: "I came in third place"

# Digits (spell out)
text = "My PIN is [1234](as: digits)"
# TTS says: "My PIN is one two three four"

# Telephone numbers
text = "Call [+1-555-0123](as: telephone)"
# TTS says: "Call plus one five five five oh one two three"

# Dates with custom formatting
text = "Today is [12/31/2024](as: date, format: mdy)"
# TTS says: "Today is December thirty-first, two thousand twenty-four"

# Time (12-hour or 24-hour)
text = "The time is [14:30](as: time)"
# TTS says: "The time is two thirty PM"

# Characters (spell out)
text = "The code is [ABC](as: characters)"
# TTS says: "The code is A B C"

# Fractions
text = "Add [1/2](as: fraction) cup of sugar"
# TTS says: "Add one half cup of sugar"

# Units
text = "The package weighs [5kg](as: unit)"
# TTS says: "The package weighs five kilograms"
```

**Supported Say-As Types:**

- `cardinal` - Numbers as cardinals: "123" → "one hundred twenty-three"
- `ordinal` - Numbers as ordinals: "3" → "third"
- `digits` - Spell out digits: "123" → "one two three"
- `number` - Alias for cardinal
- `fraction` - Fractions: "1/2" → "one half"
- `characters` - Spell out text: "ABC" → "A B C"
- `telephone` - Phone numbers: "+1-555-0123" → "plus one five five five oh one two
  three"
- `date` - Dates with format support (mdy, dmy, ymd, ym, my, md, dm, d, m, y)
- `time` - Time in 12h or 24h format
- `unit` - Units: "5kg" → "five kilograms"
- `expletive` - Censors to "beep"

**Multi-language Support:**

Say-as works with multiple languages (English, French, German, Spanish, and more):

```python
# French cardinal
text = "[123](as: cardinal)"
audio, sr = tts.create(text, voice="ff_siwis", lang="fr-fr")
# TTS says: "cent vingt-trois"

# German ordinal
text = "[3](as: ordinal)"
audio, sr = tts.create(text, voice="gf_maria", lang="de-de")
# TTS says: "dritte"
```

**Combining with Other Features:**

Say-as works seamlessly with all SSMD features:

```python
# With prosody
text = "[100](as: cardinal) +loud+ dollars!"

# With pauses
text = "[First](as: ordinal) ...c [second](as: ordinal) ...c [third](as: ordinal)!"

# With emphasis
text = "The winner is *[1](as: ordinal)*!"
```

See `examples/say_as_demo.py` for comprehensive examples.

#### 4. Optimal Phoneme Length for Better Quality

When using sentence splitting, very short sentences (like "Why?" or "Go!") can produce
poor audio quality when processed individually (only 3-8 phonemes each). The
`optimal_phoneme_length` parameter batches these short segments together to provide more
context for natural prosody:

```python
# Dialogue with many short sentences
dialogue = '"Why?" she asked. "Do it!" he said. "Go!" they shouted.'

# Without batching: Each sentence separate (poor quality for short ones)
audio1, sr = tts.create(
    dialogue,
    voice="af_bella",
    split_mode="sentence"
)

# With batching: Short sentences merged for better quality
audio2, sr = tts.create(
    dialogue,
    voice="af_bella",
    split_mode="sentence",
    optimal_phoneme_length=50  # Batch until ~50 phonemes
)
```

**Usage Modes:**

**Single Target:**

```python
# Batch segments until reaching ~50 phonemes
optimal_phoneme_length=50
```

**Array Targets (Flexible):**

```python
# Try to reach 70 phonemes (ideal)
# Fall back to 50 (good) or 30 (acceptable) if needed
optimal_phoneme_length=[30, 50, 70]
```

**How It Works:**

- Merges consecutive short segments (sentences/clauses/paragraphs)
- Stops when reaching target length OR adding next would overshoot significantly (~30%
  tolerance)
- Never exceeds `max_phoneme_length` (510)
- Preserves semantic boundaries when specified

**When to Use:**

- ✅ Dialogue-heavy text with short sentences: `optimal_phoneme_length=50`
- ✅ Mixed content (short + long sentences): `optimal_phoneme_length=[30, 50]`
- ✅ Script/screenplay with terse dialogue: `optimal_phoneme_length=[40, 60]`
- ❌ Already long sentences (no batching needed)
- ❌ When you need exact sentence-by-sentence control

**Segment Metadata:**

Batched segments use range format for `sentence` field:

- Single sentence: `sentence=0` (int)
- Batched sentences: `sentence="0-2"` (string indicating sentences 0, 1, 2 were merged)

See `examples/optimal_phoneme_length_demo.py` for a complete comparison.

## Available Voices

The library includes voices across different languages and accents. The number of
available voices depends on the model source:

### HuggingFace & GitHub v1.0 (54 voices)

- **American English**: af_alloy, af_bella, af_sarah, am_adam, am_michael, etc.
- **British English**: bf_alice, bf_emma, bm_george, bm_lewis
- **Spanish**: ef_dora, em_alex
- **French**: ff_siwis
- **Japanese**: jf_alpha, jm_kumo
- **Chinese**: zf_xiaobei, zm_yunxi
- And many more...

### GitHub v1.1-zh (103 voices)

Includes all voices from v1.0 plus additional Chinese voices:

- **English voices**: af_maple, af_sol, bf_vale (confirmed working)
- **Chinese voices**: zf_001 through zf_099, zm_009 through zm_100

**Example - Using v1.1-zh with English:**

```python
tts = pykokoro.Kokoro(model_source="github", model_variant="v1.1-zh")
audio, sr = tts.create("Hello world!", voice="af_maple", lang="en-us")
```

List all available voices:

```python
voices = tts.get_voices()
print(voices)
```

## Model Sources

PyKokoro supports downloading models from multiple sources:

### HuggingFace (Default)

The default source with 54 multi-language voices:

```python
tts = pykokoro.Kokoro(
    model_source="huggingface",
    model_quality="fp32"  # fp32, fp16, q8, q8f16, q4, q4f16, uint8, uint8f16
)
```

### GitHub v1.0

54 voices with additional `fp16-gpu` optimized quality:

```python
tts = pykokoro.Kokoro(
    model_source="github",
    model_variant="v1.0",
    model_quality="fp16-gpu"  # fp32, fp16, fp16-gpu, q8
)
```

### GitHub v1.1-zh (English + Chinese)

103 voices including English and Chinese speakers:

```python
tts = pykokoro.Kokoro(
    model_source="github",
    model_variant="v1.1-zh",
    model_quality="fp32"  # Only fp32 available
)

# Use English voices from v1.1-zh
voices = tts.get_voices()  # Returns 103 voices
audio, sr = tts.create("Hello world", voice="af_maple", lang="en-us")
```

**Note:** Chinese text generation requires proper phonemization support (currently in
development).

## Model Quality Options

Available quality options vary by source:

**HuggingFace Models:**

- `fp32`: Full precision (highest quality, largest size)
- `fp16`: Half precision (good quality, smaller size)
- `q8`: 8-bit quantized (fast, small)
- `q8f16`: 8-bit with fp16 (balanced)
- `q4`: 4-bit quantized (fastest, smallest)
- `q4f16`: 4-bit with fp16 (compact)
- `uint8`: Unsigned 8-bit (compatible)
- `uint8f16`: Unsigned 8-bit with fp16

**GitHub v1.0 Models:**

- `fp32`: Full precision
- `fp16`: Half precision
- `fp16-gpu`: GPU-optimized fp16
- `q8`: 8-bit quantized

**GitHub v1.1-zh Models:**

- `fp32`: Full precision only

```python
# HuggingFace with q8
tts = pykokoro.Kokoro(model_source="huggingface", model_quality="q8")

# GitHub v1.0 with GPU-optimized fp16
tts = pykokoro.Kokoro(model_source="github", model_variant="v1.0", model_quality="fp16-gpu")
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

### Backend Configuration

Control which phonemization backend and dictionaries to use:

```python
from pykokoro import TokenizerConfig

# Default: Full dictionaries with espeak fallback (best quality)
config = TokenizerConfig(
    backend="espeak",
    load_gold=True,
    load_silver=True,
    use_espeak_fallback=True
)

# Memory-optimized: Gold dictionary only
config = TokenizerConfig(
    backend="espeak",
    load_gold=True,
    load_silver=False,  # Saves ~22-31 MB
    use_espeak_fallback=True
)

# Fastest initialization: Pure espeak
config = TokenizerConfig(
    backend="espeak",
    load_gold=False,
    load_silver=False,
    use_espeak_fallback=True
)

# Alternative backend (requires pygoruut)
config = TokenizerConfig(
    backend="goruut"
)

tokenizer = Tokenizer(config=config)
```

**Note**: `use_dictionary` parameter is deprecated. Use `load_gold` and `load_silver`
instead for finer control.

**External G2P Libraries**: You can also use external phonemization libraries like
[Misaki](https://github.com/hexgrad/misaki):

```python
from misaki import en, espeak
import pykokoro

# Misaki G2P with espeak-ng fallback
fallback = espeak.EspeakFallback(british=False)
g2p = en.G2P(trf=False, british=False, fallback=fallback)
phonemes, _ = g2p("Hello, world!")

# Generate audio from phonemes
kokoro = pykokoro.Kokoro()
samples, sample_rate = kokoro.create(
    phonemes,
    voice="af_bella",
    is_phonemes=True
)
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
