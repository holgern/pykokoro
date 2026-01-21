# Pipeline Recon Notes

## Public exports

- `pykokoro.__init__` exports `Kokoro`, `Tokenizer`, `GenerationConfig`,
  `PhonemeSegment`, `SSMDSegment`, `SSMDMetadata`, download helpers,
  `text_to_phoneme_segments`, and utility helpers.
- `Document` from `ssmd` is re-exported in `pykokoro.__init__`.

## Synthesis path (current)

- `Kokoro.create()` in `pykokoro/onnx_backend.py` drives synthesis.
- `Kokoro._init_kokoro()` creates `OnnxSessionManager` -> ONNX Runtime session, loads
  voices via `VoiceManager`, and constructs `AudioGenerator`.
- `AudioGenerator.generate_from_segments()` consumes `phonemes.PhonemeSegment` list and
  handles voice switching, pauses, and prosody metadata.

## Text -> phoneme pipeline

- `phonemes.text_to_phoneme_segments()` is the core pipeline entry:
  - Always parses SSMD via `ssmd_parser.parse_ssmd_to_segments()`.
  - Converts SSMD segments into `phonemes.PhonemeSegment` (includes voice/prosody
    metadata).
  - Merges for `pause_mode="tts"`, cascades for max phoneme length, and optionally
    applies short-sentence phoneme pretext.

## SSMD integration

- `ssmd_parser.parse_ssmd_to_segments()` calls `ssmd.parse_sentences()` and maps
  `SSMDSegment` + `SSMDMetadata`.
- SSMD breaks create segment boundaries (pause_before/pause_after on segments).
- SSMD metadata includes language, prosody, phoneme overrides, say-as, markers, and
  voice metadata.

## Optional deps + failure modes

- spaCy: required for sentence/clause splitting in `phonemes._split_text_with_mode`;
  absent -> warning and fallback to word splitting.
- phrasplit: used in `phonemes._split_text_with_mode` for sentence/clause splitting;
  import error triggers fallback.
- kokorog2p: required by `Tokenizer` for phonemization; missing raises errors when
  phonemizing.
