#!/usr/bin/env python3
"""
Minimal pipeline example: G2P + ONNX only (no SSMD, no splitter).

This example wires a custom document parser and splitter so the pipeline:
- skips SSMD parsing
- produces a single segment for the full paragraph
- runs G2P and ONNX synthesis only

Usage:
    python examples/pipeline_g2p_onnx_minimal.py

Output:
    pipeline_g2p_onnx_minimal.wav
"""

from pykokoro import GenerationConfig, KokoroPipeline, PipelineConfig
from pykokoro.stages.base import DocumentResult
from pykokoro.stages.splitters.phrasplit import PhrasplitSplitter
from pykokoro.types import Segment, Trace


class PlainDocumentParser:
    def parse(self, text: str, cfg: PipelineConfig, trace: Trace) -> DocumentResult:
        return DocumentResult(clean_text=text)


class SingleSegmentSplitter(PhrasplitSplitter):
    def split(self, doc: DocumentResult, cfg: PipelineConfig, trace: Trace):
        text = doc.clean_text
        return [
            Segment(
                id="seg_0",
                text=text,
                char_start=0,
                char_end=len(text),
                paragraph_idx=0,
                sentence_idx=0,
            )
        ]


def main() -> None:
    text = (
        "This paragraph is synthesized without SSMD parsing or sentence splitting. "
        "The pipeline uses a single segment for the full text and runs only G2P "
        "and ONNX synthesis."
    )
    text = (
        "'That's ridiculous!' I protested. 'I'm not gonna stand here and "
        "let you insult me! What's your problem anyway?'"
    )

    cfg = PipelineConfig(
        voice="af",
        generation=GenerationConfig(lang="en-us"),
        return_trace=True,
    )
    pipeline = KokoroPipeline(
        cfg,
        doc_parser=PlainDocumentParser(),
        splitter=SingleSegmentSplitter(),
    )
    result = pipeline.run(text)
    output_path = "pipeline_g2p_onnx_minimal.wav"
    result.save_wav(output_path)
    print(f"Wrote {output_path}")

    trace = result.trace
    if trace is not None:
        if trace.warnings:
            print("Warnings:")
            for warning in trace.warnings:
                print(f"- {warning}")
        if trace.events:
            print("Trace events:")
            for event in trace.events:
                print(f"- {event.stage}:{event.name} {event.ms:.2f}ms")

    doc = pipeline.doc_parser.parse(text, cfg, Trace())
    segments = pipeline.splitter.split(doc, cfg, Trace())
    print(segments)
    phoneme_segments = pipeline.g2p.phonemize(segments, doc, cfg, Trace())
    print(f"Text: {text}")
    print("Phonemes:")
    for segment in phoneme_segments:
        print(segment.phonemes)


if __name__ == "__main__":
    main()
