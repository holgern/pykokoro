Pipeline Stages
===============

This page describes the pipeline stage system used by PyKokoro.

Stage Overview
--------------

The default pipeline wires these stages in order:

``doc_parser -> splitter -> g2p -> phoneme_processing -> audio_generation -> audio_postprocessing``

Each stage is a small adapter class, so you can replace or disable behavior with
no-op stages.

Typical wiring
~~~~~~~~~~~~~~

.. code-block:: python

   from pykokoro import KokoroPipeline, PipelineConfig
   from pykokoro.stages.doc_parsers.plain import PlainTextDocumentParser
   from pykokoro.stages.splitters.paragraph import ParagraphSplitter

   pipe = KokoroPipeline(
       PipelineConfig(voice="af"),
       doc_parser=PlainTextDocumentParser(),
       splitter=ParagraphSplitter(),
   )
   result = pipe.run("First paragraph.\n\nSecond paragraph.")

Stage showcase example
~~~~~~~~~~~~~~~~~~~~~~

The repository includes a showcase that wires multiple pipelines side-by-side:

``examples/pipeline_stage_showcase.py``

It demonstrates:

* SSMD + phrasplit + kokorog2p + phoneme processing + audio generation + audio postprocessing
* plain text + paragraph splitter + g2p + phoneme processing + audio generation
* plain text + no-op splitter + g2p + audio generation (no phoneme or audio postprocessing)

Local model files
~~~~~~~~~~~~~~~~~

To load a local ONNX model and voices file, set ``model_path`` and
``voices_path`` in ``PipelineConfig``. The pipeline will pass them through to
the backend when building the default ONNX stages.

.. code-block:: python

   from pathlib import Path

   from pykokoro import KokoroPipeline, PipelineConfig

   config = PipelineConfig(
       voice="af_bella",
       model_path=Path("/models/kokoro.onnx"),
       voices_path=Path("/models/voices.bin.npz"),
   )
   pipe = KokoroPipeline(config)
   result = pipe.run("Hello from local files.")
