#!/usr/bin/env python3
"""
Benchmark different ONNX Runtime providers.

This script compares CPU, CUDA, OpenVINO, DirectML, and CoreML performance
for TTS generation. It helps you determine which provider gives the best
performance on your system.

Usage:
    python examples/gpu_benchmark.py

Output:
    Performance comparison of available providers
"""

import time

import soundfile as sf

import pykokoro

# Test text - medium length to get meaningful timings
TEXT = (
    """
The quick brown fox jumps over the lazy dog.
She sells seashells by the seashore.
Peter Piper picked a peck of pickled peppers.
How much wood would a woodchuck chuck if a woodchuck could chuck wood?
"""
    * 5
)  # Repeat for longer audio


def benchmark_provider(provider_name: str) -> tuple[float, float] | None:
    """
    Benchmark a specific provider.

    Args:
        provider_name: Name of the provider to test

    Returns:
        Tuple of (elapsed_time, real_time_factor) or None if failed
    """
    try:
        print(f"\n{'=' * 60}")
        print(f"Benchmarking: {provider_name.upper()}")
        print(f"{'=' * 60}")

        # Initialize with specific provider
        kokoro = pykokoro.Kokoro(provider=provider_name)

        # Warmup run (important for GPU providers)
        print("Running warmup...")
        kokoro.create(TEXT[:100], voice="af_bella")

        # Actual benchmark
        print("Running benchmark...")
        start = time.time()
        samples, sr = kokoro.create(TEXT, voice="af_bella", speed=1.0)
        elapsed = time.time() - start

        # Calculate metrics
        audio_duration = len(samples) / sr
        rtf = elapsed / audio_duration  # Real-time factor

        print(f"✓ Generated {audio_duration:.2f}s of audio in {elapsed:.2f}s")
        print(f"  Real-time factor: {rtf:.2f}x")
        print(f"  Actual providers: {kokoro._session.get_providers()}")

        # Save sample output
        output_file = f"benchmark_{provider_name}.wav"
        sf.write(output_file, samples, sr)
        print(f"  Saved sample to: {output_file}")

        kokoro.close()
        return elapsed, rtf

    except Exception as e:
        print(f"✗ {provider_name.upper()} failed: {e}")
        return None


def main():
    """Run benchmarks for all available providers."""
    print("=" * 60)
    print("PYKOKORO PROVIDER BENCHMARK")
    print("=" * 60)
    print(f"\nTest text length: {len(TEXT)} characters")
    print("Voice: af_bella")
    print("Speed: 1.0")

    # Test all providers
    providers_to_test = ["cpu", "cuda", "openvino", "directml", "coreml"]
    results = {}

    for provider in providers_to_test:
        result = benchmark_provider(provider)
        if result:
            elapsed, rtf = result
            results[provider] = (elapsed, rtf)

    # Print summary
    print(f"\n{'=' * 60}")
    print("SUMMARY")
    print(f"{'=' * 60}")

    if not results:
        print("No providers were successfully tested!")
        print("\nTip: Install GPU/accelerator packages:")
        print("  pip install pykokoro[gpu]       # NVIDIA CUDA")
        print("  pip install pykokoro[openvino]  # Intel OpenVINO")
        print("  pip install pykokoro[directml]  # Windows DirectML")
        print("  pip install pykokoro[coreml]    # macOS CoreML")
        return

    # Sort by elapsed time (fastest first)
    sorted_results = sorted(results.items(), key=lambda x: x[1][0])

    print(f"\n{'Provider':<15} {'Time (s)':>10} {'RTF':>8} {'Speedup':>10}")
    print("-" * 60)

    baseline_time = sorted_results[-1][1][0]  # Slowest (usually CPU)

    for provider, (elapsed, rtf) in sorted_results:
        speedup = baseline_time / elapsed
        print(f"{provider.upper():<15} {elapsed:>10.2f} {rtf:>8.2f}x {speedup:>9.1f}x")

    # Recommendations
    print(f"\n{'=' * 60}")
    print("RECOMMENDATIONS")
    print(f"{'=' * 60}")

    fastest = sorted_results[0]
    print(f"\nFastest provider: {fastest[0].upper()}")
    print(f"  Time: {fastest[1][0]:.2f}s (RTF: {fastest[1][1]:.2f}x)")

    if fastest[1][1] < 0.5:
        print("\n✓ Excellent performance! Real-time factor < 0.5x")
    elif fastest[1][1] < 1.0:
        print("\n✓ Good performance! Real-time factor < 1.0x")
    else:
        print("\n⚠ Performance could be improved. Consider using a GPU accelerator.")

    print("\nTo use the fastest provider in your code:")
    print(f'  kokoro = pykokoro.Kokoro(provider="{fastest[0]}")')


if __name__ == "__main__":
    main()
