"""Pytest configuration and fixtures for pykokoro tests."""

import sys


def pytest_configure(config):
    """Print diagnostic information at test session start."""
    print("\n" + "=" * 70, file=sys.stderr)
    print("PYKOKORO TEST DIAGNOSTICS", file=sys.stderr)
    print("=" * 70, file=sys.stderr)
    print(f"Python version: {sys.version}", file=sys.stderr)

    try:
        import kokorog2p

        print(f"kokorog2p version: {kokorog2p.__version__}", file=sys.stderr)
    except Exception as e:
        print(f"kokorog2p import failed: {e}", file=sys.stderr)

    try:
        import subprocess

        result = subprocess.run(
            ["espeak-ng", "--version"], capture_output=True, text=True, timeout=5
        )
        print(f"espeak-ng version: {result.stdout.strip()}", file=sys.stderr)
    except Exception as e:
        print(f"espeak-ng check failed: {e}", file=sys.stderr)

    try:
        from pykokoro import create_tokenizer

        tokenizer = create_tokenizer()
        result = tokenizer.phonemize("test", lang="en-us")
        print(
            f"Test phonemization: 'test' -> '{result}' (len={len(result)})",
            file=sys.stderr,
        )
    except Exception as e:
        print(f"Test phonemization failed: {e}", file=sys.stderr)
        import traceback

        traceback.print_exc(file=sys.stderr)

    print("=" * 70 + "\n", file=sys.stderr)
