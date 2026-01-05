"""Test ONNX Runtime provider selection."""

import pytest
import onnxruntime as rt

from pykokoro import Kokoro


def test_cpu_provider():
    """Test CPU provider always works."""
    kokoro = Kokoro(provider="cpu")
    # Trigger initialization
    kokoro._init_kokoro()
    assert "CPUExecutionProvider" in kokoro._session.get_providers()
    kokoro.close()


def test_auto_provider_cpu():
    """Test auto provider defaults to CPU when no GPU."""
    kokoro = Kokoro(provider="auto", use_gpu=False)
    kokoro._init_kokoro()
    providers = kokoro._session.get_providers()
    assert "CPUExecutionProvider" in providers
    kokoro.close()


def test_default_provider():
    """Test default provider is CPU."""
    kokoro = Kokoro()
    kokoro._init_kokoro()
    assert "CPUExecutionProvider" in kokoro._session.get_providers()
    kokoro.close()


@pytest.mark.skipif(
    "CUDAExecutionProvider" not in rt.get_available_providers(),
    reason="CUDA not available",
)
def test_cuda_provider():
    """Test CUDA provider when available."""
    kokoro = Kokoro(provider="cuda")
    kokoro._init_kokoro()
    assert "CUDAExecutionProvider" in kokoro._session.get_providers()
    kokoro.close()


@pytest.mark.skipif(
    "OpenVINOExecutionProvider" not in rt.get_available_providers(),
    reason="OpenVINO not available",
)
def test_openvino_provider():
    """Test OpenVINO provider when available.

    Note: OpenVINO is currently incompatible with Kokoro models due to
    dynamic rank tensors, so we expect this to fail with a RuntimeError.
    """
    kokoro = Kokoro(provider="openvino")
    with pytest.raises(RuntimeError, match="Failed to initialize ONNX session"):
        kokoro._init_kokoro()
    kokoro.close()


@pytest.mark.skipif(
    "DmlExecutionProvider" not in rt.get_available_providers(),
    reason="DirectML not available",
)
def test_directml_provider():
    """Test DirectML provider when available."""
    kokoro = Kokoro(provider="directml")
    kokoro._init_kokoro()
    assert "DmlExecutionProvider" in kokoro._session.get_providers()
    kokoro.close()


@pytest.mark.skipif(
    "CoreMLExecutionProvider" not in rt.get_available_providers(),
    reason="CoreML not available",
)
def test_coreml_provider():
    """Test CoreML provider when available."""
    kokoro = Kokoro(provider="coreml")
    kokoro._init_kokoro()
    assert "CoreMLExecutionProvider" in kokoro._session.get_providers()
    kokoro.close()


def test_invalid_provider():
    """Test that invalid provider raises error."""
    kokoro = Kokoro(provider="invalid")  # type: ignore
    with pytest.raises(ValueError, match="Unknown provider"):
        kokoro._init_kokoro()


def test_unavailable_provider_helpful_error():
    """Test that requesting unavailable provider gives helpful error."""
    available = rt.get_available_providers()

    # Find a provider that's NOT available
    test_providers = {
        "cuda": "CUDAExecutionProvider",
        "openvino": "OpenVINOExecutionProvider",
        "directml": "DmlExecutionProvider",
        "coreml": "CoreMLExecutionProvider",
    }

    for name, provider_name in test_providers.items():
        if provider_name not in available:
            kokoro = Kokoro(provider=name)  # type: ignore
            with pytest.raises(
                RuntimeError,
                match=f"{name.upper()} provider requested but not available",
            ):
                kokoro._init_kokoro()
            return

    pytest.skip("All providers are available on this system")


def test_backward_compatibility_use_gpu_false():
    """Test legacy use_gpu=False parameter still works."""
    kokoro = Kokoro(use_gpu=False)
    kokoro._init_kokoro()
    assert "CPUExecutionProvider" in kokoro._session.get_providers()
    kokoro.close()


def test_backward_compatibility_use_gpu_true():
    """Test legacy use_gpu=True parameter still works."""
    # This should auto-select a GPU provider if available, or fall back to CPU
    kokoro = Kokoro(use_gpu=True)
    kokoro._init_kokoro()
    providers = kokoro._session.get_providers()
    assert "CPUExecutionProvider" in providers
    # At least CPU should be available
    kokoro.close()


def test_env_override(monkeypatch):
    """Test ONNX_PROVIDER environment variable override."""
    monkeypatch.setenv("ONNX_PROVIDER", "CPUExecutionProvider")
    kokoro = Kokoro(provider="cpu")  # Even with explicit provider, env wins
    kokoro._init_kokoro()
    assert "CPUExecutionProvider" in kokoro._session.get_providers()
    kokoro.close()


def test_provider_logging(caplog):
    """Test that provider selection is logged."""
    import logging

    caplog.set_level(logging.INFO)
    kokoro = Kokoro(provider="cpu")
    kokoro._init_kokoro()
    assert any("CPU" in record.message for record in caplog.records)
    kokoro.close()


def test_deprecation_warning_use_gpu(caplog):
    """Test that use_gpu shows deprecation warning."""
    import logging

    caplog.set_level(logging.WARNING)
    kokoro = Kokoro(use_gpu=True)
    assert any("deprecated" in record.message for record in caplog.records)
    kokoro.close()


def test_provider_with_inference():
    """Test that provider selection works with actual inference."""
    kokoro = Kokoro(provider="cpu")
    samples, sr = kokoro.create("Hello world", voice="af_bella")
    assert samples is not None
    assert len(samples) > 0
    assert sr == 24000
    kokoro.close()
