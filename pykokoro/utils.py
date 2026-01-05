"""Utility functions for pykokoro - config, GPU detection, encoding, etc."""

import json
import platform
import subprocess
import sys
import warnings
from collections.abc import Callable
from pathlib import Path
from threading import Thread
from typing import Any

from platformdirs import user_cache_dir, user_config_dir

from .constants import DEFAULT_CONFIG, PROGRAM_NAME

warnings.filterwarnings("ignore")

# Default encoding for subprocess
DEFAULT_ENCODING = sys.getfilesystemencoding()


def get_user_config_path() -> Path:
    """Get path to user configuration file."""
    if platform.system() != "Windows":
        # On non-Windows, prefer ~/.config/pykokoro if it already exists
        custom_dir = Path.home() / ".config" / "pykokoro"
        if custom_dir.exists():
            config_dir = custom_dir
        else:
            config_dir = Path(
                user_config_dir(
                    "pykokoro", appauthor=False, roaming=True, ensure_exists=True
                )
            )
    else:
        config_dir = Path(
            user_config_dir(
                "pykokoro", appauthor=False, roaming=True, ensure_exists=True
            )
        )
    return config_dir / "config.json"


def get_user_cache_path(folder: str | None = None) -> Path:
    """Get path to user cache directory, optionally with a subfolder."""
    cache_dir = Path(
        user_cache_dir("pykokoro", appauthor=False, opinion=True, ensure_exists=True)
    )
    if folder:
        cache_dir = cache_dir / folder
        cache_dir.mkdir(parents=True, exist_ok=True)
    return cache_dir


def load_config() -> dict[str, Any]:
    """Load configuration from file, returning defaults if not found."""
    try:
        config_path = get_user_config_path()
        if config_path.exists():
            with open(config_path, encoding="utf-8") as f:
                user_config = json.load(f)
                # Merge with defaults to ensure all keys exist
                return {**DEFAULT_CONFIG, **user_config}
    except Exception:
        pass
    return DEFAULT_CONFIG.copy()


def save_config(config: dict[str, Any]) -> bool:
    """Save configuration to file. Returns True on success."""
    try:
        config_path = get_user_config_path()
        config_path.parent.mkdir(parents=True, exist_ok=True)
        with open(config_path, "w", encoding="utf-8") as f:
            json.dump(config, f, indent=2)
        return True
    except Exception:
        return False


def reset_config() -> dict[str, Any]:
    """Reset configuration to defaults and save."""
    save_config(DEFAULT_CONFIG)
    return DEFAULT_CONFIG.copy()


def detect_encoding(file_path: str | Path) -> str:
    """Detect the encoding of a file using chardet/charset_normalizer."""
    import chardet
    import charset_normalizer

    with open(file_path, "rb") as f:
        raw_data = f.read()

    detected_encoding = None
    for detector in (charset_normalizer, chardet):
        try:
            result = detector.detect(raw_data)
            if result and result.get("encoding"):
                detected_encoding = result["encoding"]
                break
        except Exception:
            continue

    encoding = detected_encoding if detected_encoding else "utf-8"
    return encoding.lower()


def get_gpu_info(enabled: bool = True) -> tuple[str, bool]:
    """
    Check GPU acceleration availability for ONNX runtime.

    Args:
        enabled: Whether GPU acceleration is requested

    Returns:
        Tuple of (status message, is_gpu_available)
    """
    if not enabled:
        return "GPU disabled in config. Using CPU.", False

    try:
        import onnxruntime as ort

        providers = ort.get_available_providers()

        # Check for CUDA
        if "CUDAExecutionProvider" in providers:
            return "CUDA GPU available via ONNX Runtime.", True

        # Check for CoreML (Apple)
        if "CoreMLExecutionProvider" in providers:
            return "CoreML GPU available via ONNX Runtime.", True

        # Check for DirectML (Windows)
        if "DmlExecutionProvider" in providers:
            return "DirectML GPU available via ONNX Runtime.", True

        return f"No GPU providers available. Using CPU. (Available: {providers})", False
    except ImportError:
        return "ONNX Runtime not installed. Using CPU.", False
    except Exception as e:
        return f"Error checking GPU: {e}", False


def get_device(use_gpu: bool = True) -> str:
    """
    Get the appropriate execution provider for ONNX Runtime.

    Args:
        use_gpu: Whether to attempt GPU usage

    Returns:
        Execution provider name: 'CUDAExecutionProvider',
        'CoreMLExecutionProvider', or 'CPUExecutionProvider'
    """
    if not use_gpu:
        return "CPUExecutionProvider"

    try:
        import onnxruntime as ort

        providers = ort.get_available_providers()

        # Prefer CUDA
        if "CUDAExecutionProvider" in providers:
            return "CUDAExecutionProvider"

        # CoreML for Apple
        if "CoreMLExecutionProvider" in providers:
            return "CoreMLExecutionProvider"

        # DirectML for Windows
        if "DmlExecutionProvider" in providers:
            return "DmlExecutionProvider"

    except ImportError:
        pass

    return "CPUExecutionProvider"


def create_process(
    cmd: list[str] | str,
    stdin: int | None = None,
    text: bool = True,
    capture_output: bool = False,
    suppress_output: bool = False,
) -> subprocess.Popen:
    """
    Create a subprocess with proper platform handling.

    Args:
        cmd: Command to execute (list or string)
        stdin: stdin pipe option (e.g., subprocess.PIPE)
        text: Whether to use text mode
        capture_output: Whether to capture output
        suppress_output: Suppress all output (for rich progress bars)

    Returns:
        Popen object
    """
    use_shell = isinstance(cmd, str)
    kwargs: dict[str, Any] = {
        "shell": use_shell,
        "bufsize": 1,
    }

    # Suppress output if requested (avoids rich progress interference)
    if suppress_output:
        kwargs["stdout"] = subprocess.DEVNULL
        kwargs["stderr"] = subprocess.DEVNULL
    else:
        kwargs["stdout"] = subprocess.PIPE
        kwargs["stderr"] = subprocess.STDOUT

    if text and not suppress_output:
        kwargs["text"] = True
        kwargs["encoding"] = DEFAULT_ENCODING
        kwargs["errors"] = "replace"
    elif not suppress_output:
        kwargs["text"] = False
        kwargs["bufsize"] = 0

    if stdin is not None:
        kwargs["stdin"] = stdin

    if platform.system() == "Windows":
        startupinfo = subprocess.STARTUPINFO()  # type: ignore[attr-defined]
        startupinfo.dwFlags |= subprocess.STARTF_USESHOWWINDOW  # type: ignore[attr-defined]
        startupinfo.wShowWindow = subprocess.SW_HIDE  # type: ignore[attr-defined]
        kwargs.update(
            {"startupinfo": startupinfo, "creationflags": subprocess.CREATE_NO_WINDOW}  # type: ignore[attr-defined]
        )

    proc = subprocess.Popen(cmd, **kwargs)

    # Stream output to console in real-time if not capturing or suppressing
    if proc.stdout and not capture_output and not suppress_output:

        def _stream_output(stream: Any) -> None:
            if text:
                for line in stream:
                    sys.stdout.write(line)
                    sys.stdout.flush()
            else:
                while True:
                    chunk = stream.read(4096)
                    if not chunk:
                        break
                    try:
                        sys.stdout.write(
                            chunk.decode(DEFAULT_ENCODING, errors="replace")
                        )
                        sys.stdout.flush()
                    except Exception:
                        pass
            stream.close()

        Thread(target=_stream_output, args=(proc.stdout,), daemon=True).start()

    return proc


def ensure_ffmpeg() -> bool:
    """
    Ensure ffmpeg is available, installing static-ffmpeg if needed.

    Returns:
        True if ffmpeg is available
    """
    try:
        import static_ffmpeg

        static_ffmpeg.add_paths()
        return True
    except ImportError:
        return False


def load_tts_pipeline() -> tuple[Any, Any]:
    """
    Load numpy and Kokoro ONNX TTS backend.

    Returns:
        Tuple of (numpy module, Kokoro class)
    """
    import numpy as np

    from .onnx_backend import Kokoro

    return np, Kokoro


class LoadPipelineThread(Thread):
    """Thread for loading TTS pipeline in background."""

    def __init__(self, callback: Callable[[Any, Any, str | None], None]) -> None:
        super().__init__()
        self.callback = callback

    def run(self) -> None:
        try:
            np_module, kokoro_class = load_tts_pipeline()
            self.callback(np_module, kokoro_class, None)
        except Exception as e:
            self.callback(None, None, str(e))


# Sleep prevention for long conversions
_sleep_procs: dict[str, subprocess.Popen[str] | None] = {
    "Darwin": None,
    "Linux": None,
}


def prevent_sleep_start() -> None:
    """Prevent system from sleeping during conversion."""
    system = platform.system()
    if system == "Windows":
        import ctypes

        ctypes.windll.kernel32.SetThreadExecutionState(  # type: ignore[attr-defined]
            0x80000000 | 0x00000001 | 0x00000040
        )
    elif system == "Darwin":
        _sleep_procs["Darwin"] = create_process(["caffeinate"], capture_output=True)
    elif system == "Linux":
        import shutil

        if shutil.which("systemd-inhibit"):
            _sleep_procs["Linux"] = create_process(
                [
                    "systemd-inhibit",
                    f"--who={PROGRAM_NAME}",
                    "--why=Prevent sleep during TTS conversion",
                    "--what=sleep",
                    "--mode=block",
                    "sleep",
                    "infinity",
                ],
                capture_output=True,
            )


def prevent_sleep_end() -> None:
    """Allow system to sleep again."""
    system = platform.system()
    if system == "Windows":
        import ctypes

        ctypes.windll.kernel32.SetThreadExecutionState(0x80000000)  # type: ignore[attr-defined]
    elif system in ("Darwin", "Linux"):
        proc = _sleep_procs.get(system)
        if proc is not None:
            try:
                proc.terminate()
                _sleep_procs[system] = None
            except Exception:
                pass


def sanitize_filename(name: str, max_length: int = 100) -> str:
    """
    Sanitize a string for use as a filename.

    Args:
        name: The string to sanitize
        max_length: Maximum length of the result

    Returns:
        Sanitized filename
    """
    import re

    # Remove or replace invalid characters
    sanitized = re.sub(r'[<>:"/\\|?*]', "", name)
    # Replace multiple spaces/underscores with single underscore
    sanitized = re.sub(r"[\s_]+", "_", sanitized).strip("_")
    # Truncate if needed
    if len(sanitized) > max_length:
        # Try to break at underscore
        pos = sanitized[:max_length].rfind("_")
        sanitized = sanitized[: pos if pos > 0 else max_length].rstrip("_")
    return sanitized or "output"


def format_duration(seconds: float) -> str:
    """Format seconds as HH:MM:SS."""
    h = int(seconds // 3600)
    m = int((seconds % 3600) // 60)
    s = int(seconds % 60)
    return f"{h:02d}:{m:02d}:{s:02d}"


def format_size(size_bytes: int) -> str:
    """Format bytes as human-readable size."""
    size = float(size_bytes)
    for unit in ["B", "KB", "MB", "GB"]:
        if size < 1024:
            return f"{size:.1f} {unit}"
        size /= 1024
    return f"{size:.1f} TB"


def format_chapters_range(indices: list[int], total_chapters: int) -> str:
    """
    Format chapter indices into a range string for filenames.

    Returns empty string if all chapters are selected.
    Returns "chapters_X-Y" style string for partial selection (using min-max).

    Args:
        indices: 0-based chapter indices
        total_chapters: Total number of chapters in book

    Returns:
        Range string (e.g., "chapters_1-5") or empty string if all chapters
    """
    if not indices:
        return ""

    # Check if all chapters are selected
    if len(indices) == total_chapters and set(indices) == set(range(total_chapters)):
        return ""

    # Convert to 1-based and get min/max
    min_chapter = min(indices) + 1
    max_chapter = max(indices) + 1

    if min_chapter == max_chapter:
        return f"chapters_{min_chapter}"
    return f"chapters_{min_chapter}-{max_chapter}"


def format_filename_template(
    template: str,
    book_title: str = "",
    author: str = "",
    chapter_title: str = "",
    chapter_num: int = 0,
    input_stem: str = "",
    chapters_range: str = "",
    default_title: str = "Untitled",
    max_length: int = 100,
) -> str:
    """
    Format a filename template with the given variables.

    All values are sanitized before substitution.
    Falls back to input_stem or default_title if book_title is empty.

    Template variables:
        {book_title} - Sanitized book title
        {author} - Sanitized author name
        {chapter_title} - Sanitized chapter title
        {chapter_num} - Chapter number (1-based), supports format specs
        {input_stem} - Original input filename without extension
        {chapters_range} - Chapter range string (e.g., "chapters_1-5") or empty

    Args:
        template: Python format string (e.g., "{book_title}_{chapter_num:03d}")
        book_title: Book title from metadata
        author: Author name from metadata
        chapter_title: Chapter title
        chapter_num: 1-based chapter number
        input_stem: Original input filename without extension
        chapters_range: Chapter range string or empty
        default_title: Fallback title if book_title is empty
        max_length: Maximum length of final filename

    Returns:
        Formatted and sanitized filename (without extension)

    Examples:
        >>> format_filename_template("{book_title}", book_title="My Book")
        'My_Book'
        >>> format_filename_template(
        ...     "{chapter_num:03d}_{chapter_title}",
        ...     chapter_num=1,
        ...     chapter_title="Intro",
        ... )
        '001_Intro'
        >>> format_filename_template(
        ...     "{author}_{book_title}",
        ...     author="John Doe",
        ...     book_title="",
        ... )
        'John_Doe_Untitled'
    """
    # Determine effective book title with fallback
    effective_title = book_title.strip() if book_title else ""
    if not effective_title:
        effective_title = input_stem.strip() if input_stem else default_title

    # Sanitize all string values (but don't truncate yet - do that at the end)
    safe_book_title = sanitize_filename(effective_title, max_length=200)
    safe_author = sanitize_filename(author, max_length=100) if author else ""
    safe_chapter_title = (
        sanitize_filename(chapter_title, max_length=100) if chapter_title else ""
    )
    safe_input_stem = (
        sanitize_filename(input_stem, max_length=100) if input_stem else ""
    )
    safe_chapters_range = (
        sanitize_filename(chapters_range, max_length=50) if chapters_range else ""
    )

    # Build the format kwargs
    format_kwargs = {
        "book_title": safe_book_title,
        "author": safe_author,
        "chapter_title": safe_chapter_title,
        "chapter_num": chapter_num,
        "input_stem": safe_input_stem,
        "chapters_range": safe_chapters_range,
    }

    try:
        result = template.format(**format_kwargs)
    except KeyError:
        # Unknown template variable - fall back to book title
        result = safe_book_title
    except ValueError:
        # Invalid format spec - fall back to book title
        result = safe_book_title

    # Final sanitization and truncation
    result = sanitize_filename(result, max_length=max_length)

    # Ensure we have something
    return result or default_title
