"""Audio silence trimming utilities.

Copyright (c) 2013--2023, librosa development team.

Permission to use, copy, modify, and/or distribute this software for any
purpose with or without fee is hereby granted, provided that the above
copyright notice and this permission notice appear in all copies.

THE SOFTWARE IS PROVIDED "AS IS" AND THE AUTHOR DISCLAIMS ALL WARRANTIES WITH
REGARD TO THIS SOFTWARE INCLUDING ALL IMPLIED WARRANTIES OF MERCHANTABILITY
AND FITNESS. IN NO EVENT SHALL THE AUTHOR BE LIABLE FOR ANY SPECIAL, DIRECT,
INDIRECT, OR CONSEQUENTIAL DAMAGES OR ANY DAMAGES WHATSOEVER RESULTING FROM
LOSS OF USE, DATA OR PROFITS, WHETHER IN AN ACTION OF CONTRACT, NEGLIGENCE OR
OTHER TORTIOUS ACTION, ARISING OUT OF OR IN CONNECTION WITH THE USE OR
PERFORMANCE OF THIS SOFTWARE.

This file is extracted from the librosa package since we only use the trim()
function and librosa requires many dependencies.

Reference:
    - https://gist.github.com/evq/82e95a363eeeb75d15dd62abc1eb1bde
    - https://github.com/librosa/librosa/blob/894942673d55aa2206df1296b6c4c50827c7f1d6/librosa/effects.py#L612
"""

from __future__ import annotations

import warnings
from collections.abc import Callable

import numpy as np
from numpy.lib.stride_tricks import as_strided
from numpy.typing import NDArray


class TrimError(Exception):
    """Base exception for trim operations."""

    pass


class ParameterError(TrimError):
    """Exception for invalid parameters."""

    pass


def _cabs2(x: np.ndarray) -> np.ndarray:
    """Efficiently compute abs2 on complex inputs."""
    return x.real**2 + x.imag**2  # type: ignore[union-attr]


def abs2(x: np.ndarray, dtype: np.dtype | None = None) -> np.ndarray:
    """Compute the squared magnitude of a real or complex array.

    This function is equivalent to calling `np.abs(x)**2` but it
    is slightly more efficient.

    Args:
        x: Input data, either real or complex typed
        dtype: Output data type

    Returns:
        Squared magnitude of x
    """
    if np.iscomplexobj(x):
        y = _cabs2(x)
        if dtype is None:
            return y
        return y.astype(dtype)
    return np.square(x, dtype=dtype)


def amplitude_to_db(
    S: np.ndarray,
    *,
    ref: float | Callable[[np.ndarray], float] = 1.0,
    amin: float = 1e-5,
    top_db: float | None = 80.0,
) -> np.ndarray:
    """Convert an amplitude spectrogram to dB-scaled spectrogram.

    Args:
        S: Input amplitude
        ref: Reference amplitude. If callable, computed as ref(S)
        amin: Minimum threshold for S and ref
        top_db: Threshold the output at top_db below the peak

    Returns:
        S measured in dB
    """
    S = np.asarray(S)

    if np.issubdtype(S.dtype, np.complexfloating):
        warnings.warn(
            "amplitude_to_db was called on complex input so phase "
            "information will be discarded. To suppress this warning, "
            "call amplitude_to_db(np.abs(S)) instead.",
            stacklevel=2,
        )

    magnitude = np.abs(S)

    if callable(ref):
        ref_value = ref(magnitude)
    else:
        ref_value = np.abs(ref)

    out_array = magnitude if isinstance(magnitude, np.ndarray) else None
    power = np.square(magnitude, out=out_array)

    db = power_to_db(power, ref=ref_value**2, amin=amin**2, top_db=top_db)
    return db


def power_to_db(
    S: np.ndarray,
    *,
    ref: float | Callable[[np.ndarray], float] = 1.0,
    amin: float = 1e-10,
    top_db: float | None = 80.0,
) -> np.ndarray:
    """Convert a power spectrogram to decibel (dB) units.

    This computes 10 * log10(S / ref) in a numerically stable way.

    Args:
        S: Input power
        ref: Reference power. If callable, computed as ref(S)
        amin: Minimum threshold for abs(S) and ref
        top_db: Threshold the output at top_db below the peak

    Returns:
        S measured in dB
    """
    S = np.asarray(S)

    if amin <= 0:
        raise ParameterError("amin must be strictly positive")

    if np.issubdtype(S.dtype, np.complexfloating):
        warnings.warn(
            "power_to_db was called on complex input so phase "
            "information will be discarded. To suppress this warning, "
            "call power_to_db(np.abs(D)**2) instead.",
            stacklevel=2,
        )
        magnitude = np.abs(S)
    else:
        magnitude = S

    if callable(ref):
        ref_value = ref(magnitude)
    else:
        ref_value = np.abs(ref)

    log_spec = 10.0 * np.log10(np.maximum(amin, magnitude))
    log_spec -= 10.0 * np.log10(np.maximum(amin, ref_value))

    if top_db is not None:
        if top_db < 0:
            raise ParameterError("top_db must be non-negative")
        log_spec = np.maximum(log_spec, log_spec.max() - top_db)

    return log_spec


def frame(
    x: np.ndarray,
    *,
    frame_length: int,
    hop_length: int,
    axis: int = -1,
    writeable: bool = False,
    subok: bool = False,
) -> np.ndarray:
    """Slice a data array into (overlapping) frames.

    This implementation uses low-level stride manipulation to avoid
    making a copy of the data.

    Args:
        x: Array to frame
        frame_length: Length of the frame
        hop_length: Number of steps to advance between frames
        axis: The axis along which to frame
        writeable: If True, the framed view is read-write
        subok: If True, sub-classes will be passed-through

    Returns:
        A framed view of x
    """
    x = np.array(x, copy=False, subok=subok)

    if x.shape[axis] < frame_length:
        raise ParameterError(
            f"Input is too short (n={x.shape[axis]}) for frame_length={frame_length}"
        )

    if hop_length < 1:
        raise ParameterError(f"Invalid hop_length: {hop_length}")

    # Put new within-frame axis at the end
    out_strides = x.strides + (x.strides[axis],)

    # Reduce the shape on the framing axis
    x_shape_trimmed = list(x.shape)
    x_shape_trimmed[axis] -= frame_length - 1

    out_shape = tuple(x_shape_trimmed) + (frame_length,)
    xw = as_strided(
        x, strides=out_strides, shape=out_shape, subok=subok, writeable=writeable
    )

    if axis < 0:
        target_axis = axis - 1
    else:
        target_axis = axis + 1

    xw = np.moveaxis(xw, -1, target_axis)

    # Downsample along the target axis
    slices = [slice(None)] * xw.ndim
    slices[axis] = slice(0, None, hop_length)
    return xw[tuple(slices)]


def frames_to_samples(
    frames: np.ndarray | int,
    *,
    hop_length: int = 512,
    n_fft: int | None = None,
) -> np.ndarray | int:
    """Convert frame indices to audio sample indices.

    Args:
        frames: Frame index or vector of frame indices
        hop_length: Number of samples between successive frames
        n_fft: Optional FFT window length for offset calculation

    Returns:
        Sample index for each given frame number
    """
    offset = 0
    if n_fft is not None:
        offset = int(n_fft // 2)

    return (np.asanyarray(frames) * hop_length + offset).astype(int)


def rms(
    *,
    y: np.ndarray | None = None,
    S: np.ndarray | None = None,
    frame_length: int = 2048,
    hop_length: int = 512,
    center: bool = True,
    pad_mode: str = "constant",
    dtype: type = np.float32,
) -> np.ndarray:
    """Compute root-mean-square (RMS) value for each frame.

    Args:
        y: Audio time series
        S: Spectrogram magnitude
        frame_length: Length of analysis frame
        hop_length: Hop length for STFT
        center: If True, pad the signal by frame_length//2 on either side
        pad_mode: Padding mode for centered analysis
        dtype: Data type of the output array

    Returns:
        RMS value for each frame
    """
    if y is not None:
        if center:
            padding = [(0, 0) for _ in range(y.ndim)]
            padding[-1] = (int(frame_length // 2), int(frame_length // 2))
            y = np.pad(y, padding, mode=pad_mode)  # type: ignore[arg-type]

        x = frame(y, frame_length=frame_length, hop_length=hop_length)  # type: ignore[arg-type]

        # Calculate power
        power = np.mean(abs2(x, dtype=dtype), axis=-2, keepdims=True)  # type: ignore[arg-type]
    elif S is not None:
        # Check the frame length
        if S.shape[-2] != frame_length // 2 + 1:
            expected_a = S.shape[-2] * 2 - 2
            expected_b = S.shape[-2] * 2 - 1
            raise ParameterError(
                f"Since S.shape[-2] is {S.shape[-2]}, "
                f"frame_length is expected to be {expected_a} or {expected_b}; "
                f"found {frame_length}"
            )

        # Power spectrogram
        x = abs2(S, dtype=dtype)  # type: ignore[arg-type]

        # Adjust the DC and sr/2 component
        x[..., 0, :] *= 0.5
        if frame_length % 2 == 0:
            x[..., -1, :] *= 0.5

        # Calculate power
        power = 2 * np.sum(x, axis=-2, keepdims=True) / frame_length**2
    else:
        raise ParameterError("Either `y` or `S` must be input.")

    return np.sqrt(power)


def _signal_to_frame_nonsilent(
    y: np.ndarray,
    frame_length: int = 2048,
    hop_length: int = 512,
    top_db: float = 60,
    ref: Callable[[np.ndarray], float] | float = np.max,
    aggregate: Callable[[np.ndarray], float] = np.max,
) -> np.ndarray:
    """Frame-wise non-silent indicator for audio input.

    Args:
        y: Audio signal, mono or stereo
        frame_length: Number of samples per frame
        hop_length: Number of samples between frames
        top_db: Threshold below reference to consider as silence
        ref: Reference amplitude
        aggregate: Function to aggregate dB measurements across channels

    Returns:
        Indicator of non-silent frames
    """
    # Compute the MSE for the signal
    mse = rms(y=y, frame_length=frame_length, hop_length=hop_length)

    # Convert to decibels and slice out the mse channel
    db = amplitude_to_db(mse[..., 0, :], ref=ref, top_db=None)

    # Aggregate everything but the time dimension
    if db.ndim > 1:
        db = np.apply_over_axes(aggregate, db, range(db.ndim - 1))  # type: ignore[arg-type]
        # Squeeze out leading singleton dimensions
        db = np.squeeze(db, axis=tuple(range(db.ndim - 1)))

    return db > -top_db


def trim(
    y: np.ndarray,
    *,
    top_db: float = 60,
    ref: float | Callable[[np.ndarray], float] = np.max,
    frame_length: int = 2048,
    hop_length: int = 512,
    aggregate: Callable[[np.ndarray], float] = np.max,
) -> tuple[np.ndarray, NDArray[np.intp]]:
    """Trim leading and trailing silence from an audio signal.

    Silence is defined as segments of the audio signal that are `top_db`
    decibels (or more) quieter than a reference level.

    Args:
        y: Audio signal. Multi-channel is supported.
        top_db: Threshold (in decibels) below reference to consider as silence
        ref: Reference amplitude. By default, uses np.max
        frame_length: Number of samples per analysis frame
        hop_length: Number of samples between analysis frames
        aggregate: Function to aggregate across channels

    Returns:
        Tuple of:
        - y_trimmed: The trimmed signal
        - index: The interval of y corresponding to the non-silent region
    """
    non_silent = _signal_to_frame_nonsilent(
        y,
        frame_length=frame_length,
        hop_length=hop_length,
        ref=ref,
        top_db=top_db,
        aggregate=aggregate,
    )

    nonzero = np.flatnonzero(non_silent)

    if nonzero.size > 0:
        # Compute the start and end positions
        start = int(frames_to_samples(nonzero[0], hop_length=hop_length))
        end = min(
            y.shape[-1],
            int(frames_to_samples(nonzero[-1] + 1, hop_length=hop_length)),
        )
    else:
        # The entire signal is trimmed: nothing is above the threshold
        start, end = 0, 0

    # Slice the buffer and return the corresponding interval
    return y[..., start:end], np.asarray([start, end])
