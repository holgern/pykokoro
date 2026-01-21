from __future__ import annotations

import warnings


def deprecated(old: str, new: str) -> None:
    warnings.warn(f"{old} is deprecated; use {new}", DeprecationWarning, stacklevel=3)
