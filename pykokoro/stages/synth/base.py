from __future__ import annotations

from typing import Protocol

import numpy as np

from ...types import Segment
from ..g2p.kokorog2p import PhonemeSegment


class Synthesizer(Protocol):
    def synth(
        self, items: list[PhonemeSegment], *, segments: list[Segment]
    ) -> tuple[np.ndarray, int]: ...
