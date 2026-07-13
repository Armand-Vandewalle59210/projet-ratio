from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np


@dataclass(slots=True)
class Spectrum:
    """Normalized spectrum representation used by the GUI and analysis code."""

    counts: np.ndarray
    channels: np.ndarray
    energy: np.ndarray
    live_time: float | None = None
    path: Path | None = None
    calibration_coefficients: tuple[float, ...] | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def name(self) -> str:
        return self.path.name if self.path is not None else "Untitled spectrum"


@dataclass(slots=True)
class Peak:
    """Fitted peak returned by the Mariscotti workflow."""

    index: int
    peak_channel: float
    peak_energy: float
    area: float
    area_uncertainty: float
    amplitude: float
    amplitude_uncertainty: float
    sigma: float
    fwhm_channels: float
    fwhm_energy: float
    left_channel: float
    right_channel: float
    left_energy: float
    right_energy: float


@dataclass(slots=True)
class PeakToValleyResult:
    peak: Peak
    lower_min_energy: float
    lower_max_energy: float
    upper_min_energy: float
    upper_max_energy: float
    lower_counts: float
    lower_uncertainty: float
    upper_counts: float
    upper_uncertainty: float
    valley_counts: float
    valley_uncertainty: float
    ratio: float
    ratio_uncertainty: float
    acceptable: bool
