
from __future__ import annotations

from dataclasses import dataclass
import numpy as np

from projet_ratio.models import Peak, Spectrum


@dataclass(slots=True)
class RatioResult:
    method: str
    numerator_name: str
    denominator_name: str
    numerator: float
    numerator_uncertainty: float
    denominator: float
    denominator_uncertainty: float
    ratio: float
    ratio_uncertainty: float


def propagate_ratio(
    numerator: float,
    numerator_uncertainty: float,
    denominator: float,
    denominator_uncertainty: float,
) -> tuple[float, float]:
    if denominator == 0:
        raise ValueError("Denominator is zero.")

    if numerator == 0:
        raise ValueError("Numerator is zero.")

    ratio = numerator / denominator

    ratio_uncertainty = abs(ratio) * np.sqrt(
        (numerator_uncertainty / numerator) ** 2
        + (denominator_uncertainty / denominator) ** 2
    )

    return float(ratio), float(ratio_uncertainty)


def counts_at_energy(spectrum: Spectrum, energy: float) -> tuple[float, float]:
    """Interpolate counts at an energy cursor."""
    counts = float(np.interp(float(energy), spectrum.energy, spectrum.counts))
    uncertainty = float(np.sqrt(max(counts, 0.0)))
    return counts, uncertainty


def integrate_energy_window(
    spectrum: Spectrum,
    energy_min: float,
    energy_max: float,
) -> tuple[float, float]:
    e1, e2 = sorted((float(energy_min), float(energy_max)))
    mask = (spectrum.energy >= e1) & (spectrum.energy <= e2)

    total = float(np.sum(spectrum.counts[mask]))
    uncertainty = float(np.sqrt(max(total, 0.0)))

    return total, uncertainty


def peak_height_to_compton_height(
    spectrum: Spectrum,
    peak: Peak,
    compton_energy: float,
) -> RatioResult:
    peak_height = float(peak.amplitude)
    peak_height_uncertainty = float(peak.amplitude_uncertainty)

    if not np.isfinite(peak_height_uncertainty):
        peak_height_uncertainty = np.sqrt(max(peak_height, 0.0))

    compton_height, compton_uncertainty = counts_at_energy(
        spectrum,
        compton_energy,
    )

    ratio, ratio_uncertainty = propagate_ratio(
        peak_height,
        peak_height_uncertainty,
        compton_height,
        compton_uncertainty,
    )

    return RatioResult(
        method="Peak height / Compton height",
        numerator_name="Peak height",
        denominator_name="Compton height",
        numerator=peak_height,
        numerator_uncertainty=peak_height_uncertainty,
        denominator=compton_height,
        denominator_uncertainty=compton_uncertainty,
        ratio=ratio,
        ratio_uncertainty=ratio_uncertainty,
    )


def peak_area_to_compton_area(
    spectrum: Spectrum,
    peak: Peak,
    compton_min_energy: float,
    compton_max_energy: float,
) -> RatioResult:
    peak_area = float(peak.area)
    peak_area_uncertainty = float(peak.area_uncertainty)

    if not np.isfinite(peak_area_uncertainty):
        peak_area_uncertainty = np.sqrt(max(peak_area, 0.0))

    compton_area, compton_uncertainty = integrate_energy_window(
        spectrum,
        compton_min_energy,
        compton_max_energy,
    )

    ratio, ratio_uncertainty = propagate_ratio(
        peak_area,
        peak_area_uncertainty,
        compton_area,
        compton_uncertainty,
    )

    return RatioResult(
        method="Peak area / Compton ROI area",
        numerator_name="Peak area",
        denominator_name="Compton ROI area",
        numerator=peak_area,
        numerator_uncertainty=peak_area_uncertainty,
        denominator=compton_area,
        denominator_uncertainty=compton_uncertainty,
        ratio=ratio,
        ratio_uncertainty=ratio_uncertainty,
    )


def peak_height_to_background_height(
    spectrum: Spectrum,
    peak: Peak,
    left_background_min: float,
    left_background_max: float,
    right_background_min: float,
    right_background_max: float,
) -> RatioResult:
    peak_height = float(peak.amplitude)
    peak_height_uncertainty = float(peak.amplitude_uncertainty)

    left_counts, left_unc = integrate_energy_window(
        spectrum,
        left_background_min,
        left_background_max,
    )
    right_counts, right_unc = integrate_energy_window(
        spectrum,
        right_background_min,
        right_background_max,
    )

    left_width = abs(left_background_max - left_background_min)
    right_width = abs(right_background_max - right_background_min)

    if left_width <= 0 or right_width <= 0:
        raise ValueError("Background ROI width must be positive.")

    left_height = left_counts / left_width
    right_height = right_counts / right_width

    background_height = 0.5 * (left_height + right_height)

    background_uncertainty = 0.5 * np.sqrt(
        (left_unc / left_width) ** 2
        + (right_unc / right_width) ** 2
    )

    ratio, ratio_uncertainty = propagate_ratio(
        peak_height,
        peak_height_uncertainty,
        background_height,
        background_uncertainty,
    )

    return RatioResult(
        method="Peak height / local background height",
        numerator_name="Peak height",
        denominator_name="Local background height",
        numerator=peak_height,
        numerator_uncertainty=peak_height_uncertainty,
        denominator=background_height,
        denominator_uncertainty=background_uncertainty,
        ratio=ratio,
        ratio_uncertainty=ratio_uncertainty,
    )


def peak_area_to_peak_area(
    peak_a: Peak,
    peak_b: Peak,
) -> RatioResult:
    ratio, ratio_uncertainty = propagate_ratio(
        peak_a.area,
        peak_a.area_uncertainty,
        peak_b.area,
        peak_b.area_uncertainty,
    )

    return RatioResult(
        method="Peak area / peak area",
        numerator_name=f"Peak A area ({peak_a.peak_energy:.3f} keV)",
        denominator_name=f"Peak B area ({peak_b.peak_energy:.3f} keV)",
        numerator=peak_a.area,
        numerator_uncertainty=peak_a.area_uncertainty,
        denominator=peak_b.area,
        denominator_uncertainty=peak_b.area_uncertainty,
        ratio=ratio,
        ratio_uncertainty=ratio_uncertainty,
    )
