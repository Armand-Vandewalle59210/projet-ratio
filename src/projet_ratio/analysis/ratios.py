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
    if numerator == 0:
        raise ValueError("Numerator is zero.")
    if denominator == 0:
        raise ValueError("Denominator is zero.")

    ratio = float(numerator / denominator)

    ratio_uncertainty = abs(ratio) * np.sqrt(
        (numerator_uncertainty / numerator) ** 2
        + (denominator_uncertainty / denominator) ** 2
    )

    return ratio, float(ratio_uncertainty)


def counts_at_energy(spectrum: Spectrum, energy: float) -> tuple[float, float]:
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


def mean_counts_in_window(
    spectrum: Spectrum,
    energy_min: float,
    energy_max: float,
) -> tuple[float, float]:
    """Mean counts in an energy window, with uncertainty on the mean."""
    e1, e2 = sorted((float(energy_min), float(energy_max)))
    mask = (spectrum.energy >= e1) & (spectrum.energy <= e2)

    selected_counts = np.asarray(spectrum.counts[mask], dtype=float)

    if selected_counts.size == 0:
        raise ValueError(
            f"No spectrum channels found in background window [{e1:.3f}, {e2:.3f}] keV."
        )

    total = float(np.sum(selected_counts))
    mean = float(np.mean(selected_counts))

    # Poisson uncertainty on total is sqrt(total).
    # Uncertainty on the mean is sqrt(total) / number of channels.
    uncertainty = float(np.sqrt(max(total, 0.0)) / selected_counts.size)

    return mean, uncertainty


def peak_height_to_compton_height(
    spectrum: Spectrum,
    peak: Peak,
    compton_energy: float,
) -> RatioResult:
    peak_height = float(peak.amplitude)
    peak_height_uncertainty = float(peak.amplitude_uncertainty)

    if not np.isfinite(peak_height_uncertainty):
        peak_height_uncertainty = float(np.sqrt(max(peak_height, 0.0)))

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
        denominator_name=f"Compton height at {compton_energy:.3f} keV",
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
        peak_area_uncertainty = float(np.sqrt(max(peak_area, 0.0)))

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
        denominator_name=f"Compton ROI area [{compton_min_energy:.3f}, {compton_max_energy:.3f}] keV",
        numerator=peak_area,
        numerator_uncertainty=peak_area_uncertainty,
        denominator=compton_area,
        denominator_uncertainty=compton_uncertainty,
        ratio=ratio,
        ratio_uncertainty=ratio_uncertainty,
    )


def peak_height_to_local_background_height(
    spectrum: Spectrum,
    peak: Peak,
    left_background_min: float,
    left_background_max: float,
    right_background_min: float,
    right_background_max: float,
) -> RatioResult:
    """Peak fitted height divided by local background height.

    The background height is the average of the mean counts in the left and
    right background ROIs.
    """
    peak_height = float(peak.amplitude)
    peak_height_uncertainty = float(peak.amplitude_uncertainty)

    if not np.isfinite(peak_height_uncertainty):
        peak_height_uncertainty = float(np.sqrt(max(peak_height, 0.0)))

    left_mean, left_uncertainty = mean_counts_in_window(
        spectrum,
        left_background_min,
        left_background_max,
    )
    right_mean, right_uncertainty = mean_counts_in_window(
        spectrum,
        right_background_min,
        right_background_max,
    )

    background_height = 0.5 * (left_mean + right_mean)
    background_uncertainty = 0.5 * np.sqrt(
        left_uncertainty**2 + right_uncertainty**2
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
        denominator_name=(
            f"Local background height "
            f"[{left_background_min:.3f}, {left_background_max:.3f}] keV and "
            f"[{right_background_min:.3f}, {right_background_max:.3f}] keV"
        ),
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
    """Area ratio between two fitted peaks."""
    if peak_a.index == peak_b.index:
        raise ValueError("Peak A and Peak B must be different peaks.")

    area_a = float(peak_a.area)
    area_b = float(peak_b.area)

    area_a_uncertainty = float(peak_a.area_uncertainty)
    area_b_uncertainty = float(peak_b.area_uncertainty)

    if not np.isfinite(area_a_uncertainty):
        area_a_uncertainty = float(np.sqrt(max(area_a, 0.0)))

    if not np.isfinite(area_b_uncertainty):
        area_b_uncertainty = float(np.sqrt(max(area_b, 0.0)))

    ratio, ratio_uncertainty = propagate_ratio(
        area_a,
        area_a_uncertainty,
        area_b,
        area_b_uncertainty,
    )

    return RatioResult(
        method="Peak area / peak area",
        numerator_name=f"Peak A area at {peak_a.peak_energy:.3f} keV",
        denominator_name=f"Peak B area at {peak_b.peak_energy:.3f} keV",
        numerator=area_a,
        numerator_uncertainty=area_a_uncertainty,
        denominator=area_b,
        denominator_uncertainty=area_b_uncertainty,
        ratio=ratio,
        ratio_uncertainty=ratio_uncertainty,
    )

def peak_height_to_peak_height(
    peak_a: Peak,
    peak_b: Peak,
) -> RatioResult:
    """Height ratio between two fitted peaks."""
    if peak_a.index == peak_b.index:
        raise ValueError("Peak A and Peak B must be different peaks.")

    height_a = float(peak_a.amplitude)
    height_b = float(peak_b.amplitude)

    height_a_uncertainty = float(peak_a.amplitude_uncertainty)
    height_b_uncertainty = float(peak_b.amplitude_uncertainty)

    if not np.isfinite(height_a_uncertainty):
        height_a_uncertainty = float(np.sqrt(max(height_a, 0.0)))

    if not np.isfinite(height_b_uncertainty):
        height_b_uncertainty = float(np.sqrt(max(height_b, 0.0)))

    ratio, ratio_uncertainty = propagate_ratio(
        height_a,
        height_a_uncertainty,
        height_b,
        height_b_uncertainty,
    )

    return RatioResult(
        method="Peak height / peak height",
        numerator_name=f"Peak A height at {peak_a.peak_energy:.3f} keV",
        denominator_name=f"Peak B height at {peak_b.peak_energy:.3f} keV",
        numerator=height_a,
        numerator_uncertainty=height_a_uncertainty,
        denominator=height_b,
        denominator_uncertainty=height_b_uncertainty,
        ratio=ratio,
        ratio_uncertainty=ratio_uncertainty,
    )
