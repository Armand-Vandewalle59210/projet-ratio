from __future__ import annotations

import numpy as np

from projet_ratio.models import Peak, PeakToValleyResult, Spectrum


def integrate_energy_window(
    spectrum: Spectrum,
    energy_min: float,
    energy_max: float,
) -> tuple[float, float]:
    """Integrate counts in an energy interval with Poisson uncertainty."""
    e1, e2 = sorted((float(energy_min), float(energy_max)))
    mask = (spectrum.energy >= e1) & (spectrum.energy <= e2)
    total = float(np.sum(spectrum.counts[mask]))
    uncertainty = float(np.sqrt(max(total, 0.0)))
    return total, uncertainty


def calculate_peak_to_valley(
    spectrum: Spectrum,
    peak: Peak,
    lower_min_energy: float,
    lower_max_energy: float,
    upper_min_energy: float,
    upper_max_energy: float,
) -> PeakToValleyResult:
    """Calculate peak-area / valley-discontinuity ratio."""
    lower_counts, lower_unc = integrate_energy_window(
        spectrum,
        lower_min_energy,
        lower_max_energy,
    )
    upper_counts, upper_unc = integrate_energy_window(
        spectrum,
        upper_min_energy,
        upper_max_energy,
    )

    valley_counts = lower_counts - upper_counts
    valley_unc = float(np.sqrt(lower_unc**2 + upper_unc**2))

    if valley_counts <= 0:
        raise ValueError(
            "The valley discontinuity is not positive. Move the valley regions "
            "or check that the selected peak is appropriate."
        )
    if peak.area <= 0:
        raise ValueError("The selected peak area is not positive.")

    ratio = float(peak.area / valley_counts)
    ratio_uncertainty = ratio * np.sqrt(
        (peak.area_uncertainty / peak.area) ** 2
        + (valley_unc / valley_counts) ** 2
    )

    
    if upper_counts <= 0:
        acceptable = lower_counts > 0
    else:
        acceptable = (lower_counts / upper_counts) >= 2


    return PeakToValleyResult(
        peak=peak,
        lower_min_energy=min(lower_min_energy, lower_max_energy),
        lower_max_energy=max(lower_min_energy, lower_max_energy),
        upper_min_energy=min(upper_min_energy, upper_max_energy),
        upper_max_energy=max(upper_min_energy, upper_max_energy),
        lower_counts=lower_counts,
        lower_uncertainty=lower_unc,
        upper_counts=upper_counts,
        upper_uncertainty=upper_unc,
        valley_counts=valley_counts,
        valley_uncertainty=valley_unc,
        ratio=ratio,
        ratio_uncertainty=float(ratio_uncertainty),
        acceptable=acceptable,
    )
