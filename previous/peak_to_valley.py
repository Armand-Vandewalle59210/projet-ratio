from __future__ import annotations

import numpy as np

from projet_ratio.models import Peak, PeakToValleyResult, Spectrum


def integrate_energy_window(spectrum: Spectrum, energy_min: float, energy_max: float) -> tuple[float, float]:
    """Integrate counts in an energy interval with Poisson uncertainty."""
    e1, e2 = sorted((float(energy_min), float(energy_max)))
    mask = (spectrum.energy >= e1) & (spectrum.energy <= e2)
    total = float(np.sum(spectrum.counts[mask]))
    uncertainty = float(np.sqrt(max(total, 0.0)))
    return total, uncertainty


def calculate_peak_to_valley_ratio(self) -> None:
    self.plot.enforce_valley_size(self.valley_size.value())

    lower_min, lower_max = self.plot.lower_region_values()
    upper_min, upper_max = self.plot.upper_region_values()

    result = calculate_peak_to_valley(
        self.spectrum,
        self.selected_peak,
        lower_min,
        lower_max,
        upper_min,
        upper_max,
    )

    self.last_ratio_result = result

    inverse_ratio = 1.0 / result.ratio
    inverse_ratio_uncertainty = result.ratio_uncertainty / (result.ratio ** 2)

    self.results.setText(
        f"Ratio method: Peak area / valley discontinuity\n\n"
        f"Selected peak: {result.peak.peak_energy:.3f} keV\n"
        f"Peak area: {result.peak.area:.3f} ± {result.peak.area_uncertainty:.3f}\n\n"
        f"Valley size: {self.valley_size.value():.2f} keV each\n"
        f"Valley distance from peak: {self.valley_distance.value():.2f} keV\n"
        f"Lower valley [{result.lower_min_energy:.2f}, {result.lower_max_energy:.2f}] keV: "
        f"{result.lower_counts:.3f} ± {result.lower_uncertainty:.3f}\n"
        f"Upper valley [{result.upper_min_energy:.2f}, {result.upper_max_energy:.2f}] keV: "
        f"{result.upper_counts:.3f} ± {result.upper_uncertainty:.3f}\n\n"
        f"Valley discontinuity: {result.valley_counts:.3f} ± {result.valley_uncertainty:.3f}\n"
        f"Peak-to-valley ratio Q: {result.ratio:.6g} ± {result.ratio_uncertainty:.3g}\n"
        f"Valley-to-peak ratio 1/Q: {inverse_ratio:.6g} ± {inverse_ratio_uncertainty:.3g}\n"
        f"Acceptability lower/upper ≥ 2: {result.acceptable}"
    )

