import numpy as np

from projet_ratio.models import Peak, Spectrum
from projet_ratio.analysis.peak_to_valley import calculate_peak_to_valley


def test_peak_to_valley_basic():
    energy = np.arange(0, 100, dtype=float)
    counts = np.ones_like(energy) * 10
    counts[(energy >= 30) & (energy <= 39)] = 30
    counts[(energy >= 60) & (energy <= 69)] = 10

    spectrum = Spectrum(counts=counts, channels=energy.copy(), energy=energy)
    peak = Peak(
        index=1,
        peak_channel=50,
        peak_energy=50,
        area=1000,
        area_uncertainty=30,
        amplitude=100,
        amplitude_uncertainty=5,
        sigma=2,
        fwhm_channels=4.7,
        fwhm_energy=4.7,
        left_channel=47.5,
        right_channel=52.5,
        left_energy=47.5,
        right_energy=52.5,
    )

    result = calculate_peak_to_valley(spectrum, peak, 30, 39, 60, 69)
    assert result.lower_counts == 300
    assert result.upper_counts == 100
    assert result.valley_counts == 200
    assert result.ratio == 5
