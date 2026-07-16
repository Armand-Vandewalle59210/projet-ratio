from __future__ import annotations

import math
from typing import Any

import numpy as np
from scipy.optimize import curve_fit


from projet_ratio.models import Peak, Spectrum


def _moving_average(values: np.ndarray, window: int) -> np.ndarray:
    """Centered moving average with edge padding."""
    values = np.asarray(values, dtype=float)
    window = max(1, int(window))
    if window % 2 == 0:
        window += 1
    if window == 1:
        return values.copy()
    pad = window // 2
    padded = np.pad(values, pad, mode="edge")
    kernel = np.ones(window, dtype=float) / window
    return np.convolve(padded, kernel, mode="valid")


def _robust_sigma(values: np.ndarray) -> float:
    """Robust global sigma estimate from MAD."""
    values = np.asarray(values, dtype=float)
    median = np.median(values)
    mad = np.median(np.abs(values - median))
    sigma = 1.4826 * mad
    if not np.isfinite(sigma) or sigma <= 0:
        sigma = np.std(values)
    if not np.isfinite(sigma) or sigma <= 0:
        sigma = 1.0
    return float(sigma)


def mariscotti_transform(counts: np.ndarray, z: int = 5, w: int = 5) -> tuple[np.ndarray, np.ndarray]:
    """Compute the smoothed second-difference transform.

    This follows the practical implementation used in your original documented
    script: first compute the second difference, then smooth it z times with an
    odd moving-average window w.
    """
    counts = np.asarray(counts, dtype=float)
    if counts.ndim != 1:
        raise ValueError("counts must be a 1D array")
    if len(counts) < 3:
        raw = np.zeros_like(counts, dtype=float)
        return raw.copy(), raw

    raw = np.zeros_like(counts, dtype=float)
    raw[1:-1] = counts[2:] - 2.0 * counts[1:-1] + counts[:-2]

    transform = raw.copy()
    for _ in range(max(0, int(z))):
        transform = _moving_average(transform, int(w))

    return transform, raw

def _savgol_smooth_numpy(values: np.ndarray, window: int, polyorder: int) -> np.ndarray:
    """Small pure-NumPy Savitzky-Golay smoothing replacement.

    This avoids scipy.signal.savgol_filter, which caused PyInstaller/SciPy
    packaging problems, while preserving peak shapes better than a moving average.
    """
    values = np.asarray(values, dtype=float)
    n = len(values)

    if n < 5:
        return values.copy()

    window = max(5, int(window))
    if window % 2 == 0:
        window += 1

    if window >= n:
        window = n if n % 2 == 1 else n - 1

    if window < 5:
        return values.copy()

    polyorder = max(1, min(int(polyorder), window - 1))

    half = window // 2
    x = np.arange(-half, half + 1, dtype=float)

    # Vandermonde matrix for polynomial fit.
    design = np.vander(x, polyorder + 1, increasing=True)

    # Coefficients that return the fitted value at x = 0.
    coeffs = np.linalg.pinv(design)[0]

    padded = np.pad(values, half, mode="edge")

    # Convolution uses reversed coefficients.
    return np.convolve(padded, coeffs[::-1], mode="valid")

def find_candidates_strict(
    counts: np.ndarray,
    axis: np.ndarray,
    z: int = 5,
    w: int = 9,
    sigma_factor: float = 2.0,
    min_negative_width: int = 3,
    smooth_counts: bool = True,
    sg_window: int = 9,
    sg_polyorder: int = 3,
) -> tuple[list[dict[str, Any]], np.ndarray, float]:
    """Find peak candidates using a stricter Mariscotti-like lobe rule.

    A candidate must contain a significant positive lobe, followed by a
    significant negative lobe, followed by a significant positive lobe. This is
    much closer to the original Mariscotti peak-recognition idea than accepting
    every negative region of the second-difference transform.
    """
    counts = np.asarray(counts, dtype=float)
    axis = np.asarray(axis, dtype=float)
    if len(counts) != len(axis):
        raise ValueError("counts and axis must have the same length")
        
    if smooth_counts:
        counts_for_transform = _savgol_smooth_numpy(
            counts,
            sg_window,
            sg_polyorder,
        )
    else:
        counts_for_transform = counts

    transform, _raw = mariscotti_transform(counts_for_transform, z=z, w=w)
    sigma_t = _robust_sigma(transform)
    threshold = float(sigma_factor) * sigma_t

    sign = np.sign(transform)
    sign[np.abs(transform) < threshold] = 0

    n = len(transform)
    candidates: list[dict[str, Any]] = []
    i = 1

    while i < n - 1:
        # First positive side lobe.
        if sign[i] <= 0:
            i += 1
            continue
        p1_start = i
        while i < n - 1 and sign[i] > 0:
            i += 1
        p1_end = i - 1

        # Central negative lobe.
        if i >= n - 1 or sign[i] >= 0:
            continue
        n_start = i
        while i < n - 1 and sign[i] < 0:
            i += 1
        n_end = i - 1

        # Second positive side lobe.
        if i >= n - 1 or sign[i] <= 0:
            continue
        p2_start = i
        while i < n - 1 and sign[i] > 0:
            i += 1
        p2_end = i - 1

        negative_width = n_end - n_start + 1
        if negative_width < int(min_negative_width):
            continue

        p1_max = float(np.max(transform[p1_start:p1_end + 1])) if p1_end >= p1_start else 0.0
        p2_max = float(np.max(transform[p2_start:p2_end + 1])) if p2_end >= p2_start else 0.0
        n_min = float(np.min(transform[n_start:n_end + 1])) if n_end >= n_start else 0.0

        if p1_max < threshold or p2_max < threshold or -n_min < threshold:
            continue

        center_idx = int(np.argmin(transform[n_start:n_end + 1]) + n_start)
        left_idx = max(0, p1_start)
        right_idx = min(n - 1, p2_end)

        candidates.append(
            {
                "left_idx": left_idx,
                "right_idx": right_idx,
                "center_idx": center_idx,
                "left_value": float(axis[left_idx]),
                "right_value": float(axis[right_idx]),
                "center_value": float(axis[center_idx]),
                "negative_width": int(negative_width),
                "p1_max": p1_max,
                "p2_max": p2_max,
                "n_min": n_min,
            }
        )

    return candidates, transform, sigma_t


def _gaussian_with_linear_background(x, amplitude, mu, sigma, b0, b1):
    return amplitude * np.exp(-0.5 * ((x - mu) / sigma) ** 2) + b0 + b1 * x


def _energy_to_channel(spectrum: Spectrum, energy_position: float) -> float:
    return float(np.interp(float(energy_position), spectrum.energy, spectrum.channels))


def _fit_candidate_on_energy_axis(
    spectrum: Spectrum,
    candidate: dict[str, Any],
    fit_half_width: int = 8,
) -> Peak | None:
    """Fit one candidate with x expressed in calibrated energy, not channel."""
    counts = np.asarray(spectrum.counts, dtype=float)
    energy = np.asarray(spectrum.energy, dtype=float)

    c = int(candidate["center_idx"])
    left = max(0, c - int(fit_half_width))
    right = min(len(counts) - 1, c + int(fit_half_width))

    x = energy[left:right + 1]
    y = counts[left:right + 1]
    if len(x) < 5:
        return None

    baseline = 0.5 * (counts[left] + counts[right])
    amplitude0 = max(1.0, counts[c] - baseline)
    mu0 = float(energy[c])
    sigma0 = max(0.3, float(abs(x[-1] - x[0])) / 6.0)
    b0_0 = baseline
    b1_0 = 0.0

    # Bounds are deliberately kept. Without bounds the optimizer can push the
    # centroid outside the local peak window or inflate sigma, which changes the
    # fitted area and therefore the peak-to-valley ratio dramatically.
    try:
        popt, pcov = curve_fit(
            _gaussian_with_linear_background,
            x,
            y,
            p0=[amplitude0, mu0, sigma0, b0_0, b1_0],
            bounds=(
                [0.0, float(np.min(x)), 0.3, -np.inf, -np.inf],
                [np.inf, float(np.max(x)), np.inf, np.inf, np.inf],
            ),
            maxfev=10000,
        )
    except Exception:
        return None

    amplitude, peak_energy, sigma_energy, b0, b1 = [float(v) for v in popt]
    if amplitude <= 0 or sigma_energy <= 0 or not np.isfinite(peak_energy):
        return None

    area = float(amplitude * sigma_energy * math.sqrt(2.0 * math.pi))
    fwhm_energy = float(2.354820045 * sigma_energy)
    left_energy = float(peak_energy - fwhm_energy / 2.0)
    right_energy = float(peak_energy + fwhm_energy / 2.0)

    if pcov is not None and np.shape(pcov) == (5, 5) and np.all(np.isfinite(pcov)):
        var_amp = max(float(pcov[0, 0]), 0.0)
        var_sigma = max(float(pcov[2, 2]), 0.0)
        cov_amp_sigma = float(pcov[0, 2])
        amp_unc = float(math.sqrt(var_amp))
        sigma_unc = float(math.sqrt(var_sigma))
        var_area = 2.0 * math.pi * (
            sigma_energy**2 * var_amp
            + amplitude**2 * var_sigma
            + 2.0 * amplitude * sigma_energy * cov_amp_sigma
        )
        area_unc = float(math.sqrt(max(var_area, 0.0)))
    else:
        amp_unc = float("nan")
        sigma_unc = float("nan")
        area_unc = float("nan")

    peak_channel = _energy_to_channel(spectrum, peak_energy)
    left_channel = _energy_to_channel(spectrum, left_energy)
    right_channel = _energy_to_channel(spectrum, right_energy)
    fwhm_channels = abs(right_channel - left_channel)

    return Peak(
        index=-1,
        peak_channel=peak_channel,
        peak_energy=peak_energy,
        area=area,
        area_uncertainty=area_unc,
        amplitude=amplitude,
        amplitude_uncertainty=amp_unc,
        sigma=sigma_energy,
        fwhm_channels=fwhm_channels,
        fwhm_energy=fwhm_energy,
        left_channel=left_channel,
        right_channel=right_channel,
        left_energy=left_energy,
        right_energy=right_energy,
    )


def detect_peaks(
    spectrum: Spectrum,
    z: int = 5,
    w: int = 9,
    sigma_factor: float = 2.0,
    min_negative_width: int = 3,
    fit_half_width: int = 8,
    smooth_counts: bool = True,
    sg_window: int = 9,
    sg_polyorder: int = 3,
) -> tuple[list[Peak], list[dict[str, Any]], np.ndarray, float]:
    """Detect and fit peaks using strict Mariscotti-like detection on energy axis."""
    candidates, transform, sigma_t = find_candidates_strict(
        spectrum.counts,
        spectrum.energy,
        z=z,
        w=w,
        sigma_factor=sigma_factor,
        min_negative_width=min_negative_width,
        smooth_counts=smooth_counts,
        sg_window=sg_window,
        sg_polyorder=sg_polyorder,
    )

    peaks: list[Peak] = []
    for candidate in candidates:
        peak = _fit_candidate_on_energy_axis(spectrum, candidate, fit_half_width=fit_half_width)
        if peak is not None:
            peaks.append(peak)

    peaks.sort(key=lambda peak: peak.peak_energy)
    for i, peak in enumerate(peaks, start=1):
        peak.index = i

    return peaks, candidates, transform, sigma_t


def nearest_peak(peaks: list[Peak], target_energy: float, tolerance_kev: float) -> Peak:
    if not peaks:
        raise ValueError("No peaks have been detected.")

    candidates = [
        peak for peak in peaks
        if abs(peak.peak_energy - float(target_energy)) <= float(tolerance_kev)
    ]
    if not candidates:
        detected = ", ".join(f"{peak.peak_energy:.2f}" for peak in peaks[:20])
        raise ValueError(
            f"No detected peak within {tolerance_kev:.2f} keV of {target_energy:.2f} keV. "
            f"Detected energies: {detected}"
        )

    return min(candidates, key=lambda peak: abs(peak.peak_energy - float(target_energy)))
