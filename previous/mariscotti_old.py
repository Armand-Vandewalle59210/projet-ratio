from __future__ import annotations

import math
from typing import Any

import numpy as np
from scipy.optimize import curve_fit
from scipy.signal import savgol_filter

from projet_ratio.models import Peak, Spectrum


def _moving_average(values: np.ndarray, window: int) -> np.ndarray:
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
    values = np.asarray(values, dtype=float)
    median = np.median(values)
    mad = np.median(np.abs(values - median))
    sigma = 1.4826 * mad
    if not np.isfinite(sigma) or sigma <= 0:
        sigma = np.std(values)
    if not np.isfinite(sigma) or sigma <= 0:
        sigma = 1.0
    return float(sigma)


def _sanitize_savgol_window(window: int, polyorder: int, n: int) -> tuple[int, int]:
    if n < 5:
        return 1, 0
    window = max(3, int(window))
    if window % 2 == 0:
        window += 1
    if window >= n:
        window = n - 1 if n % 2 == 0 else n
    if window % 2 == 0:
        window -= 1
    polyorder = max(1, min(int(polyorder), window - 1))
    return window, polyorder


def mariscotti_transform(counts: np.ndarray, z: int = 5, w: int = 5) -> np.ndarray:
    """Simplified Mariscotti transform based on a smoothed second difference."""
    counts = np.asarray(counts, dtype=float)
    second = np.zeros_like(counts, dtype=float)
    if len(counts) >= 3:
        second[1:-1] = counts[:-2] - 2.0 * counts[1:-1] + counts[2:]
        second[0] = second[1]
        second[-1] = second[-2]

    transformed = second
    for _ in range(max(0, int(z))):
        transformed = _moving_average(transformed, int(w))
    return transformed


def find_candidates(
    counts: np.ndarray,
    z: int = 5,
    w: int = 5,
    sigma_factor: float = 2.0,
    min_negative_width: int = 3,
    smooth_counts: bool = True,
    sg_window: int = 9,
    sg_polyorder: int = 3,
) -> tuple[list[dict[str, Any]], np.ndarray, float]:
    counts = np.asarray(counts, dtype=float)
    working = counts.copy()

    if smooth_counts:
        window, polyorder = _sanitize_savgol_window(sg_window, sg_polyorder, len(working))
        if window > 1:
            working = savgol_filter(working, window, polyorder)

    transform = mariscotti_transform(working, z=z, w=w)
    sigma = _robust_sigma(transform)
    threshold = -abs(float(sigma_factor)) * sigma

    candidates: list[dict[str, Any]] = []
    in_region = False
    start = 0

    for i, value in enumerate(transform):
        if value < threshold and not in_region:
            start = i
            in_region = True
        elif value >= threshold and in_region:
            end = i - 1
            if end - start + 1 >= int(min_negative_width):
                local = transform[start : end + 1]
                center = start + int(np.argmin(local))
                candidates.append({"left_index": start, "right_index": end, "center_index": center})
            in_region = False

    if in_region:
        end = len(transform) - 1
        if end - start + 1 >= int(min_negative_width):
            local = transform[start : end + 1]
            center = start + int(np.argmin(local))
            candidates.append({"left_index": start, "right_index": end, "center_index": center})

    return candidates, transform, sigma


def _gaussian_with_linear_background(x, amplitude, mu, sigma, b0, b1):
    sigma = np.maximum(np.abs(sigma), 1e-12)
    return amplitude * np.exp(-0.5 * ((x - mu) / sigma) ** 2) + b0 + b1 * x


def _interp_energy(spectrum: Spectrum, channel_position: float) -> float:
    return float(np.interp(channel_position, spectrum.channels, spectrum.energy))


def _fit_candidate(spectrum: Spectrum, candidate: dict[str, Any], fit_half_width: int) -> Peak | None:
    counts = spectrum.counts
    channels = spectrum.channels
    center = int(candidate["center_index"])
    left = max(0, center - int(fit_half_width))
    right = min(len(counts) - 1, center + int(fit_half_width))

    if right - left + 1 < 5:
        return None

    x = np.asarray(channels[left : right + 1], dtype=float)
    y = np.asarray(counts[left : right + 1], dtype=float)

    edge_background = float(np.median(np.r_[y[:2], y[-2:]]))
    amplitude0 = max(float(y.max() - edge_background), 1.0)
    mu0 = float(channels[center])
    sigma0 = max((float(x.max() - x.min()) / 6.0), 1.0)
    b0 = edge_background
    b1 = 0.0

    try:
        popt, pcov = curve_fit(
            _gaussian_with_linear_background,
            x,
            y,
            p0=[amplitude0, mu0, sigma0, b0, b1],
            maxfev=10000,
        )
    except Exception:
        return None

    amplitude, mu, sigma, b0, b1 = [float(v) for v in popt]
    sigma = abs(sigma)
    if amplitude <= 0 or sigma <= 0 or not np.isfinite(mu):
        return None

    perr = np.sqrt(np.diag(pcov)) if pcov is not None and np.all(np.isfinite(pcov)) else np.full(5, np.nan)
    amplitude_unc = float(perr[0]) if np.isfinite(perr[0]) else math.sqrt(abs(amplitude))
    sigma_unc = float(perr[2]) if np.isfinite(perr[2]) else 0.0

    area = float(amplitude * sigma * math.sqrt(2.0 * math.pi))
    area_unc = area * math.sqrt(
        (amplitude_unc / amplitude) ** 2 + (sigma_unc / sigma) ** 2
    ) if amplitude > 0 and sigma > 0 else math.sqrt(abs(area))

    fwhm_channels = float(2.354820045 * sigma)
    left_channel = float(mu - fwhm_channels / 2.0)
    right_channel = float(mu + fwhm_channels / 2.0)

    peak_energy = _interp_energy(spectrum, mu)
    left_energy = _interp_energy(spectrum, left_channel)
    right_energy = _interp_energy(spectrum, right_channel)
    fwhm_energy = abs(right_energy - left_energy)

    return Peak(
        index=-1,
        peak_channel=float(mu),
        peak_energy=peak_energy,
        area=area,
        area_uncertainty=float(area_unc),
        amplitude=amplitude,
        amplitude_uncertainty=amplitude_unc,
        sigma=sigma,
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
    """Detect and fit peaks using the simplified Mariscotti workflow."""
    candidates, transform, sigma = find_candidates(
        spectrum.counts,
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
        peak = _fit_candidate(spectrum, candidate, fit_half_width=fit_half_width)
        if peak is not None:
            peaks.append(peak)

    peaks.sort(key=lambda p: p.peak_energy)
    for i, peak in enumerate(peaks, start=1):
        peak.index = i

    return peaks, candidates, transform, sigma


def nearest_peak(peaks: list[Peak], target_energy: float, tolerance_kev: float) -> Peak:
    if not peaks:
        raise ValueError("No peaks have been detected.")

    candidates = [p for p in peaks if abs(p.peak_energy - target_energy) <= tolerance_kev]
    if not candidates:
        detected = ", ".join(f"{p.peak_energy:.2f}" for p in peaks[:20])
        raise ValueError(
            f"No detected peak within {tolerance_kev:.2f} keV of {target_energy:.2f} keV. "
            f"Detected energies: {detected}"
        )

    return min(candidates, key=lambda p: abs(p.peak_energy - target_energy))
