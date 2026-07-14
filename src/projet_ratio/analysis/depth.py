from __future__ import annotations

import math

from scipy.optimize import brentq


def _validate_positive(name: str, value: float) -> float:
    value = float(value)
    if not math.isfinite(value) or value <= 0:
        raise ValueError(f"{name} must be a positive finite number.")
    return value


def calculate_point_source_depth(
    peak_to_valley_ratio: float,
    peak_to_valley_uncertainty: float,
    k: float,
    mu: float,
    density: float,
) -> tuple[float, float]:
    """Depth for a point source or source below a finite material thickness.

    The app calculates Q = peak / valley.
    The small-angle Compton model uses R = valley / peak = k * mu * density * x.
    Therefore x = 1 / (Q * k * mu * density).

    Units:
    - k: 1/keV
    - mu: cm^2/g
    - density: g/cm^3
    - depth x: cm
    """
    q = _validate_positive("peak_to_valley_ratio", peak_to_valley_ratio)
    sigma_q = max(float(peak_to_valley_uncertainty), 0.0)
    k = _validate_positive("k", k)
    mu = _validate_positive("mu", mu)
    density = _validate_positive("density", density)

    depth = 1.0 / (q * k * mu * density)
    sigma_depth = depth * sigma_q / q
    return float(depth), float(sigma_depth)


def _uniform_ratio(depth: float, k: float, mu: float, density: float) -> float:
    """R = valley / peak for a uniformly contaminated layer."""
    a = mu * density * depth
    if abs(a) < 1e-8:
        return 0.5 * k * a
    numerator = 1.0 - math.exp(-a) * (1.0 + a)
    denominator = 1.0 - math.exp(-a)
    return k * numerator / denominator


def calculate_uniform_contamination_depth(
    discontinuity_to_peak_ratio: float,
    discontinuity_to_peak_uncertainty: float,
    k: float,
    mu: float,
    density: float,
    max_depth_cm: float = 10000.0,
) -> tuple[float, float]:
    """Depth for a uniformly contaminated matrix layer.

    Solves:
        R = k * [1 - exp(-mu*rho*x)*(1 + mu*rho*x)] / [1 - exp(-mu*rho*x)]

    where R = valley / peak.
    """
    r = _validate_positive("discontinuity_to_peak_ratio", discontinuity_to_peak_ratio)
    sigma_r = max(float(discontinuity_to_peak_uncertainty), 0.0)
    k = _validate_positive("k", k)
    mu = _validate_positive("mu", mu)
    density = _validate_positive("density", density)
    max_depth_cm = _validate_positive("max_depth_cm", max_depth_cm)

    if r >= k:
        raise ValueError(
            "No finite uniform-layer depth exists because valley/peak ratio is >= k. "
            "Check k, valley selection, or use the point-source model."
        )

    def f(x: float) -> float:
        return _uniform_ratio(x, k, mu, density) - r

    upper = 1.0
    while f(upper) < 0 and upper < max_depth_cm:
        upper *= 2.0

    if upper >= max_depth_cm and f(upper) < 0:
        raise ValueError("Could not bracket the uniform-layer depth solution.")

    depth = brentq(f, 0.0, upper, xtol=1e-9, rtol=1e-9, maxiter=100)

    # Numerical uncertainty propagation: sigma_x = sigma_R / |dR/dx|.
    eps = max(depth * 1e-5, 1e-5)
    left = max(depth - eps, 0.0)
    right = depth + eps
    derivative = (_uniform_ratio(right, k, mu, density) - _uniform_ratio(left, k, mu, density)) / (right - left)
    sigma_depth = sigma_r / abs(derivative) if derivative != 0 else float("nan")
    return float(depth), float(sigma_depth)
