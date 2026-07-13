from __future__ import annotations

from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd

from projet_ratio.models import Spectrum


def _as_float_array(values: Iterable[float]) -> np.ndarray:
    return np.asarray(list(values), dtype=float)


def _energy_from_coefficients(channels: np.ndarray, coefficients: Iterable[float] | None) -> np.ndarray:
    """Build energy axis from polynomial calibration coefficients.

    Coefficients are interpreted as E = a0 + a1*C + a2*C^2 + ...
    If no coefficients are available, the channel axis is used as a fallback.
    """
    channels = np.asarray(channels, dtype=float)
    if coefficients is None:
        return channels.copy()

    coeffs = list(float(c) for c in coefficients)
    if not coeffs:
        return channels.copy()

    energy = np.zeros_like(channels, dtype=float)
    for power, coefficient in enumerate(coeffs):
        energy += coefficient * channels**power
    return energy


def _read_csv(path: Path) -> Spectrum:
    """Read a flexible CSV spectrum.

    Supported column patterns:
    - counts only: one numeric column
    - channel/counts: columns named like channel and counts
    - energy/counts: columns named like energy and counts
    - channel/energy/counts: all three columns
    """
    df = pd.read_csv(path)
    if df.empty:
        raise ValueError("CSV file is empty.")

    normalized = {str(c).strip().lower(): c for c in df.columns}

    def find_column(names: tuple[str, ...]) -> str | None:
        for candidate in names:
            for lower_name, original_name in normalized.items():
                if candidate in lower_name:
                    return original_name
        return None

    counts_col = find_column(("counts", "count", "cnt", "cps"))
    channel_col = find_column(("channel", "chan", "ch"))
    energy_col = find_column(("energy", "kev", "energie"))

    numeric_df = df.select_dtypes(include=["number"])
    if counts_col is None:
        if numeric_df.shape[1] == 0:
            raise ValueError("CSV file must contain at least one numeric counts column.")
        counts_col = numeric_df.columns[-1]

    counts = np.asarray(df[counts_col], dtype=float)

    if channel_col is not None:
        channels = np.asarray(df[channel_col], dtype=float)
    else:
        channels = np.arange(len(counts), dtype=float)

    if energy_col is not None:
        energy = np.asarray(df[energy_col], dtype=float)
        coeffs = None
    else:
        energy = channels.copy()
        coeffs = None

    return Spectrum(
        counts=counts,
        channels=channels,
        energy=energy,
        live_time=None,
        path=path,
        calibration_coefficients=coeffs,
        metadata={"reader": "csv", "counts_column": str(counts_col)},
    )


def _read_with_specutils(path: Path) -> Spectrum:
    """Read CNF/N42/XML through the local SpecUtils module.

    This keeps SpecUtils optional: the app can still open CSV files on systems
    where SpecUtils is not installed.
    """
    try:
        import SpecUtils  # type: ignore
    except ImportError as exc:
        raise ImportError(
            "SpecUtils is required for CNF, N42, and XML files. "
            "Install your SpecUtils package or use CSV input."
        ) from exc

    spec = SpecUtils.SpecFile()
    spec.loadFile(str(path), SpecUtils.ParserType.Auto)
    measurements = spec.measurements()
    if not measurements:
        raise ValueError("No measurement found in spectrum file.")

    measurement = measurements[0]

    # SpecUtils Python bindings can differ slightly between builds, so use a
    # defensive sequence of common method/property names.
    counts = None
    for name in ("gamma_counts", "gammaCounts", "counts"):
        value = getattr(measurement, name, None)
        if callable(value):
            counts = value()
            break
        if value is not None:
            counts = value
            break

    if counts is None:
        raise AttributeError("Could not extract counts from SpecUtils measurement.")

    counts = np.asarray(counts, dtype=float)
    channels = np.arange(len(counts), dtype=float)

    live_time = None
    for name in ("live_time", "liveTime", "gamma_live_time"):
        value = getattr(measurement, name, None)
        if callable(value):
            live_time = float(value())
            break
        if value is not None:
            live_time = float(value)
            break

    coeffs = None
    calibration = getattr(measurement, "energy_calibration", None)
    if callable(calibration):
        calibration = calibration()

    if calibration is not None:
        for name in ("coefficients", "coeffs", "polynomial_coefficients"):
            value = getattr(calibration, name, None)
            if callable(value):
                coeffs = tuple(float(c) for c in value())
                break
            if value is not None:
                coeffs = tuple(float(c) for c in value)
                break

    # Fallback: some SpecUtils versions expose channel_energies directly.
    energy = None
    for name in ("channel_energies", "energies"):
        value = getattr(measurement, name, None)
        if callable(value):
            energy = np.asarray(value(), dtype=float)
            break
        if value is not None:
            energy = np.asarray(value, dtype=float)
            break

    if energy is None or len(energy) != len(counts):
        energy = _energy_from_coefficients(channels, coeffs)

    return Spectrum(
        counts=counts,
        channels=channels,
        energy=energy,
        live_time=live_time,
        path=path,
        calibration_coefficients=coeffs,
        metadata={"reader": "SpecUtils"},
    )


def load_spectrum(path: str | Path) -> Spectrum:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(path)

    suffix = path.suffix.lower()
    if suffix == ".csv":
        return _read_csv(path)
    if suffix in {".cnf", ".n42", ".xml"}:
        return _read_with_specutils(path)

    raise ValueError(f"Unsupported file type: {suffix}")
