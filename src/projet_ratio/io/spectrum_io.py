from __future__ import annotations

import csv
import re
from pathlib import Path
from typing import Iterable

import numpy as np

from projet_ratio.models import Spectrum


AXIS_NAME_PATTERNS = (
    "energy", "energie", "kev", "channel", "chan", "ch", "x", "bin"
)

# The entries are deliberately broad. Examples matched here include:
# Counts, count, Count_1000s, Smeared_at_1000s, N, cps, counts_per_s, etc.
COUNTS_NAME_PATTERNS = (
    "smeared", "count", "counts", "cnt", "cps", "peak", "peaks", "spectrum", "y", "n"
)

ENERGY_NAME_PATTERNS = ("energy", "energie", "kev", "e_kev")
CHANNEL_NAME_PATTERNS = ("channel", "channels", "chan", "ch", "bin")


def _energy_from_coefficients(channels: np.ndarray, coefficients: Iterable[float] | None) -> np.ndarray:
    """Build energy axis from polynomial calibration coefficients.

    Coefficients are interpreted as E = a0 + a1*C + a2*C^2 + ...
    If no coefficients are available, the channel axis is used as a fallback.
    """
    channels = np.asarray(channels, dtype=float)
    if coefficients is None:
        return channels.copy()

    coeffs = [float(c) for c in coefficients]
    if not coeffs:
        return channels.copy()

    energy = np.zeros_like(channels, dtype=float)
    for power, coefficient in enumerate(coeffs):
        energy += coefficient * channels**power
    return energy


def _normalize_name(name: str) -> str:
    """Normalize a CSV header for robust matching."""
    return re.sub(r"[^a-z0-9]+", "_", str(name).strip().lower()).strip("_")


def _name_contains_any(name: str, patterns: tuple[str, ...]) -> bool:
    normalized = _normalize_name(name)
    tokens = set(normalized.split("_"))
    for pattern in patterns:
        p = _normalize_name(pattern)
        if p in tokens or p in normalized:
            return True
    return False


def _score_counts_column(header: str, index: int, numeric_count: int) -> float:
    """Rank likely counts columns while avoiding energy/channel columns."""
    name = _normalize_name(header)
    score = float(numeric_count)

    if _name_contains_any(name, COUNTS_NAME_PATTERNS):
        score += 10_000
    if _name_contains_any(name, ("count", "counts", "cnt", "cps")):
        score += 20_000
    if _name_contains_any(name, ("smeared", "spectrum")):
        score += 8_000
    if _name_contains_any(name, AXIS_NAME_PATTERNS):
        score -= 15_000

    # If there is no useful name, prefer later numeric columns over the first
    # column, because the first column is often channel or energy.
    score += index * 0.01
    return score


def _read_csv(path: Path) -> Spectrum:
    """Read a flexible CSV spectrum without importing pandas.

    Supported examples:
    - one numeric column: interpreted as counts
    - Energy_keV / Counts
    - channel / count_1000s
    - x / Smeared_at_1000s
    - any numeric column containing count, counts, cnt, cps, smeared, spectrum, y, etc.
    """
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        sample = handle.read(4096)
        handle.seek(0)
        try:
            dialect = csv.Sniffer().sniff(sample, delimiters=",;\t")
        except csv.Error:
            dialect = csv.excel
        rows = [row for row in csv.reader(handle, dialect) if row and any(cell.strip() for cell in row)]

    if not rows:
        raise ValueError("CSV file is empty.")

    def to_float(cell: str) -> float:
        cell = cell.strip().replace(" ", "")
        # Support decimal comma when it is not also used as a delimiter.
        cell = cell.replace(",", ".")
        return float(cell)

    def row_is_numeric(row: list[str]) -> bool:
        try:
            [to_float(cell) for cell in row if cell.strip()]
            return True
        except ValueError:
            return False

    has_header = not row_is_numeric(rows[0])
    if has_header:
        headers = [cell.strip() for cell in rows[0]]
        data_rows = rows[1:]
    else:
        width = max(len(row) for row in rows)
        headers = [f"column_{i}" for i in range(width)]
        data_rows = rows

    width = len(headers)
    numeric_columns: list[list[float]] = [[] for _ in range(width)]
    for row in data_rows:
        for index in range(width):
            if index >= len(row):
                numeric_columns[index].append(float("nan"))
                continue
            try:
                numeric_columns[index].append(to_float(row[index]))
            except ValueError:
                numeric_columns[index].append(float("nan"))

    numeric_counts = [int(np.isfinite(np.asarray(col, dtype=float)).sum()) for col in numeric_columns]
    valid_indices = [i for i, n in enumerate(numeric_counts) if n > 0]
    if not valid_indices:
        raise ValueError("CSV file must contain at least one numeric column.")

    def first_named_column(patterns: tuple[str, ...]) -> int | None:
        for index, header in enumerate(headers):
            if index in valid_indices and _name_contains_any(header, patterns):
                return index
        return None

    energy_index = first_named_column(ENERGY_NAME_PATTERNS)
    channel_index = first_named_column(CHANNEL_NAME_PATTERNS)

    # Pick the best counts column using names and numeric availability.
    candidate_indices = [i for i in valid_indices if i not in {energy_index, channel_index}]
    if not candidate_indices:
        candidate_indices = valid_indices

    counts_index = max(
        candidate_indices,
        key=lambda i: _score_counts_column(headers[i], i, numeric_counts[i]),
    )

    counts_all = np.asarray(numeric_columns[counts_index], dtype=float)
    finite_mask = np.isfinite(counts_all)
    if not np.any(finite_mask):
        raise ValueError("Counts column does not contain numeric data.")

    counts = np.nan_to_num(counts_all[finite_mask], nan=0.0)

    if channel_index is not None and channel_index < len(numeric_columns):
        channels = np.asarray(numeric_columns[channel_index], dtype=float)[finite_mask]
    else:
        channels = np.arange(len(counts), dtype=float)

    if energy_index is not None and energy_index < len(numeric_columns):
        energy = np.asarray(numeric_columns[energy_index], dtype=float)[finite_mask]
        calibration_coefficients = None
    else:
        energy = channels.copy()
        calibration_coefficients = None

    return Spectrum(
        counts=counts,
        channels=channels,
        energy=energy,
        live_time=None,
        path=path,
        calibration_coefficients=calibration_coefficients,
        metadata={
            "reader": "csv",
            "counts_column": headers[counts_index],
            "energy_column": None if energy_index is None else headers[energy_index],
            "channel_column": None if channel_index is None else headers[channel_index],
        },
    )


def _call_first_available(obj, names: tuple[str, ...]):
    """Try several method/property names used by different SpecUtils builds."""
    for name in names:
        value = getattr(obj, name, None)
        if value is None:
            continue
        if callable(value):
            return value()
        return value
    return None


def _read_with_specutils(path: Path) -> Spectrum:
    """Read CNF/N42/XML through SpecUtils and use embedded calibration when present."""
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

    counts = _call_first_available(
        measurement,
        ("gammaCounts", "gamma_counts", "counts", "channelCounts", "channel_counts"),
    )
    if counts is None:
        raise AttributeError("Could not extract counts from SpecUtils measurement.")

    counts = np.asarray(list(counts), dtype=float)
    channels = np.arange(len(counts), dtype=float)

    live_time = _call_first_available(
        measurement,
        ("liveTime", "live_time", "gammaLiveTime", "gamma_live_time", "realTime"),
    )
    live_time = None if live_time is None else float(live_time)

    # This is the important fix for your files: your earlier Spectrum_io code
    # used meas.calibrationCoeffs(), so the clean app must try that first too.
    coeffs = _call_first_available(
        measurement,
        ("calibrationCoeffs", "calibration_coeffs", "energyCalibrationCoeffs", "energy_calibration_coeffs"),
    )

    if coeffs is not None:
        try:
            coeffs = tuple(float(c) for c in coeffs)
        except TypeError:
            coeffs = None

    # Some bindings expose a calibration object instead of calibrationCoeffs().
    if coeffs is None:
        calibration = _call_first_available(measurement, ("energy_calibration", "energyCalibration", "calibration"))
        if calibration is not None:
            coeffs = _call_first_available(
                calibration,
                ("coefficients", "coeffs", "polynomial_coefficients", "calibrationCoeffs"),
            )
            if coeffs is not None:
                try:
                    coeffs = tuple(float(c) for c in coeffs)
                except TypeError:
                    coeffs = None

    # If SpecUtils can directly provide channel energies, use them. Otherwise
    # compute energy from calibration coefficients.
    energy = _call_first_available(
        measurement,
        ("channelEnergies", "channel_energies", "energies", "gammaChannelEnergies"),
    )
    if energy is not None:
        energy = np.asarray(list(energy), dtype=float)

    if energy is None or len(energy) != len(counts):
        energy = _energy_from_coefficients(channels, coeffs)

    return Spectrum(
        counts=counts,
        channels=channels,
        energy=energy,
        live_time=live_time,
        path=path,
        calibration_coefficients=coeffs,
        metadata={"reader": "SpecUtils", "calibration_coefficients": coeffs},
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