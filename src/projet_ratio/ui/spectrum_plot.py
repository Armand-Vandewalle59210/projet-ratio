from __future__ import annotations

import numpy as np
import pyqtgraph as pg
from PySide6.QtCore import Signal
from PySide6.QtWidgets import QWidget, QVBoxLayout

from projet_ratio.models import Peak, Spectrum


class SpectrumPlot(QWidget):
    """PyQtGraph widget with spectrum, peak markers, and valley ROIs."""

    valley_regions_changed = Signal()

    def __init__(self, parent=None):
        super().__init__(parent)
        self.plot = pg.PlotWidget(background="w")
        self.plot.setLabel("bottom", "Energy", units="keV")
        self.plot.setLabel("left", "Counts")
        self.plot.showGrid(x=True, y=True, alpha=0.25)
        self.plot.addLegend()

        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(self.plot)

        self.spectrum_curve = None
        self.peak_items: list[pg.PlotDataItem] = []
        self.log_mode = False

        self.lower_region = pg.LinearRegionItem(values=[631.0, 649.0], brush=(80, 150, 255, 45))
        self.upper_region = pg.LinearRegionItem(values=[672.0, 690.0], brush=(255, 160, 60, 45))
        self.lower_region.setZValue(10)
        self.upper_region.setZValue(10)
        self.lower_region.sigRegionChanged.connect(self.valley_regions_changed.emit)
        self.upper_region.sigRegionChanged.connect(self.valley_regions_changed.emit)
        self.plot.addItem(self.lower_region)
        self.plot.addItem(self.upper_region)

    def set_log_mode(self, enabled: bool) -> None:
        self.log_mode = bool(enabled)
        self.plot.setLogMode(x=False, y=self.log_mode)

    def set_spectrum(self, spectrum: Spectrum) -> None:
        if self.spectrum_curve is not None:
            self.plot.removeItem(self.spectrum_curve)
        y = np.asarray(spectrum.counts, dtype=float)
        if self.log_mode:
            positive = y[y > 0]
            if positive.size:
                y = np.clip(y, positive.min() * 0.5, None)
        self.spectrum_curve = self.plot.plot(
            spectrum.energy,
            y,
            pen=pg.mkPen("#1565C0", width=1.2),
            name="Spectrum",
        )
        self.plot.enableAutoRange()

    def set_peaks(self, peaks: list[Peak]) -> None:
        self.clear_peaks()
        if not peaks:
            return
        x = np.asarray([p.peak_energy for p in peaks], dtype=float)
        y = np.asarray([max(p.amplitude, 1.0) for p in peaks], dtype=float)
        item = self.plot.plot(
            x,
            y,
            pen=None,
            symbol="t",
            symbolSize=10,
            symbolBrush="#D32F2F",
            name="Detected peaks",
        )
        self.peak_items.append(item)

    def clear_peaks(self) -> None:
        for item in self.peak_items:
            self.plot.removeItem(item)
        self.peak_items.clear()

    def lower_region_values(self) -> tuple[float, float]:
        a, b = self.lower_region.getRegion()
        return float(a), float(b)

    def upper_region_values(self) -> tuple[float, float]:
        a, b = self.upper_region.getRegion()
        return float(a), float(b)

    def set_region_defaults_for_peak(self, peak_energy: float) -> None:
        """Place valley regions around a selected peak.

        The defaults mimic the Cs-137 example: roughly -30 to -12 keV and
        +10 to +28 keV around 661.6 keV.
        """
        e = float(peak_energy)
        self.lower_region.setRegion((e - 30.0, e - 12.0))
        self.upper_region.setRegion((e + 10.0, e + 28.0))
