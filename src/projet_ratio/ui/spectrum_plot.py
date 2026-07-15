from __future__ import annotations

import numpy as np
import pyqtgraph as pg
from PySide6.QtCore import Signal
from PySide6.QtWidgets import QWidget, QVBoxLayout

from projet_ratio.models import Peak, Spectrum


class SpectrumPlot(QWidget):
    """PyQtGraph widget with spectrum, selected peak marker, and valley ROIs."""

    valley_regions_changed = Signal()

    def __init__(self, parent=None):
        super().__init__(parent)
        self.plot = pg.PlotWidget(background="w")
        self.plot.setLabel("bottom", "Energy", units="keV")
        self.plot.setLabel("left", "Counts")
        self.plot.showGrid(x=True, y=True, alpha=0.25)
        self.plot.addLegend()

        # Prevent PyQtGraph from displaying keV values as kkeV-scaled values.
        for axis_name in ("bottom", "left"):
            axis = self.plot.getAxis(axis_name)
            if hasattr(axis, "enableAutoSIPrefix"):
                axis.enableAutoSIPrefix(False)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(self.plot)

        self.spectrum: Spectrum | None = None
        self.spectrum_curve: pg.PlotDataItem | None = None
        self.selected_peak_line: pg.InfiniteLine | None = None
        self.selected_peak_marker: pg.PlotDataItem | None = None
        self.all_peak_items: list[pg.GraphicsObject] = []
        self.log_mode = False
        self.show_all_peaks = False
        self._updating_regions = False

        self.lower_region = pg.LinearRegionItem(values=[631.0, 649.0], brush=(80, 150, 255, 45))
        self.upper_region = pg.LinearRegionItem(values=[672.0, 690.0], brush=(255, 160, 60, 45))
        self.lower_region.setZValue(10)
        self.upper_region.setZValue(10)
        self.lower_region.sigRegionChanged.connect(self._region_changed)
        self.upper_region.sigRegionChanged.connect(self._region_changed)
        
        self.plot.addItem(self.lower_region)
        self.plot.addItem(self.upper_region)
                
        self.compton_cursor = pg.InfiniteLine(
            pos=400.0,
            angle=90,
            movable=True,
            pen=pg.mkPen("#2E7D32", width=2),
            label="Compton cursor: {value:.2f} keV",
            labelOpts={"position": 0.85, "color": "#2E7D32"},
        )
        self.compton_cursor.setZValue(30)
        self.plot.addItem(self.compton_cursor)
        
        self.compton_region = pg.LinearRegionItem(
            values=[350.0, 450.0],
            brush=(40, 180, 90, 45),
        )
        self.compton_region.setZValue(9)
        self.plot.addItem(self.compton_region)
        
        self.background_left_region = pg.LinearRegionItem(
            values=[600.0, 620.0],
            brush=(150, 80, 200, 40),
        )
        self.background_right_region = pg.LinearRegionItem(
            values=[700.0, 720.0],
            brush=(150, 80, 200, 40),
        )

        self.background_left_region.setZValue(8)
        self.background_right_region.setZValue(8)

        self.plot.addItem(self.background_left_region)
        self.plot.addItem(self.background_right_region)


    def _region_changed(self) -> None:
        if not self._updating_regions:
            self.valley_regions_changed.emit()

    def set_log_mode(self, enabled: bool) -> None:
        self.log_mode = bool(enabled)
        self.plot.setLogMode(x=False, y=self.log_mode)

    def set_show_all_peak_markers(self, enabled: bool) -> None:
        self.show_all_peaks = bool(enabled)

    def _display_counts(self, counts: np.ndarray) -> np.ndarray:
        y = np.asarray(counts, dtype=float)
        if self.log_mode:
            positive = y[y > 0]
            if positive.size:
                y = np.clip(y, positive.min() * 0.5, None)
        return y

    def y_at_energy(self, energy: float) -> float:
        if self.spectrum is None:
            return 1.0
        y = self._display_counts(self.spectrum.counts)
        return float(np.interp(float(energy), self.spectrum.energy, y))

    def set_spectrum(self, spectrum: Spectrum) -> None:
        self.spectrum = spectrum
        if self.spectrum_curve is not None:
            self.plot.removeItem(self.spectrum_curve)
        y = self._display_counts(spectrum.counts)
        self.spectrum_curve = self.plot.plot(
            spectrum.energy,
            y,
            pen=pg.mkPen("#1565C0", width=1.2),
            name="Spectrum",
        )
        self.plot.enableAutoRange()

    def set_peaks(self, peaks: list[Peak], selected_peak: Peak | None = None) -> None:
        """Update peak indicators.

        By default only the selected peak is drawn. If show_all_peaks is enabled,
        all detected peaks are drawn as thin grey vertical guide lines.
        """
        self.clear_peak_indicators()

        if self.show_all_peaks:
            for peak in peaks:
                line = pg.InfiniteLine(
                    pos=peak.peak_energy,
                    angle=90,
                    movable=False,
                    pen=pg.mkPen((120, 120, 120, 90), width=1, style=pg.QtCore.Qt.PenStyle.DotLine),
                )
                line.setZValue(3)
                self.plot.addItem(line)
                self.all_peak_items.append(line)

        if selected_peak is not None:
            self.set_selected_peak(selected_peak)

    def set_selected_peak(self, peak: Peak | None) -> None:
        if self.selected_peak_line is not None:
            self.plot.removeItem(self.selected_peak_line)
            self.selected_peak_line = None
        if self.selected_peak_marker is not None:
            self.plot.removeItem(self.selected_peak_marker)
            self.selected_peak_marker = None

        if peak is None:
            return

        self.selected_peak_line = pg.InfiniteLine(
            pos=peak.peak_energy,
            angle=90,
            movable=False,
            pen=pg.mkPen("#C2185B", width=2),
            label=f"Selected peak: {peak.peak_energy:.2f} keV",
            labelOpts={"position": 0.92, "color": "#C2185B"},
        )
        self.selected_peak_line.setZValue(20)
        self.plot.addItem(self.selected_peak_line)

        y = self.y_at_energy(peak.peak_energy)
        self.selected_peak_marker = self.plot.plot(
            [peak.peak_energy],
            [y],
            pen=None,
            symbol="o",
            symbolSize=11,
            symbolBrush="#C2185B",
            symbolPen=pg.mkPen("w", width=1.5),
            name="Selected peak",
        )
        self.selected_peak_marker.setZValue(21)

    def clear_peak_indicators(self) -> None:
        for item in self.all_peak_items:
            self.plot.removeItem(item)
        self.all_peak_items.clear()
        if self.selected_peak_line is not None:
            self.plot.removeItem(self.selected_peak_line)
            self.selected_peak_line = None
        if self.selected_peak_marker is not None:
            self.plot.removeItem(self.selected_peak_marker)
            self.selected_peak_marker = None

    def clear_peaks(self) -> None:
        self.clear_peak_indicators()

    def lower_region_values(self) -> tuple[float, float]:
        a, b = self.lower_region.getRegion()
        return float(a), float(b)

    def upper_region_values(self) -> tuple[float, float]:
        a, b = self.upper_region.getRegion()
        return float(a), float(b)

    def set_region_defaults_for_peak(
        self,
        peak_energy: float,
        valley_size: float = 18.0,
        valley_distance: float = 12.0,
    ) -> None:
        """Place equal-width valley regions around a selected peak."""
        e = float(peak_energy)
        size = max(float(valley_size), 0.1)
        distance = max(float(valley_distance), 0.0)
        self._updating_regions = True
        
        try:
            self.lower_region.setRegion((e - distance - size, e - distance))
            self.upper_region.setRegion((e + distance, e + distance + size))

        finally:
            self._updating_regions = False
        self.valley_regions_changed.emit()

    
    def enforce_valley_size(self, valley_size: float) -> None:
        """Force both valley ROIs to have the same width without emitting signals.

        This method must stay silent. If this emits valley_regions_changed,
        the application can enter a recursive loop:

        region changed -> calculate -> enforce size -> region changed -> calculate ...
        """
        size = max(float(valley_size), 0.1)
        lower_min, lower_max = sorted(self.lower_region.getRegion())
        upper_min, upper_max = sorted(self.upper_region.getRegion())

        self._updating_regions = True
        try:
            self.lower_region.setRegion((lower_max - size, lower_max))
            self.upper_region.setRegion((upper_min, upper_min + size))
        finally:
            self._updating_regions = False
    
    def compton_cursor_energy(self) -> float:
        return float(self.compton_cursor.value())


    def compton_region_values(self) -> tuple[float, float]:
        a, b = self.compton_region.getRegion()
        return float(a), float(b)

    def background_region_values(self) -> tuple[float, float, float, float]:
        left_min, left_max = self.background_left_region.getRegion()
        right_min, right_max = self.background_right_region.getRegion()

        return (
            float(left_min),
            float(left_max),
            float(right_min),
            float(right_max),
        )


    def set_background_defaults_for_peak(
        self,
        peak_energy: float,
        background_size: float = 20.0,
        background_distance: float = 25.0,
    ) -> None:
        """Place two equal background ROIs around the selected peak."""
        e = float(peak_energy)
        size = max(float(background_size), 0.1)
        distance = max(float(background_distance), 0.0)

        self.background_left_region.setRegion(
            (e - distance - size, e - distance)
        )
        self.background_right_region.setRegion(
            (e + distance, e + distance + size)
        )
