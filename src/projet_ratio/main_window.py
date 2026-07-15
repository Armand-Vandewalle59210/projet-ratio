from __future__ import annotations

from pathlib import Path

from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QCheckBox,
    QDoubleSpinBox,
    QComboBox,
    QFileDialog,
    QFormLayout,
    QGroupBox,
    QLabel,
    QMainWindow,
    QMessageBox,
    QPushButton,
    QScrollArea,
    QSpinBox,
    QSplitter,
    QTableWidget,
    QTableWidgetItem,
    QTextEdit,
    QVBoxLayout,
    QWidget,
)

from projet_ratio.analysis.mariscotti import detect_peaks, nearest_peak
from projet_ratio.analysis.peak_to_valley import calculate_peak_to_valley
from projet_ratio.io.spectrum_io import load_spectrum
from projet_ratio.models import Peak, Spectrum
from projet_ratio.ui.spectrum_plot import SpectrumPlot

from projet_ratio.analysis.depth import (
    calculate_point_source_depth,
    calculate_uniform_contamination_depth,
)


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Peak-to-Valley Gamma Spectrometry")

        self.spectrum: Spectrum | None = None
        self.peaks: list[Peak] = []
        self.selected_peak: Peak | None = None
        self.last_ratio_result = None
        self.plot = SpectrumPlot()
        self.plot.valley_regions_changed.connect(self.calculate_if_possible)


        self.ratio_method = QComboBox()
        self.ratio_method.addItems([
            "Peak area / valley discontinuity",
            "Peak height / Compton height",
            "Peak area / Compton ROI area",
            "Peak height / local background height",
            "Peak area / peak area",
        ])

        self.open_button = QPushButton("Open spectrum")
        self.open_button.clicked.connect(self.open_spectrum)

        self.detect_button = QPushButton("Detect peaks")
        self.detect_button.clicked.connect(self.detect_peaks_only)

        self.calculate_button = QPushButton("Calculate ratio")
        self.calculate_button.clicked.connect(self.calculate_ratio)

        self.log_scale = QCheckBox("Log Y axis")
        self.log_scale.stateChanged.connect(self.toggle_log_scale)

        self.show_all_peaks = QCheckBox("Show all detected peak guide lines")
        self.show_all_peaks.setChecked(False)
        self.show_all_peaks.stateChanged.connect(self.toggle_all_peak_markers)

        self.target_energy = self._double_spin(0.0, 4000.0, 661.6, 1, " keV")
        self.tolerance = self._double_spin(0.1, 200.0, 6.0, 1, " keV")

        self.valley_size = self._double_spin(0.1, 500.0, 18.0, 1, " keV")
        self.valley_size.valueChanged.connect(self.apply_valley_size)
        
        self.valley_distance = self._double_spin(0.0, 500.0, 12.0, 1, " keV")
        self.valley_distance.valueChanged.connect(self.apply_valley_geometry)

        self.depth_model = QComboBox()
        self.depth_model.addItems([
            "Point source / finite depth",
            "Uniform contamination layer",
        ])

        self.k_input = self._double_spin(1e-12, 1.0, 2.26e-3, 6, " 1/keV")
        self.mu_input = self._double_spin(1e-12, 10.0, 7.85e-2, 6, " cm²/g")
        self.density_input = self._double_spin(1e-12, 50.0, 3.5, 4, " g/cm³")

        self.calculate_depth_button = QPushButton("Calculate depth")
        self.calculate_depth_button.clicked.connect(self.calculate_depth)

        self.z_input = self._int_spin(0, 50, 5)
        self.w_input = self._int_spin(1, 101, 9)
        self.sigma_factor = self._double_spin(0.1, 50.0, 2.0, 2, "")
        self.min_negative_width = self._int_spin(1, 100, 3)
        self.fit_half_width = self._int_spin(2, 200, 8)
        self.smooth_counts = QCheckBox("Smooth counts before transform")
        self.smooth_counts.setChecked(True)
        self.sg_window = self._int_spin(3, 101, 9)
        self.sg_polyorder = self._int_spin(1, 8, 3)

        self.file_label = QLabel("No file loaded")
        self.file_label.setWordWrap(True)
        self.results = QTextEdit()
        self.results.setReadOnly(True)
        self.results.setMinimumHeight(180)

        self.peaks_table = QTableWidget(0, 5)
        self.peaks_table.setHorizontalHeaderLabels(["#", "Energy keV", "Area", "FWHM keV", "Channel"])
        self.peaks_table.cellClicked.connect(self.select_peak_from_table)

        controls = self._build_controls()
        splitter = QSplitter(Qt.Orientation.Horizontal)
        splitter.addWidget(controls)
        splitter.addWidget(self.plot)
        splitter.setStretchFactor(0, 0)
        splitter.setStretchFactor(1, 1)
        splitter.setSizes([380, 1000])
        self.setCentralWidget(splitter)

    @staticmethod
    def _double_spin(minimum: float, maximum: float, value: float, decimals: int, suffix: str) -> QDoubleSpinBox:
        widget = QDoubleSpinBox()
        widget.setRange(minimum, maximum)
        widget.setValue(value)
        widget.setDecimals(decimals)
        widget.setSuffix(suffix)
        widget.setKeyboardTracking(False)
        return widget

    @staticmethod
    def _int_spin(minimum: int, maximum: int, value: int) -> QSpinBox:
        widget = QSpinBox()
        widget.setRange(minimum, maximum)
        widget.setValue(value)
        widget.setKeyboardTracking(False)
        return widget

    def _build_controls(self) -> QWidget:
        panel = QWidget()
        layout = QVBoxLayout(panel)
        layout.setContentsMargins(8, 8, 8, 8)
        layout.setSpacing(8)
        file_box = QGroupBox("File")
        file_layout = QVBoxLayout(file_box)
        file_layout.addWidget(self.open_button)
        file_layout.addWidget(self.file_label)
        file_layout.addWidget(self.log_scale)
        file_layout.addWidget(self.show_all_peaks)

        peak_box = QGroupBox("Peak and valley selection")
        peak_layout = QFormLayout(peak_box)
        peak_layout.addRow("Target energy", self.target_energy)
        peak_layout.addRow("Tolerance", self.tolerance)
        peak_layout.addRow("Valley size", self.valley_size)
        peak_layout.addRow("Valley distance", self.valley_distance)
        peak_layout.addRow("Ratio method", self.ratio_method)
        depth_box = QGroupBox("Depth calculation")
        depth_layout = QFormLayout(depth_box)
        depth_layout.addRow("Model", self.depth_model)
        depth_layout.addRow("k", self.k_input)
        depth_layout.addRow("μ", self.mu_input)
        depth_layout.addRow("Density ρ", self.density_input)
        depth_layout.addRow(self.calculate_depth_button)

        mariscotti_box = QGroupBox("Peak parameters")
        mariscotti_box.setCheckable(True)
        mariscotti_box.setChecked(False)
        mariscotti_outer_layout = QVBoxLayout(mariscotti_box)

        self.mariscotti_content = QWidget()
        mariscotti_layout = QFormLayout(self.mariscotti_content)
        mariscotti_layout.addRow("z", self.z_input)
        mariscotti_layout.addRow("w", self.w_input)
        mariscotti_layout.addRow("Sigma factor", self.sigma_factor)
        mariscotti_layout.addRow("Min negative width", self.min_negative_width)
        mariscotti_layout.addRow("Fit half-width", self.fit_half_width)
        mariscotti_layout.addRow(self.smooth_counts)
        mariscotti_layout.addRow("SG window", self.sg_window)
        mariscotti_layout.addRow("SG polyorder", self.sg_polyorder)
        self.mariscotti_content.setVisible(False)

        mariscotti_box.toggled.connect(self.mariscotti_content.setVisible)
        mariscotti_outer_layout.addWidget(self.mariscotti_content)

        layout.addWidget(file_box)
        layout.addWidget(mariscotti_box)

        layout.addWidget(self.detect_button)
        layout.addWidget(self.calculate_button)

        layout.addWidget(QLabel("Detected peaks"))
        layout.addWidget(self.peaks_table)
        layout.addWidget(peak_box)

        layout.addWidget(self.calculate_button)
        layout.addWidget(depth_box)
        layout.addWidget(QLabel("Results"))
        layout.addWidget(self.results)
        layout.addStretch(1)
        scroll_area = QScrollArea()
        scroll_area.setWidgetResizable(True)
        scroll_area.setWidget(panel)
        scroll_area.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        scroll_area.setVerticalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAsNeeded)
        # Keep the left panel usable but not ridiculously wide.
        scroll_area.setMinimumWidth(390)
        scroll_area.setMaximumWidth(480)

        return scroll_area


    def open_spectrum(self) -> None:
        path, _ = QFileDialog.getOpenFileName(
            self,
            "Open spectrum",
            "",
            "Spectrum files (*.csv *.CSV *.n42 *.N42 *.xml *.XML *.cnf *.CNF);;All files (*.*)",
        )
        if not path:
            return

        try:
            self.spectrum = load_spectrum(Path(path))
        except Exception as exc:
            self._show_error("Could not open spectrum", exc)
            return

        self.peaks = []
        self.selected_peak = None
        self.plot.clear_peaks()
        self.plot.set_spectrum(self.spectrum)
        self.peaks_table.setRowCount(0)
        live_time = "unknown" if self.spectrum.live_time is None else f"{self.spectrum.live_time:.3f} s"
        coeffs = self.spectrum.calibration_coefficients
        coeff_text = "none" if not coeffs else ", ".join(f"{c:.6g}" for c in coeffs)
        e_min = float(min(self.spectrum.energy))
        e_max = float(max(self.spectrum.energy))
        self.file_label.setText(
            f"{self.spectrum.name}\n"
            f"Channels: {len(self.spectrum.counts)}\n"
            f"Live time: {live_time}\n"
            f"Energy range: {e_min:.2f} - {e_max:.2f} keV\n"
            f"Calibration: {coeff_text}"
        )
        self.results.setText("Spectrum loaded. Click 'Detect peaks'.")

    def toggle_log_scale(self) -> None:
        self.plot.set_log_mode(self.log_scale.isChecked())
        if self.spectrum is not None:
            self.plot.set_spectrum(self.spectrum)
            self.plot.set_peaks(self.peaks, self.selected_peak)

    def toggle_all_peak_markers(self) -> None:
        self.plot.set_show_all_peak_markers(self.show_all_peaks.isChecked())
        self.plot.set_peaks(self.peaks, self.selected_peak)

    def apply_valley_size(self) -> None:
        self.plot.enforce_valley_size(self.valley_size.value())
        # Do not force a ratio calculation here unless a result already exists.
        # The Calculate button remains the explicit calculation action.

    def apply_valley_geometry(self) -> None:
        """Update valley positions from valley size and valley distance."""
        if self.selected_peak is None:
            return

        self.plot.set_region_defaults_for_peak(
            self.selected_peak.peak_energy,
            self.valley_size.value(),
            self.valley_distance.value(),
        )

    def detect_peaks_only(self) -> None:
        """Detect all peaks, update table/markers, and do not calculate ratio."""
        if self.spectrum is None:
            self.results.setText("Load a spectrum first.")
            return

        try:
            self.peaks, _, _, _ = detect_peaks(
                self.spectrum,
                z=self.z_input.value(),
                w=self.w_input.value(),
                sigma_factor=self.sigma_factor.value(),
                min_negative_width=self.min_negative_width.value(),
                fit_half_width=self.fit_half_width.value(),
                smooth_counts=self.smooth_counts.isChecked(),
                sg_window=self.sg_window.value(),
                sg_polyorder=self.sg_polyorder.value(),
            )
        except Exception as exc:
            self._show_error("Peak detection failed", exc)
            return

        self._populate_peaks_table()

        # Convenience selection only. No ratio calculation is performed here.
        self.selected_peak = None
        target_message = ""
        try:
            self.selected_peak = nearest_peak(self.peaks, self.target_energy.value(), self.tolerance.value())
            self._select_peak_row(self.selected_peak)
            
            self.plot.set_region_defaults_for_peak(
                self.selected_peak.peak_energy,
                self.valley_size.value(),
                self.valley_distance.value(),
            )

            target_message = f" Nearest target peak selected: {self.selected_peak.peak_energy:.3f} keV."
        except Exception:
            target_message = (
                f" No peak close to {self.target_energy.value():.1f} keV was auto-selected. "
                "Select a peak in the table or change the target/tolerance."
            )

        self.plot.set_peaks(self.peaks, self.selected_peak)
        self.results.setText(
            f"Detected {len(self.peaks)} fitted peaks." + target_message + "\n"
            "Click 'Calculate ratio' when the correct peak and valley regions are selected."
        )

    def _populate_peaks_table(self) -> None:
        self.peaks_table.setRowCount(len(self.peaks))
        for row, peak in enumerate(self.peaks):
            values = [
                str(peak.index),
                f"{peak.peak_energy:.3f}",
                f"{peak.area:.3f}",
                f"{peak.fwhm_energy:.3f}",
                f"{peak.peak_channel:.3f}",
            ]
            for col, value in enumerate(values):
                item = QTableWidgetItem(value)
                item.setData(Qt.ItemDataRole.UserRole, peak.index)
                self.peaks_table.setItem(row, col, item)
        self.peaks_table.resizeColumnsToContents()

    def _select_peak_row(self, peak: Peak | None) -> None:
        if peak is None:
            return
        for row, candidate in enumerate(self.peaks):
            if candidate.index == peak.index:
                self.peaks_table.selectRow(row)
                self.peaks_table.scrollToItem(self.peaks_table.item(row, 0))
                return

    def select_peak_from_table(self, row: int, column: int) -> None:
        """Select a peak and update valley placement, but do not calculate yet."""
        if row < 0 or row >= len(self.peaks):
            return
        self.selected_peak = self.peaks[row]
        self.target_energy.setValue(self.selected_peak.peak_energy)
        self.plot.set_peaks(self.peaks, self.selected_peak)
        
        self.plot.set_region_defaults_for_peak(
            self.selected_peak.peak_energy,
            self.valley_size.value(),
            self.valley_distance.value(),
        )

        self.results.setText(
            f"Selected peak: {self.selected_peak.peak_energy:.3f} keV.\n"
            "Adjust valley regions if needed, then click 'Calculate ratio'."
        )

    def calculate_if_possible(self) -> None:
        # Region dragging should not recalculate automatically anymore.
        # This keeps the workflow explicit: Detect peaks -> select/adjust -> Calculate ratio.
        return

    def calculate_ratio(self, show_errors: bool = True) -> None:
        if self.spectrum is None:
            self.results.setText("Load a spectrum first.")
            return

        if not self.peaks:
            self.results.setText("Detect peaks first, then select a peak or use a valid target energy.")
            return

        if self.selected_peak is None:
            try:
                self.selected_peak = nearest_peak(self.peaks, self.target_energy.value(), self.tolerance.value())
                self._select_peak_row(self.selected_peak)
                self.plot.set_peaks(self.peaks, self.selected_peak)
                
                self.plot.set_region_defaults_for_peak(
                    self.selected_peak.peak_energy,
                    self.valley_size.value(),
                    self.valley_distance.value(),
                )

            except Exception:
                self.results.setText(
                    "No peak is selected and no detected peak is close enough to the target energy.\n"
                    "Select one peak in the table or increase the tolerance, then click Calculate ratio again."
                )
                return

        self.plot.enforce_valley_size(self.valley_size.value())
        lower_min, lower_max = self.plot.lower_region_values()
        upper_min, upper_max = self.plot.upper_region_values()

        try:
            result = calculate_peak_to_valley(
                self.spectrum,
                self.selected_peak,
                lower_min,
                lower_max,
                upper_min,
                upper_max,
            )
            self.last_ratio_result = result
        except Exception as exc:
            if show_errors:
                self._show_error("Calculation failed", exc)
            else:
                self.results.setText(str(exc))
            return
        inverse_ratio=1.0/result.ratio
        inverse_ratio_uncertainty= result.ratio_uncertainty/(result.ratio**2)
        self.results.setText(
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
            f"valley to peak ratio 1/Q: {inverse_ratio:.6g} ± {inverse_ratio_uncertainty:.3g}\n"
            f"Acceptability lower/upper ≥ 2: {result.acceptable}"
        )
    def calculate_depth(self) -> None:
        """Calculate depth from the latest peak-to-valley result."""
        if self.last_ratio_result is None:
            self.calculate_ratio(show_errors=False)

        if self.last_ratio_result is None:
            self.results.setText(
                "Calculate the peak-to-valley ratio first. Depth needs a valid ratio result."
            )
            return

        result = self.last_ratio_result

        # Your app ratio is Q = peak / valley.
        q_peak_over_valley = float(result.ratio)

        # The theory uses R = valley / peak.
        discontinuity_over_peak = 1.0 / q_peak_over_valley
        sigma_discontinuity_over_peak = result.ratio_uncertainty / (q_peak_over_valley**2)

        k = self.k_input.value()
        mu = self.mu_input.value()
        density = self.density_input.value()

        try:
            if self.depth_model.currentIndex() == 0:
                depth, depth_uncertainty = calculate_point_source_depth(
                    peak_to_valley_ratio=q_peak_over_valley,
                    peak_to_valley_uncertainty=result.ratio_uncertainty,
                    k=k,
                    mu=mu,
                    density=density,
                )
                model_name = "Point source / finite depth"
            else:
                depth, depth_uncertainty = calculate_uniform_contamination_depth(
                    discontinuity_to_peak_ratio=discontinuity_over_peak,
                    discontinuity_to_peak_uncertainty=sigma_discontinuity_over_peak,
                    k=k,
                    mu=mu,
                    density=density,
                )
                model_name = "Uniform contamination layer"

        except Exception as exc:
            self._show_error("Depth calculation failed", exc)
            return

        previous = self.results.toPlainText().rstrip()

        self.results.setText(
            previous
            + "\n\n"
            + f"Depth model: {model_name}\n"
            + f"Discontinuity/peak ratio: {discontinuity_over_peak:.6g} ± {sigma_discontinuity_over_peak:.3g}\n"
            + f"k: {k:.6g} 1/keV\n"
            + f"μ: {mu:.6g} cm²/g\n"
            + f"ρ: {density:.6g} g/cm³\n"
            + f"Depth: {depth:.6g} ± {depth_uncertainty:.3g} cm"
        )

    def _show_error(self, title: str, exc: Exception) -> None:
        QMessageBox.critical(self, title, str(exc))
        self.results.setText(f"{title}:\n{exc}")
