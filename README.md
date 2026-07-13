# Peak-to-Valley Gamma Spectrometry

A clean PySide6 + PyQtGraph desktop application for calculating peak-to-valley ratios from gamma spectrometry spectra.

## Features

- Load `.csv`, `.n42`, `.xml`, and `.cnf` spectra when the required reader dependencies are installed.
- Display counts versus calibrated energy.
- Detect peaks with a simplified Mariscotti second-difference method and Gaussian fitting.
- Select lower and upper valley regions interactively on the plot.
- Calculate:
  - fitted peak area and uncertainty,
  - lower valley counts,
  - upper valley counts,
  - valley discontinuity,
  - peak-to-valley ratio and propagated uncertainty.

The continuum calculation from the earlier prototype has intentionally been removed to keep this first version focused.

## Project structure

```text
src/projet_ratio/
  app.py                       # application entry point
  main_window.py               # PySide6 main window
  models.py                    # shared dataclasses
  analysis/
    mariscotti.py              # peak detection and Gaussian fitting
    peak_to_valley.py          # peak-to-valley calculation
  io/
    spectrum_io.py             # CSV and SpecUtils-based spectrum loading
  ui/
    spectrum_plot.py           # PyQtGraph plotting widget
```

## Installation for development

```powershell
python -m venv .venv
.venv\Scripts\activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

If you need `.cnf`, `.n42`, or `.xml` support, install your local `SpecUtils` package as well. CSV loading works without `SpecUtils`.

## Run the app

From the repository root:

```powershell
python -m projet_ratio.app
```

If you use the `src` layout without installing the package first:

```powershell
$env:PYTHONPATH = "src"
python -m projet_ratio.app
```

## Build Windows executable

Start with a folder-based build because it is easier to debug than `--onefile`:

```powershell
pyinstaller --name PeakToValley --windowed --onedir --paths src src/projet_ratio/app.py
```

If `SpecUtils` is not detected automatically, try adding:

```powershell
pyinstaller --name PeakToValley --windowed --onedir --paths src --hidden-import SpecUtils src/projet_ratio/app.py
```

The executable will be created in `dist/PeakToValley/`.
