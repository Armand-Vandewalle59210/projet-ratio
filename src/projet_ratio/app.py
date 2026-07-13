from __future__ import annotations

import sys
import traceback
from pathlib import Path

from PySide6.QtWidgets import QApplication, QMessageBox

from projet_ratio.main_window import MainWindow


def _install_exception_hook() -> None:
    """Show unexpected exceptions instead of letting a frozen EXE disappear."""

    def handle_exception(exc_type, exc_value, exc_traceback):
        message = "".join(traceback.format_exception(exc_type, exc_value, exc_traceback))

        try:
            Path("peak_to_valley_error.log").write_text(message, encoding="utf-8")
        except Exception:
            pass

        QMessageBox.critical(None, "Unexpected error", message)

    sys.excepthook = handle_exception


def main() -> int:
    _install_exception_hook()

    app = QApplication(sys.argv)
    app.setApplicationName("Peak-to-Valley Gamma Spectrometry")

    window = MainWindow()
    window.resize(1400, 850)
    window.show()

    return app.exec()


if __name__ == "__main__":
    raise SystemExit(main())
