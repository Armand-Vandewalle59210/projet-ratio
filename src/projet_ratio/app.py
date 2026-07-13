from __future__ import annotations

import sys

from PySide6.QtWidgets import QApplication

from projet_ratio.main_window import MainWindow


def main() -> int:
    app = QApplication(sys.argv)
    app.setApplicationName("Peak-to-Valley Gamma Spectrometry")
    window = MainWindow()
    window.resize(1400, 850)
    window.show()
    return app.exec()


if __name__ == "__main__":
    raise SystemExit(main())
