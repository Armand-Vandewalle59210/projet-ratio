$ErrorActionPreference = "Stop"

python -m pip install --upgrade pip
python -m pip install -r requirements.txt
pyinstaller --name PeakToValley --windowed --onedir --paths src src/projet_ratio/app.py

Write-Host "Build finished. Check dist/PeakToValley/"
