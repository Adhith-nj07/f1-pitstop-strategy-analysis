"""
Downloads the F1 CSV tables used by the notebook into ./data

Source: https://github.com/TracingInsights/RaceData (CC0 / public domain),
an auto-updated mirror of Ergast/Jolpica F1 data.

Usage (from the repo root):
    python scripts/download_data.py
"""
import sys
import urllib.request
from pathlib import Path

BASE = "https://raw.githubusercontent.com/TracingInsights/RaceData/main/data"
TABLES = [
    "races", "drivers", "constructors", "results", "lap_times",
    "pit_stops", "status", "circuits", "qualifying",
]

def main():
    out = Path(__file__).resolve().parent.parent / "data"
    out.mkdir(exist_ok=True)
    for name in TABLES:
        dest = out / f"{name}.csv"
        print(f"Downloading {name}.csv ...", end=" ", flush=True)
        try:
            urllib.request.urlretrieve(f"{BASE}/{name}.csv", dest)
        except Exception as exc:
            print(f"FAILED ({exc})")
            sys.exit(1)
        print(f"{dest.stat().st_size / 1e6:.1f} MB")
    print("Done. You can now run the notebook.")

if __name__ == "__main__":
    main()
