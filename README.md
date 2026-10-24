# F1 Pit Stop Strategy & Driver Consistency Analysis

**Live demo:** https://adhith-nj07.github.io/f1-pitstop-strategy-analysis/ &nbsp;·&nbsp;
**Notebook:** [`notebooks/f1_pitstop_strategy_analysis.ipynb`](notebooks/f1_pitstop_strategy_analysis.ipynb)

A data science project analyzing real Formula 1 race data (76 seasons, 1,172 races, 880K+ lap
records, 22K+ pit stops) with pandas, plus an interactive HUD-style web report of the findings.

## Questions answered

1. Have pit crews gotten faster over time, and which teams are fastest today?
2. Does the number of pit stops in a race relate to finishing position?
3. Which drivers are the most consistent lap-to-lap, and does consistency predict results?

## Key findings

- Median pit lane time fell from ~24.2s (2014–15) to ~23.1s (2025), with a COVID-era spike in 2020.
  Fastest vs. slowest crew in 2025 differ by about one second (Sauber 22.60s vs. Aston Martin 23.61s).
- One-stop races average a better finish (8.9) than four-stop races (10.5) — a correlation, not a
  strategy rule: the dataset has no tire, weather, or safety-car data.
- Lap-time consistency (avg. per-race CV, 1.9–2.3% across the 2025 grid) shows essentially **no
  correlation** with average finishing position (r ≈ 0.05): consistency looks like a threshold skill,
  not a differentiator.

## Method notes (the details that matter)

- **Pit "duration" is full pit-lane time** (~20–25s), not the ~2–3s stationary stop shown on TV.
- **Consistency is computed within each race first, then averaged.** Pooling laps across circuits
  mixes track length with driver variability and inflates the metric several-fold (an early bug in
  this project, since fixed).
- Laps >15% slower than a driver's race median (pit laps, safety cars, traffic) and lap 1 are excluded.

## Quick start

```bash
git clone https://github.com/Adhith-nj07/f1-pitstop-strategy-analysis.git
cd f1-pitstop-strategy-analysis
pip install -r requirements.txt
python scripts/download_data.py        # fetches the CSVs into ./data (~28 MB)
jupyter notebook notebooks/f1_pitstop_strategy_analysis.ipynb
```

## Project structure

```
├── docs/index.html              # self-contained web report (served by GitHub Pages)
├── notebooks/                   # main analysis notebook (executed, outputs included)
├── scripts/
│   ├── download_data.py         # downloads the source CSVs
│   └── fetch_live_telemetry.py  # optional: FastF1 telemetry (run locally, needs open internet)
├── generate_hud_charts.py       # re-renders the charts in the website's theme -> outputs/hud/
├── build_notebook.py            # generates the notebook programmatically
├── outputs/                     # exported chart PNGs
└── data/                        # populated by download_data.py (git-ignored)
```

## Data

[TracingInsights/RaceData](https://github.com/TracingInsights/RaceData) — CC0, an auto-updated
mirror of Ergast/Jolpica F1 data. Not affiliated with Formula 1 or the FIA.

## License

MIT — see [LICENSE](LICENSE).
