"""
Regenerates the 5 analysis charts with a transparent-background, cyan/amber HUD
palette so they blend into the dark holographic website theme instead of looking
like default white matplotlib exports. Same underlying data/methodology as the
original notebook charts — only the visual styling changes.
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm

CYAN = "#FF3B3B"
CYAN_DIM = "#7A1620"
AMBER = "#F5A623"
TEXT = "#FFD9D9"
GRID = "#3A1414"
BG = "none"

plt.rcParams.update({
    "figure.facecolor": BG,
    "axes.facecolor": BG,
    "savefig.facecolor": BG,
    "savefig.transparent": True,
    "text.color": TEXT,
    "axes.edgecolor": GRID,
    "axes.labelcolor": TEXT,
    "xtick.color": TEXT,
    "ytick.color": TEXT,
    "grid.color": GRID,
    "font.family": "monospace",
    "font.size": 12,
})

DATA = "data"
OUT = "outputs/hud"
import os
os.makedirs(OUT, exist_ok=True)

races        = pd.read_csv(f"{DATA}/races.csv")
drivers      = pd.read_csv(f"{DATA}/drivers.csv")
constructors = pd.read_csv(f"{DATA}/constructors.csv")
results      = pd.read_csv(f"{DATA}/results.csv")
lap_times    = pd.read_csv(f"{DATA}/lap_times.csv")
pit_stops    = pd.read_csv(f"{DATA}/pit_stops.csv")

drivers["driver_name"] = drivers["forename"] + " " + drivers["surname"]
pit_stops["milliseconds"] = pd.to_numeric(pit_stops["milliseconds"], errors="coerce")
pit_stops = pit_stops.merge(races[["raceId", "year", "name"]], on="raceId", how="left")
pit_modern = pit_stops[pit_stops["year"] >= 2011].dropna(subset=["milliseconds"]).copy()
pit_modern["duration_s"] = pit_modern["milliseconds"] / 1000
clean_pits = pit_modern[pit_modern["duration_s"] < 60]

results_full = results.merge(races[["raceId", "year"]], on="raceId", how="left")
lap_times_full = lap_times.merge(races[["raceId", "year"]], on="raceId", how="left").merge(
    drivers[["driverId", "driver_name"]], on="driverId", how="left"
)

def style_ax(ax):
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_color(GRID)
    ax.spines["bottom"].set_color(GRID)
    ax.grid(True, alpha=0.5, linewidth=0.6)
    ax.tick_params(colors=TEXT, labelsize=11)

# ---------------------------------------------------------------- chart 1
yearly_pit = clean_pits.groupby("year")["duration_s"].median().reset_index()
fig, ax = plt.subplots(figsize=(9, 4.6))
# glow effect: layered lines with increasing alpha/decreasing width
for lw, a in [(8, 0.06), (5, 0.10), (2.5, 1.0)]:
    ax.plot(yearly_pit["year"], yearly_pit["duration_s"], color=CYAN, linewidth=lw, alpha=a, solid_capstyle="round")
ax.scatter(yearly_pit["year"], yearly_pit["duration_s"], color=CYAN, s=28, zorder=5, edgecolor="none")
ax.set_xlabel("SEASON")
ax.set_ylabel("MEDIAN PIT LANE TIME (s)")
style_ax(ax)
plt.tight_layout()
plt.savefig(f"{OUT}/01_pitstop_duration_by_year.png", dpi=170)
plt.close()

# ---------------------------------------------------------------- chart 2
pit_modern2 = pit_modern.merge(results[["raceId","driverId","constructorId"]].drop_duplicates(), on=["raceId","driverId"], how="left")
pit_modern2 = pit_modern2.merge(constructors[["constructorId","name"]].rename(columns={"name":"constructor"}), on="constructorId", how="left")
latest_full_season = int(races["year"].max()) - 1
season_pits = pit_modern2[(pit_modern2["year"] == latest_full_season) & (pit_modern2["duration_s"] < 60)]
team_speed = season_pits.groupby("constructor")["duration_s"].median().sort_values().reset_index()

fig, ax = plt.subplots(figsize=(9, 5))
colors = [CYAN if i == 0 else CYAN_DIM for i in range(len(team_speed))]
bars = ax.barh(team_speed["constructor"], team_speed["duration_s"], color=colors, height=0.6)
ax.invert_yaxis()
ax.set_xlabel("MEDIAN PIT STOP DURATION (s)")
style_ax(ax)
ax.spines["left"].set_visible(False)
plt.tight_layout()
plt.savefig(f"{OUT}/02_team_pitstop_speed.png", dpi=170)
plt.close()

# ---------------------------------------------------------------- chart 3
stops_per_driver_race = (pit_modern.groupby(["raceId","driverId"])["stop"].max().reset_index().rename(columns={"stop":"num_stops"}))
merged = results_full.merge(stops_per_driver_race, on=["raceId","driverId"], how="inner")
finished = merged[merged["position"].notna()].copy()
finished["position"] = pd.to_numeric(finished["position"], errors="coerce")
finished = finished.dropna(subset=["position"])
finished = finished[finished["num_stops"].between(1, 4)]

fig, ax = plt.subplots(figsize=(8, 4.8))
bp = ax.boxplot(
    [finished[finished["num_stops"] == n]["position"] for n in [1,2,3,4]],
    labels=["1", "2", "3", "4"], patch_artist=True, widths=0.5,
    medianprops=dict(color=AMBER, linewidth=2),
    boxprops=dict(facecolor="#0E293380", edgecolor=CYAN, linewidth=1.3),
    whiskerprops=dict(color=CYAN, linewidth=1.2),
    capprops=dict(color=CYAN, linewidth=1.2),
    flierprops=dict(markeredgecolor=CYAN_DIM, marker="o", markersize=3, alpha=0.4),
)
ax.invert_yaxis()
ax.set_xlabel("NUMBER OF PIT STOPS")
ax.set_ylabel("FINISHING POSITION")
style_ax(ax)
plt.tight_layout()
plt.savefig(f"{OUT}/03_stops_vs_position.png", dpi=170)
plt.close()

# ---------------------------------------------------------------- chart 4
season_laps = lap_times_full[(lap_times_full["year"] == latest_full_season) & (lap_times_full["lap"] > 1)].copy()
def filter_clean_laps(group):
    median = group["milliseconds"].median()
    return group[group["milliseconds"] < median * 1.15]
season_laps_clean = season_laps.groupby(["raceId","driverId"], group_keys=False).apply(filter_clean_laps, include_groups=False)
season_laps_clean = season_laps_clean.join(lap_times_full[["driver_name"]], how="left") if "driver_name" not in season_laps_clean.columns else season_laps_clean
per_race_cv = (season_laps.groupby(["driver_name","raceId"]).apply(lambda g: filter_clean_laps(g)["milliseconds"].agg(["mean","std","count"]), include_groups=False).reset_index())
per_race_cv.columns = ["driver_name","raceId","mean","std","n"]
per_race_cv = per_race_cv[per_race_cv["n"] >= 10]
per_race_cv["cv_pct"] = (per_race_cv["std"] / per_race_cv["mean"]) * 100
consistency = (per_race_cv.groupby("driver_name").agg(cv_pct=("cv_pct","mean"), n_races=("raceId","nunique")).reset_index())
consistency = consistency[consistency["n_races"] >= 5].sort_values("cv_pct").reset_index(drop=True)

fig, ax = plt.subplots(figsize=(9, 7.2))
top20 = consistency.head(20)
colors = [CYAN if i == 0 else CYAN_DIM for i in range(len(top20))]
ax.barh(top20["driver_name"], top20["cv_pct"], color=colors, height=0.62)
ax.invert_yaxis()
ax.set_xlabel("LAP TIME CV (%) — LOWER IS MORE CONSISTENT")
style_ax(ax)
ax.spines["left"].set_visible(False)
plt.tight_layout()
plt.savefig(f"{OUT}/04_driver_consistency.png", dpi=170)
plt.close()

# ---------------------------------------------------------------- chart 5
avg_finish = (results_full[results_full["year"] == latest_full_season]
              .assign(position=lambda d: pd.to_numeric(d["position"], errors="coerce"))
              .dropna(subset=["position"])
              .merge(drivers[["driverId","driver_name"]], on="driverId", how="left")
              .groupby("driver_name")["position"].mean().reset_index()
              .rename(columns={"position":"avg_finish_position"}))
combo = consistency.merge(avg_finish, on="driver_name", how="inner")

fig, ax = plt.subplots(figsize=(8, 6.2))
ax.scatter(combo["cv_pct"], combo["avg_finish_position"], color=CYAN, s=55, zorder=5, edgecolor="#04222C", linewidth=0.8)
z = np.polyfit(combo["cv_pct"], combo["avg_finish_position"], 1)
xs = np.linspace(combo["cv_pct"].min(), combo["cv_pct"].max(), 50)
ax.plot(xs, np.polyval(z, xs), color=AMBER, linewidth=2, linestyle="--")
ax.invert_yaxis()
ax.set_xlabel("LAP TIME CV (%)")
ax.set_ylabel("AVG FINISHING POSITION")
style_ax(ax)
plt.tight_layout()
plt.savefig(f"{OUT}/05_consistency_vs_results.png", dpi=170)
plt.close()

print("All 5 HUD-themed charts regenerated in", OUT)
