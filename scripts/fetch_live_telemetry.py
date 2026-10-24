"""
Optional extension: pull real car telemetry (speed, throttle, brake, gear, tire
compound) using the FastF1 library, which talks directly to F1's live timing
service. This requires unrestricted internet access, so run it on your own
machine (it will NOT work in a network-locked sandbox/CI environment).

Usage:
    pip install fastf1
    python scripts/fetch_live_telemetry.py --year 2024 --gp Monza --session R

This saves:
    - lap and tire compound data to data/telemetry/{year}_{gp}_{session}_laps.csv
    - a speed-trace comparison plot for the fastest lap of the top 2 finishers
"""
import argparse
import os

import fastf1
import fastf1.plotting
import matplotlib.pyplot as plt


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--year", type=int, default=2024)
    parser.add_argument("--gp", type=str, default="Monza")
    parser.add_argument("--session", type=str, default="R", help="R=Race, Q=Qualifying, FP1/FP2/FP3")
    args = parser.parse_args()

    os.makedirs("data/telemetry_cache", exist_ok=True)
    os.makedirs("data/telemetry", exist_ok=True)
    os.makedirs("outputs", exist_ok=True)

    fastf1.Cache.enable_cache("data/telemetry_cache")

    session = fastf1.get_session(args.year, args.gp, args.session)
    session.load()  # pulls lap times, telemetry, tire data, weather from F1's live timing API

    laps = session.laps
    out_csv = f"data/telemetry/{args.year}_{args.gp}_{args.session}_laps.csv".replace(" ", "_")
    laps.to_csv(out_csv, index=False)
    print(f"Saved {len(laps)} laps (with tire compound + stint data) to {out_csv}")

    # Speed trace comparison for the top two finishers' fastest laps
    top_drivers = laps["Driver"].unique()[:2]
    fig, ax = plt.subplots(figsize=(10, 5))
    for drv in top_drivers:
        fastest = laps.pick_drivers(drv).pick_fastest()
        tel = fastest.get_car_data().add_distance()
        ax.plot(tel["Distance"], tel["Speed"], label=drv)

    ax.set_xlabel("Distance (m)")
    ax.set_ylabel("Speed (km/h)")
    ax.set_title(f"Fastest Lap Speed Trace — {args.year} {args.gp} ({args.session})")
    ax.legend()
    plt.tight_layout()
    out_png = f"outputs/speed_trace_{args.year}_{args.gp}_{args.session}.png".replace(" ", "_")
    plt.savefig(out_png, dpi=150)
    print(f"Saved speed trace plot to {out_png}")


if __name__ == "__main__":
    main()
