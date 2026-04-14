"""
Animate tyre degradation over a race from training_data_with_physics.parquet.

Examples:
    python "Python Code/animate_tyre_degradation.py" --race-id "2022_Spanish Grand Prix" --driver RIC --output "Figures/phase_c_physics/tyre_deg_RIC_spain.mp4"
    python "Python Code/animate_tyre_degradation.py" --output "Figures/phase_c_physics/tyre_deg_auto.gif"
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.animation import FuncAnimation


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Animate tyre degradation for a race")
    parser.add_argument(
        "--data",
        default="training_data_with_physics.parquet",
        help="Path to input parquet file",
    )
    parser.add_argument(
        "--race-id",
        default="2022_Spanish Grand Prix",
        help='RaceID to animate, e.g. "2022_Spanish Grand Prix"',
    )
    parser.add_argument(
        "--driver",
        default="RIC",
        help="Driver code (e.g. RIC). Default is RIC.",
    )
    parser.add_argument(
        "--output",
        default=None,
        help="Output file path (.mp4 or .gif). If omitted, shows interactive animation.",
    )
    parser.add_argument("--fps", type=int, default=12, help="Output frames per second")
    parser.add_argument("--interval", type=int, default=120, help="Frame interval (ms) for interactive mode")
    parser.add_argument("--dpi", type=int, default=140, help="Saved animation DPI")
    return parser.parse_args()


def select_driver(df_race: pd.DataFrame, requested: str | None) -> str:
    if requested:
        if requested not in set(df_race["Driver"].astype(str)):
            available = sorted(df_race["Driver"].astype(str).unique())
            raise ValueError(f"Driver '{requested}' not in race. Available: {available}")
        return requested

    counts = df_race.groupby("Driver", as_index=False).size().sort_values("size", ascending=False)
    return str(counts.iloc[0]["Driver"])


def load_series(data_path: str, race_id: str, driver: str | None) -> pd.DataFrame:
    df = pd.read_parquet(data_path)

    required = ["RaceID", "Driver", "LapNumber", "TyreHealth", "Compound", "StintNumber"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    df_race = df[df["RaceID"].astype(str) == str(race_id)].copy()
    if df_race.empty:
        sample = sorted(df["RaceID"].astype(str).unique())[:10]
        raise ValueError(f"RaceID '{race_id}' not found. Example RaceIDs: {sample}")

    chosen_driver = select_driver(df_race, driver)
    dfd = df_race[df_race["Driver"].astype(str) == chosen_driver].copy()
    dfd = dfd.sort_values("LapNumber")

    if dfd.empty:
        raise ValueError("No rows found after filtering race/driver")

    return dfd


def make_animation(dfd: pd.DataFrame, race_id: str, output: str | None, fps: int, interval: int, dpi: int) -> None:
    plt.style.use("dark_background")
    fig, ax = plt.subplots(figsize=(11, 6.2))
    fig.patch.set_facecolor("#111111")
    ax.set_facecolor("#111111")

    laps = dfd["LapNumber"].to_list()
    tyre_health = dfd["TyreHealth"].to_list()
    compounds = dfd["Compound"].astype(str).to_list()
    stints = dfd["StintNumber"].astype(int).to_list()

    driver = str(dfd.iloc[0]["Driver"])

    ax.set_xlim(min(laps) - 1, max(laps) + 1)
    y_min = min(tyre_health)
    y_max = max(tyre_health)
    pad = max(0.004, (y_max - y_min) * 0.15)
    ax.set_ylim(max(0.0, y_min - pad), min(1.02, y_max + pad))

    ax.set_xlabel("Lap Number", fontsize=12)
    ax.set_ylabel("Tyre Health q", fontsize=12)
    ax.set_title(f"Tyre Degradation Animation - {race_id} - {driver}", fontsize=14, pad=12)
    ax.grid(True, alpha=0.25)

    line, = ax.plot([], [], color="#f5f5f5", linewidth=2.5)
    scatter = ax.scatter([], [], c=[], cmap="plasma", s=35, vmin=min(stints), vmax=max(stints))
    current_dot = ax.scatter([], [], s=120, color="#ff2b2b", edgecolor="white", linewidth=1.2, zorder=5)

    # Mark stint reset points for visual clarity.
    reset_x = []
    for i in range(1, len(stints)):
        if stints[i] > stints[i - 1]:
            reset_x.append(laps[i])
    for x in reset_x:
        ax.axvline(x=x, color="#ff2b2b", linestyle="--", linewidth=1.0, alpha=0.5)

    info_text = ax.text(
        0.02,
        0.96,
        "",
        transform=ax.transAxes,
        fontsize=11,
        verticalalignment="top",
        bbox=dict(boxstyle="round", facecolor="#1f1f1f", edgecolor="#bbbbbb", alpha=0.85),
    )

    def init():
        line.set_data([], [])
        scatter.set_offsets(np.empty((0, 2)))
        scatter.set_array(pd.Series([], dtype=float).to_numpy())
        current_dot.set_offsets(np.empty((0, 2)))
        info_text.set_text("")
        return line, scatter, current_dot, info_text

    def update(frame: int):
        x = laps[: frame + 1]
        y = tyre_health[: frame + 1]
        s = stints[: frame + 1]

        line.set_data(x, y)
        scatter.set_offsets(list(zip(x, y)))
        scatter.set_array(pd.Series(s, dtype=float).to_numpy())
        current_dot.set_offsets([[x[-1], y[-1]]])

        info_text.set_text(
            f"Lap: {x[-1]}\n"
            f"Compound: {compounds[frame]}\n"
            f"Stint: {stints[frame]}\n"
            f"TyreHealth: {y[-1]:.4f}"
        )

        return line, scatter, current_dot, info_text

    anim = FuncAnimation(fig, update, init_func=init, frames=len(laps), interval=interval, blit=False, repeat=False)

    if output:
        out_path = Path(output)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        suffix = out_path.suffix.lower()

        if suffix == ".gif":
            anim.save(out_path, writer="pillow", fps=fps, dpi=dpi)
        else:
            anim.save(out_path, writer="ffmpeg", fps=fps, dpi=dpi)
        print(f"Saved animation: {out_path}")
    else:
        plt.show()


def main() -> None:
    args = parse_args()
    dfd = load_series(args.data, args.race_id, args.driver)
    make_animation(
        dfd=dfd,
        race_id=args.race_id,
        output=args.output,
        fps=args.fps,
        interval=args.interval,
        dpi=args.dpi,
    )


if __name__ == "__main__":
    main()
