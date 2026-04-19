'''
Shifts ONLY the target (LapTimeSec) so that features observed on
lap N are used to predict the lap time of lap N+1.

All physics features remain as current-lap observations (inputs).
'''

import pandas as pd
import numpy as np
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parents[1]
INPUT_PARQUET_PATH = BASE_DIR / "data" / "processed" / "training_data_with_physics.parquet"
OUTPUT_PARQUET_PATH = BASE_DIR / "data" / "processed" / "training_data_with_physics_shifted.parquet"

GROUP_KEYS = ["RaceID", "Driver"]


def add_race_lap_fraction(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add FuelProxy as RaceLapNumber / max laps within each race.

    Falls back to LapNumber if RaceLapNumber is not present.
    """
    lap_col = "RaceLapNumber" if "RaceLapNumber" in df.columns else "LapNumber"
    if lap_col not in df.columns or "RaceID" not in df.columns:
        print("  FuelProxy: skipped (missing RaceID or lap number column)")
        return df

    df = df.copy()
    max_laps = df.groupby("RaceID")[lap_col].transform("max")
    df["FuelProxy"] = (df[lap_col] / max_laps).clip(0.0, 1.0)
    print(
        "  Added FuelProxy: "
        f"mean={df['FuelProxy'].mean():.3f}, "
        f"std={df['FuelProxy'].std():.3f}"
    )
    return df

if __name__ == "__main__":
    print("Reading data from:", INPUT_PARQUET_PATH)
    df = pd.read_parquet(INPUT_PARQUET_PATH)
    initial_df = df.copy()

    print(f"Rows before: {len(df):,}")
    df = df.sort_values(GROUP_KEYS + ["LapNumber"]).reset_index(drop=True)
    df = add_race_lap_fraction(df)

    # ── ONLY shift the target ──────────────────────────────────
    # shift(-1) pulls the NEXT row's value into the current row
    # After this, row for lap N has:
    #   Features: lap N (current, observed, known)
    #   Target:   lap N+1 (the one we're predicting)
    df["LapTimeSec"] = df.groupby(GROUP_KEYS)["LapTimeSec"].shift(-1)
    print("  Shifted LapTimeSec by -1 (target is now next lap's time)")

    # DO NOT shift physics features — they are inputs
    # TyreHealth, TyreTemp_C, Psi_T, SlidingProxy, Fz_N all stay
    # as observed on the current (completed) lap
    print("  Physics features: kept as current-lap observations")

    # ── Drop rows where target is NaN ──────────────────────────
    # Last lap of each driver's race has no next-lap target
    n_before = len(df)
    df = df.dropna(subset=["LapTimeSec"])
    print(f"  Dropped last laps: {n_before:,} → {len(df):,}")

    df.to_parquet(OUTPUT_PARQUET_PATH, index=False)
    print(f"Saved: {OUTPUT_PARQUET_PATH}")

    # ── Verification ───────────────────────────────────────────
    print("\nVerification:")
    for _ in range(3):
        idx = np.random.randint(0, len(df))
        sample = df.iloc[idx]
        race_id = sample["RaceID"]
        driver = sample["Driver"]
        lap_n = int(sample["LapNumber"])

        # Original data for lap N (features should match)
        orig_n = initial_df[
            (initial_df["RaceID"] == race_id) &
            (initial_df["Driver"] == driver) &
            (initial_df["LapNumber"] == lap_n)
        ]
        # Original data for lap N+1 (target should match)
        orig_n1 = initial_df[
            (initial_df["RaceID"] == race_id) &
            (initial_df["Driver"] == driver) &
            (initial_df["LapNumber"] == lap_n + 1)
        ]

        if orig_n.empty or orig_n1.empty:
            print(f"  Sample {idx}: boundary lap, skipping")
            continue

        orig_n = orig_n.iloc[0]
        orig_n1 = orig_n1.iloc[0]
        print(f"\n  {driver} lap {lap_n}:")

        # Physics should match lap N (NOT shifted)
        for col in ["TyreHealth", "TyreTemp_C", "Psi_T", "SlidingProxy", "Fz_N"]:
            if col in df.columns:
                match = "✓" if abs(sample[col] - orig_n[col]) < 1e-6 else "✗"
                print(f"    {col}: {match} (shifted={sample[col]:.4f}, orig_lap{lap_n}={orig_n[col]:.4f})")

        # Target should match lap N+1 (shifted)
        match = "✓" if abs(sample["LapTimeSec"] - orig_n1["LapTimeSec"]) < 1e-6 else "✗"
        print(f"    LapTimeSec: {match} (shifted={sample['LapTimeSec']:.3f}, orig_lap{lap_n+1}={orig_n1['LapTimeSec']:.3f})")