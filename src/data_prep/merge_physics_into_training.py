"""
merge_physics_into_training.py
==============================
Merges physics-informed features from physics_modelled_2022_2025.csv
onto training_data.parquet, producing training_data_with_physics.parquet.

This lets you run the baseline and physics-enhanced models on the
SAME dataset with the SAME filtering, so the comparison is fair.

The merge uses (RaceID, Driver, LapNumber) as join keys.
Since the two datasets use slightly different RaceID formats,
the script normalises them before joining.

Physics features added:
  - TyreHealth
  - TyreTemp_C
  - Psi_T
  - SlidingProxy
  - Fz_N

Usage:
    python merge_physics_into_training.py

Input:
    training_data.parquet
    physics_modelled_2022_2025.csv

Output:
    training_data_with_physics.parquet
    training_data_with_physics.csv  
"""

import pandas as pd
import numpy as np
import re
from pathlib import Path

PARQUET_PATH = "training_data.parquet"
PHYSICS_PATH = "physics_modelled_2022_2025.csv"
OUTPUT_PARQUET = "training_data_with_physics.parquet"
OUTPUT_CSV = "training_data_with_physics.csv"

# Physics columns to merge
PHYSICS_COLS = [
    "TyreHealth",
    "TyreTemp_C",
    "Psi_T",
    "SlidingProxy",
    "Fz_N",
]


def add_stint_number(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add StintNumber per (RaceID, Driver), incrementing when TireAge drops.

    This approximates tyre changes / new stints using lap-level data:
      - sort by LapNumber within each race-driver group
      - when TireAge decreases, start a new stint
    """
    required = ["RaceID", "Driver", "LapNumber", "TireAge"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        print(f"\n  WARNING: Cannot derive StintNumber; missing columns: {missing}")
        return df

    # Compute on a sorted copy so stint transitions follow lap order.
    work = df.copy()
    work["_orig_idx"] = np.arange(len(work))
    work = work.sort_values(["RaceID", "Driver", "LapNumber"])

    work["StintNumber"] = (
        work.groupby(["RaceID", "Driver"])["TireAge"]
        .transform(lambda x: (x.diff() < 0).cumsum() + 1)
        .astype(int)
    )

    # Restore original row order for downstream parity with parquet base.
    work = work.sort_values("_orig_idx").drop(columns=["_orig_idx"])

    print("\n── Stint feature ──")
    print(
        "  Added StintNumber "
        f"(range: {work['StintNumber'].min()}-{work['StintNumber'].max()}, "
        f"mean: {work['StintNumber'].mean():.2f})"
    )
    return work


def normalise_race_id(race_id: str) -> str:
    """
    Normalise RaceID to a common format for matching.
    
    Parquet format:  '2023_Spanish Grand Prix'
    Physics format:  '2023_Spanish Grand Prix'
    
    Both should already match, but just in case:
    - strip whitespace
    - lowercase for comparison
    """
    s = str(race_id).strip()
    return s


def extract_year_event(race_id: str):
    """Extract (year, event_name) from RaceID like '2023_Spanish Grand Prix'."""
    s = str(race_id).strip()
    # Try splitting on first underscore
    parts = s.split("_", 1)
    if len(parts) == 2:
        try:
            year = int(parts[0])
            return year, parts[1].strip()
        except ValueError:
            pass
    return None, s


def main():
    print("=" * 60)
    print("  Merging physics features into training dataset")
    print("=" * 60)
    
    # ── Load datasets ─────────────────────────────────────────
    print(f"\nLoading parquet: {PARQUET_PATH}")
    parquet_df = pd.read_parquet(PARQUET_PATH)
    print(f"  Rows: {len(parquet_df):,}")
    print(f"  Races: {parquet_df['RaceID'].nunique()}")
    print(f"  Columns: {len(parquet_df.columns)}")
    
    print(f"\nLoading physics: {PHYSICS_PATH}")
    physics_df = pd.read_csv(PHYSICS_PATH)
    print(f"  Rows: {len(physics_df):,}")
    print(f"  Races: {physics_df['RaceID'].nunique()}")
    
    # ── Check RaceID formats ─────────────────────────────────
    print("\n── RaceID format comparison ──")
    print(f"  Parquet examples:")
    for rid in sorted(parquet_df["RaceID"].unique())[:5]:
        print(f"    '{rid}'")
    
    print(f"  Physics examples:")
    for rid in sorted(physics_df["RaceID"].unique())[:5]:
        print(f"    '{rid}'")
    

    parquet_df["_merge_rid"] = parquet_df["RaceID"].apply(normalise_race_id)
    physics_df["_merge_rid"] = physics_df["RaceID"].apply(normalise_race_id)
    
    # Check overlap
    parquet_rids = set(parquet_df["_merge_rid"].unique())
    physics_rids = set(physics_df["_merge_rid"].unique())
    overlap = parquet_rids & physics_rids
    
    print(f"\n── RaceID overlap ──")
    print(f"  Parquet races:  {len(parquet_rids)}")
    print(f"  Physics races:  {len(physics_rids)}")
    print(f"  Overlapping:    {len(overlap)}")
    
    if len(overlap) == 0:
        print("\n  WARNING: No RaceID overlap! Trying fuzzy matching...")
        # Try matching on year + event name keywords
        parquet_parsed = {
            rid: extract_year_event(rid) for rid in parquet_rids
        }
        physics_parsed = {
            rid: extract_year_event(rid) for rid in physics_rids
        }
        
        print("  Parquet parsed:")
        for rid, (y, e) in list(parquet_parsed.items())[:5]:
            print(f"    {rid} -> year={y}, event='{e}'")
        print("  Physics parsed:")
        for rid, (y, e) in list(physics_parsed.items())[:5]:
            print(f"    {rid} -> year={y}, event='{e}'")
            
        mapping = {}
        for p_rid, (p_year, p_event) in physics_parsed.items():
            for t_rid, (t_year, t_event) in parquet_parsed.items():
                if p_year == t_year and (
                    p_event.lower() in t_event.lower() or
                    t_event.lower() in p_event.lower()
                ):
                    mapping[p_rid] = t_rid
                    break
        
        if mapping:
            print(f"\n  Fuzzy matched {len(mapping)} races")
            physics_df["_merge_rid"] = physics_df["_merge_rid"].map(
                lambda x: mapping.get(x, x)
            )
            overlap = set(parquet_df["_merge_rid"].unique()) & set(physics_df["_merge_rid"].unique())
            print(f"  Overlap after fuzzy match: {len(overlap)}")
        else:
            print("  No fuzzy matches found. Cannot merge.")
            return
    
    # ── Ensure LapNumber exists in both ──────────────────────
    if "LapNumber" not in parquet_df.columns:
        print("  WARNING: LapNumber not in parquet. Using index-based approach.")
        return
    
    # ── Prepare physics features for merge ───────────────────
    # Keep only the columns we need + merge keys
    available_physics = [c for c in PHYSICS_COLS if c in physics_df.columns]
    missing_physics = [c for c in PHYSICS_COLS if c not in physics_df.columns]
    
    if missing_physics:
        print(f"\n  Missing physics columns (will be NaN): {missing_physics}")
    
    print(f"  Available physics columns: {available_physics}")
    
    # Ensure Driver format matches
    parquet_df["_merge_driver"] = parquet_df["Driver"].astype(str).str.strip()
    physics_df["_merge_driver"] = physics_df["Driver"].astype(str).str.strip()
    
    # Ensure LapNumber is int
    parquet_df["_merge_lap"] = pd.to_numeric(parquet_df["LapNumber"], errors="coerce").astype("Int64")
    physics_df["_merge_lap"] = pd.to_numeric(physics_df["LapNumber"], errors="coerce").astype("Int64")
    
    # Select physics columns for merge
    physics_merge = physics_df[
        ["_merge_rid", "_merge_driver", "_merge_lap"] + available_physics
    ].copy()
    
    # Drop duplicates on merge keys (keep first)
    physics_merge = physics_merge.drop_duplicates(
        subset=["_merge_rid", "_merge_driver", "_merge_lap"],
        keep="first"
    )
    
    print(f"\n  Physics rows for merge: {len(physics_merge):,}")
    
    # ── Merge ────────────────────────────────────────────────
    n_before = len(parquet_df)
    
    merged = parquet_df.merge(
        physics_merge,
        on=["_merge_rid", "_merge_driver", "_merge_lap"],
        how="left",
        suffixes=("", "_phys"),
    )
    
    # Clean up merge columns
    merged = merged.drop(columns=["_merge_rid", "_merge_driver", "_merge_lap"], errors="ignore")
    
    # Handle any duplicate columns from merge
    for col in merged.columns:
        if col.endswith("_phys"):
            base_col = col.replace("_phys", "")
            if base_col in merged.columns:
                # Keep original, drop physics duplicate
                merged = merged.drop(columns=[col])
    
    n_after = len(merged)
    n_matched = merged[available_physics[0]].notna().sum() if available_physics else 0
    
    print(f"\n── Merge results ──")
    print(f"  Rows before: {n_before:,}")
    print(f"  Rows after:  {n_after:,}")
    print(f"  Physics matched: {n_matched:,} ({n_matched/n_after:.1%})")
    print(f"  Physics NaN:     {n_after - n_matched:,} ({(n_after-n_matched)/n_after:.1%})")
    
    # ── Fill NaN physics features with defaults ──────────────
    defaults = {
        "DamageState": 0.0,
        "TyreHealth": 1.0,
        "TyreTemp_C": 50.0,
        "Psi_T": 1.0,
        "SlidingProxy": 0.5,
        "Fz_N": 15000.0,
        "BrakeIntensity": 0.5,
        "AccelIntensity": 0.5,
        "CorneringSeverity": 0.5,
    }
    
    for col, default in defaults.items():
        if col in merged.columns:
            n_fill = merged[col].isna().sum()
            if n_fill > 0:
                merged[col] = merged[col].fillna(default)
                print(f"  Filled {n_fill:,} NaN in {col} with {default}")

    # ── Add derived stint index ──────────────────────────────
    merged = add_stint_number(merged)
    
    # ── Summary of physics features ──────────────────────────
    print(f"\n── Physics feature summary ──")
    for col in available_physics:
        if col in merged.columns:
            print(f"  {col}: mean={merged[col].mean():.4f}  "
                  f"std={merged[col].std():.4f}  "
                  f"range=[{merged[col].min():.4f}, {merged[col].max():.4f}]")
    
    # ── Save ─────────────────────────────────────────────────
    print(f"\n── Saving ──")
    
    merged.to_parquet(OUTPUT_PARQUET, index=False)
    print(f"  Saved: {OUTPUT_PARQUET} ({Path(OUTPUT_PARQUET).stat().st_size / 1e6:.1f} MB)")
    
    merged.to_csv(OUTPUT_CSV, index=False)
    print(f"  Saved: {OUTPUT_CSV} ({Path(OUTPUT_CSV).stat().st_size / 1e6:.1f} MB)")
    
    print(f"\n  Final dataset:")
    print(f"    Rows:     {len(merged):,}")
    print(f"    Columns:  {len(merged.columns)}")
    print(f"    Races:    {merged['RaceID'].nunique()}")
    print(f"    New cols: {available_physics}")
    if "StintNumber" in merged.columns:
        print("    Derived:  ['StintNumber']")
    
    print(f"\n  Done! Use '{OUTPUT_PARQUET}' as DATA_PATH in both models.")


if __name__ == "__main__":
    main()
