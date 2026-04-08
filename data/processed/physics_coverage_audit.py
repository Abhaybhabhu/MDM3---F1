"""
Audit coverage between training and physics datasets.

Purpose:
- Identify which training join keys do not exist in physics data.
- Break missing coverage down by race, driver, and lap.
- Flag likely formatting mismatches (case/whitespace differences).
- Export CSV reports you can use to backfill or fix key formatting.

Usage:
    python data/processed/physics_coverage_audit.py

Optional arguments:
    --training path/to/training_data.parquet
    --physics path/to/physics_modelled_2022_2025.parquet
    --output-dir path/to/output_reports
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Iterable

import pandas as pd


JOIN_KEYS = ["RaceID", "Driver", "LapNumber"]


def load_parquet_robust(path: Path) -> pd.DataFrame:
    """Load parquet with engine fallbacks for compatibility."""
    errors: list[str] = []

    try:
        return pd.read_parquet(path)
    except Exception as exc:
        errors.append(f"pandas.read_parquet(default) failed: {exc}")

    try:
        return pd.read_parquet(path, engine="pyarrow")
    except Exception as exc:
        errors.append(f"pandas.read_parquet(engine='pyarrow') failed: {exc}")

    try:
        return pd.read_parquet(path, engine="fastparquet")
    except Exception as exc:
        errors.append(f"pandas.read_parquet(engine='fastparquet') failed: {exc}")

    msg = [
        f"Could not read parquet file: {path}",
        "Tried:",
        *[f"  - {e}" for e in errors],
    ]
    raise RuntimeError("\n".join(msg))


def normalize_text(series: pd.Series) -> pd.Series:
    """Normalize text for mismatch detection (not exact join logic)."""
    return (
        series.astype("string")
        .str.strip()
        .str.lower()
        .str.replace(r"\s+", " ", regex=True)
    )


def ensure_columns(df: pd.DataFrame, required: Iterable[str], name: str) -> None:
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"{name} is missing required columns: {missing}")


def cast_join_types(df: pd.DataFrame) -> pd.DataFrame:
    """Cast join keys to stable types used for key comparison."""
    out = df.copy()
    out["RaceID"] = out["RaceID"].astype("string").str.strip()
    out["Driver"] = out["Driver"].astype("string").str.strip()

    # Keep nullable integer to avoid dropping bad rows silently.
    out["LapNumber"] = pd.to_numeric(out["LapNumber"], errors="coerce").astype("Int64")
    return out


def add_missing_reason(missing_df: pd.DataFrame, phys_keys: pd.DataFrame) -> pd.DataFrame:
    """Classify why each key is missing from physics."""
    out = missing_df.copy()

    phys_races = phys_keys[["RaceID"]].drop_duplicates().assign(_race_present=True)
    phys_race_driver = (
        phys_keys[["RaceID", "Driver"]]
        .drop_duplicates()
        .assign(_race_driver_present=True)
    )

    out = out.merge(phys_races, on="RaceID", how="left")
    out = out.merge(phys_race_driver, on=["RaceID", "Driver"], how="left")

    out["_race_present"] = out["_race_present"].notna()
    out["_race_driver_present"] = out["_race_driver_present"].notna()

    out["missing_reason"] = "LapNumber missing for RaceID+Driver"
    out.loc[~out["_race_present"], "missing_reason"] = "RaceID missing in physics"
    out.loc[out["_race_present"] & ~out["_race_driver_present"], "missing_reason"] = (
        "Driver missing for RaceID in physics"
    )

    return out.drop(columns=["_race_present", "_race_driver_present"])


def detect_formatting_mismatches(missing_df: pd.DataFrame, phys_keys: pd.DataFrame) -> pd.DataFrame:
    """Find missing exact keys that would match after light text normalization."""
    m = missing_df.copy()
    p = phys_keys.copy()

    m["RaceID_norm"] = normalize_text(m["RaceID"])
    m["Driver_norm"] = normalize_text(m["Driver"])

    p["RaceID_norm"] = normalize_text(p["RaceID"])
    p["Driver_norm"] = normalize_text(p["Driver"])

    norm_phys = p[["RaceID_norm", "Driver_norm", "LapNumber"]].drop_duplicates()
    m = m.merge(
        norm_phys.assign(_norm_match=True),
        on=["RaceID_norm", "Driver_norm", "LapNumber"],
        how="left",
    )
    m["possible_formatting_mismatch"] = m["_norm_match"].notna()

    return m.drop(columns=["_norm_match"])


def run_audit(training_path: Path, physics_path: Path, output_dir: Path) -> None:
    print("=" * 72)
    print("PHYSICS COVERAGE AUDIT")
    print("=" * 72)

    print(f"\nLoading training dataset: {training_path}")
    train_df = load_parquet_robust(training_path)
    print(f"  Rows: {len(train_df):,}  Cols: {len(train_df.columns)}")

    print(f"\nLoading physics dataset:  {physics_path}")
    phys_df = load_parquet_robust(physics_path)
    print(f"  Rows: {len(phys_df):,}  Cols: {len(phys_df.columns)}")

    ensure_columns(train_df, JOIN_KEYS, "Training dataset")
    ensure_columns(phys_df, JOIN_KEYS, "Physics dataset")

    train_keys = cast_join_types(train_df[JOIN_KEYS]).drop_duplicates()
    phys_keys = cast_join_types(phys_df[JOIN_KEYS]).drop_duplicates()

    # Expose rows with invalid LapNumber casting.
    train_bad_lap = train_keys[train_keys["LapNumber"].isna()].copy()
    phys_bad_lap = phys_keys[phys_keys["LapNumber"].isna()].copy()

    train_keys = train_keys[train_keys["LapNumber"].notna()].copy()
    phys_keys = phys_keys[phys_keys["LapNumber"].notna()].copy()

    print(f"\nUnique training keys: {len(train_keys):,}")
    print(f"Unique physics keys:  {len(phys_keys):,}")

    merged = train_keys.merge(
        phys_keys.assign(_in_physics=True),
        on=JOIN_KEYS,
        how="left",
    )

    missing = merged[merged["_in_physics"].isna()].drop(columns=["_in_physics"]).copy()
    matched = len(train_keys) - len(missing)
    match_pct = (matched / len(train_keys) * 100.0) if len(train_keys) else 0.0

    print("\nCoverage against training keys:")
    print(f"  Matched:   {matched:,} ({match_pct:.1f}%)")
    print(f"  Missing:   {len(missing):,} ({100.0 - match_pct:.1f}%)")

    missing = add_missing_reason(missing, phys_keys)
    missing = detect_formatting_mismatches(missing, phys_keys)

    # Physics-only keys can reveal over-generation or filtering mismatch.
    physics_only = phys_keys.merge(train_keys.assign(_in_training=True), on=JOIN_KEYS, how="left")
    physics_only = physics_only[physics_only["_in_training"].isna()].drop(columns=["_in_training"])

    # Summaries.
    missing_by_race = (
        missing.groupby("RaceID", dropna=False)
        .size()
        .rename("missing_keys")
        .reset_index()
        .sort_values("missing_keys", ascending=False)
    )

    missing_by_race_driver = (
        missing.groupby(["RaceID", "Driver"], dropna=False)
        .size()
        .rename("missing_keys")
        .reset_index()
        .sort_values("missing_keys", ascending=False)
    )

    missing_by_reason = (
        missing.groupby("missing_reason", dropna=False)
        .size()
        .rename("count")
        .reset_index()
        .sort_values("count", ascending=False)
    )

    possible_format_mismatch = missing[missing["possible_formatting_mismatch"]].copy()

    output_dir.mkdir(parents=True, exist_ok=True)

    paths = {
        "missing_keys": output_dir / "missing_training_keys_in_physics.csv",
        "missing_by_race": output_dir / "missing_by_race.csv",
        "missing_by_race_driver": output_dir / "missing_by_race_driver.csv",
        "missing_by_reason": output_dir / "missing_by_reason.csv",
        "possible_format_mismatch": output_dir / "possible_formatting_mismatches.csv",
        "physics_only": output_dir / "physics_keys_not_in_training.csv",
        "train_bad_lap": output_dir / "training_invalid_lapnumber_rows.csv",
        "phys_bad_lap": output_dir / "physics_invalid_lapnumber_rows.csv",
    }

    missing.to_csv(paths["missing_keys"], index=False)
    missing_by_race.to_csv(paths["missing_by_race"], index=False)
    missing_by_race_driver.to_csv(paths["missing_by_race_driver"], index=False)
    missing_by_reason.to_csv(paths["missing_by_reason"], index=False)
    possible_format_mismatch.to_csv(paths["possible_format_mismatch"], index=False)
    physics_only.to_csv(paths["physics_only"], index=False)
    train_bad_lap.to_csv(paths["train_bad_lap"], index=False)
    phys_bad_lap.to_csv(paths["phys_bad_lap"], index=False)

    print("\nTop missing races:")
    if len(missing_by_race):
        for _, row in missing_by_race.head(10).iterrows():
            print(f"  {row['RaceID']}: {int(row['missing_keys']):,}")
    else:
        print("  None")

    print("\nMissing reason breakdown:")
    for _, row in missing_by_reason.iterrows():
        print(f"  {row['missing_reason']}: {int(row['count']):,}")

    print("\nPotential formatting mismatches (exact-miss but normalized-hit):")
    print(f"  {len(possible_format_mismatch):,}")

    print("\nPhysics keys not used by training:")
    print(f"  {len(physics_only):,}")

    print("\nReports written:")
    for path in paths.values():
        print(f"  {path}")


def main() -> None:
    script_dir = Path(__file__).resolve().parent

    parser = argparse.ArgumentParser(description="Audit missing physics join coverage.")
    parser.add_argument(
        "--training",
        type=Path,
        default=script_dir / "training_data.parquet",
        help="Path to training parquet",
    )
    parser.add_argument(
        "--physics",
        type=Path,
        default=script_dir / "physics_modelled_2022_2025.parquet",
        help="Path to physics parquet",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=script_dir / "audit_outputs",
        help="Directory to write CSV reports",
    )
    args = parser.parse_args()

    run_audit(args.training, args.physics, args.output_dir)


if __name__ == "__main__":
    main()
