"""
pace_model_physics.py
=====================
PHYSICS-ENHANCED version of the CatBoost pace model.

This is identical to pace_model.py (v2) but adds physics-informed
tyre features from the thermal degradation model:
  - DamageState     (1 - TyreHealth, cumulative tyre wear)
  - TyreTemp_C      (latent tyre temperature from thermal model)
  - SlidingProxy     (telemetry-derived driving intensity proxy)
  - Fz_N            (estimated vertical load)

The purpose is to test whether physics-informed features improve
lap-time prediction compared to the baseline model that uses only
compound, tyre age, and race context features.

This is the key ablation test for the MDM3 project:
  Baseline model  = compound + tyre age + race context
  Physics model   = baseline + DamageState + TyreTemp + SlidingProxy + Fz

Usage:
    python pace_model_physics.py

Input:
    physics_modelled_2022_2025.csv — from build_physics_dataset_all_seasons.py

Output:
    models/pace_model_physics.cbm
    models/track_baselines_physics.json
    models/training_report_physics.txt
"""

import json
import pathlib
import time
from typing import Any, Optional, cast

import numpy as np
import pandas as pd
from catboost import CatBoostRegressor, Pool
from sklearn.model_selection import GroupKFold
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score,
)

try:
    import optuna

    OPTUNA_AVAILABLE = True
except ImportError:
    OPTUNA_AVAILABLE = False
    print(
        "   Optuna not installed. Hyperparameter tuning "
        "will be skipped. Install with: pip install optuna"
    )


# ═══════════════════════════════════════════════════════════════
# 1. GPU DETECTION
# ═══════════════════════════════════════════════════════════════

def detect_gpu() -> bool:
    """Detect CUDA GPU by attempting a tiny CatBoost fit."""
    print("  Detecting GPU availability…")

    try:
        import subprocess

        result = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=name,memory.total",
                "--format=csv,noheader",
            ],
            capture_output=True,
            text=True,
            timeout=5,
        )
        if result.returncode == 0 and result.stdout.strip():
            for i, line in enumerate(
                result.stdout.strip().split("\n")
            ):
                print(f"   GPU {i}: {line.strip()}")
        else:
            print("     No NVIDIA GPU detected")
            return False
    except (FileNotFoundError, subprocess.TimeoutExpired):
        print("     nvidia-smi not found")
        return False

    try:
        CatBoostRegressor(
            iterations=5,
            task_type="GPU",
            devices="0",
            verbose=0,
        ).fit(
            np.array([[1, 2], [3, 4], [5, 6]]),
            np.array([1, 2, 3]),
        )
        print("     GPU training verified")
        return True
    except Exception as e:
        print(f"      GPU fit failed: {e}")
        return False


# ═══════════════════════════════════════════════════════════════
# 2. CONFIGURATION
# ═══════════════════════════════════════════════════════════════

# ── Paths ─────────────────────────────────────────────────────
DATA_PATH = "training_data_with_physics.parquet"
MODEL_DIR = pathlib.Path("models")
MODEL_DIR.mkdir(exist_ok=True)
MODEL_PATH = MODEL_DIR / "pace_model_physics.cbm"
BASELINES_PATH = MODEL_DIR / "track_baselines_physics.json"
REPORT_PATH = MODEL_DIR / "training_report_physics.txt"

# ── Target ────────────────────────────────────────────────────
RAW_TARGET = "LapTimeSec"
DELTA_TARGET = "LapTimeDelta"

# ── Outlier Filtering ─────────────────────────────────────────
# Removes safety car laps, VSC periods, pit anomalies, and
# other unpredictable events that the model cannot learn from.
# The v1 model had deltas ranging from -6.4s to +46.2s — the
# +46s tails are physically meaningless for pace prediction.
OUTLIER_CONFIG: dict[str, Any] = {
    "lower_quantile": 0.001,
    "upper_quantile": 0.995,
    "hard_cap_seconds": 15.0,
}

# ── Baseline Mode ─────────────────────────────────────────────
# "track"        — per-TrackName median (original, 22 groups)
# "track_season" — per-TrackName-per-Season median (~44-66 groups)
#
# "track_season" is preferred when the dataset spans multiple
# regulation eras (e.g. 2022 ground-effect vs 2023/2024).
# The per-track median across seasons pools fundamentally
# different cars, inflating delta variance.
BASELINE_MODE = "track_season"  # or "track"

# ── Features ──────────────────────────────────────────────────
CATEGORICAL_FEATURES = [
    "Compound",
    "TrackName",       # aliased from Location in load_data
    "Driver",
    "Team",
]

# CircuitLength and NumberOfCorners are REMOVED (v1 fix).
# RaceLapFraction ADDED (v2 fix) — normalises race progress
# to [0, 1], generalising across tracks with different total
# laps better than raw RaceLapNumber.
#
# PHYSICS FEATURES ADDED (v3 — physics-enhanced):
#   DamageState   — cumulative tyre wear from thermal model (1 - TyreHealth)
#   TyreTemp_C    — latent tyre temperature from thermal model
#   SlidingProxy  — telemetry-derived driving intensity proxy
#   Fz_N          — estimated vertical load from speed + aero model
NUMERICAL_FEATURES = [
    "TireAge",
    "TireAgeSq",
    "RaceLapNumber",
    "RaceLapFraction",
    "TrackTemp",
    "AirTemp",
    "Humidity",
    "WindSpeed",
    "GapToCarAhead",
    "DRS_Available",
    # ── Physics-informed features ──
    "DamageState",
    "TyreTemp_C",
    "SlidingProxy",
    "Fz_N",
]

ALL_FEATURES = CATEGORICAL_FEATURES + NUMERICAL_FEATURES

GROUP_KEY = "RaceID"

# ── GPU ───────────────────────────────────────────────────────
GPU_AVAILABLE = detect_gpu()
TASK_TYPE = "GPU" if GPU_AVAILABLE else "CPU"

# ── CatBoost Defaults ────────────────────────────────────────
# v2: Conservative defaults to prevent the 5-iteration collapse.
#   - learning_rate 0.05 → 0.02 (slower, more trees)
#   - depth 8 → 6 (shallower, less memorisation)
#   - l2_leaf_reg added (explicit regularisation)
#   - min_data_in_leaf added (prevents tiny leaves)
DEFAULT_PARAMS: dict[str, Any] = {
    "iterations": 2500,
    "learning_rate": 0.02,
    "depth": 6,
    "l2_leaf_reg": 10.0,
    "min_data_in_leaf": 50,
    "loss_function": "RMSE",
    "verbose": 200,
    "random_seed": 42,
    "posterior_sampling": True,
    "langevin": True,
}

GPU_OVERRIDES: dict[str, Any] = {"border_count": 128}

# ── Tuning Configuration ─────────────────────────────────────
# v2 changes:
#   - subsample_frac 0.6 → 1.0 (hardware can handle it,
#     eliminates subsample/fulldata mismatch)
#   - n_trials 25 → 30 (more exploration with full data)
#   - max_iterations 1500 → 2000 (conservative LR needs more)
TUNING_CONFIG: dict[str, Any] = {
    "n_trials": 30,
    "n_splits": 2,
    "subsample_frac": 1.0,
    "max_iterations": 2000,
    "early_stopping_rounds": 80,
}

CV_CONFIG: dict[str, Any] = {
    "n_splits": 3,
    "early_stopping_rounds": 100,
}


def _make_cv_params(
    params: dict[str, Any],
    use_gpu: bool = True,
) -> dict[str, Any]:
    """Build params safe for CV / tuning (no CPU-only keys)."""
    p = {**params}
    p.pop("posterior_sampling", None)
    p.pop("langevin", None)
    if use_gpu and GPU_AVAILABLE:
        p["task_type"] = "GPU"
        p.update(GPU_OVERRIDES)
    else:
        p["task_type"] = "CPU"
    return p


# ═══════════════════════════════════════════════════════════════
# 3. DATA LOADING & BASELINE NORMALISATION
# ═══════════════════════════════════════════════════════════════

def load_data(path: str = DATA_PATH) -> pd.DataFrame:
    """Load and validate the training dataset."""
    print(f"\n  Loading data from: {path}")
    df = (
        pd.read_parquet(path)
        if path.endswith(".parquet")
        else pd.read_csv(path)
    )
    print(f"   Rows: {len(df):,}")

    # ── Physics dataset compatibility ─────────────────────────
    # The physics dataset uses 'Location' instead of 'TrackName'
    # and 'LapTime_s' instead of 'LapTimeSec'.
    # Create aliases so the rest of the pipeline works unchanged.
    if "TrackName" not in df.columns and "Location" in df.columns:
        df["TrackName"] = df["Location"]
        print("   Aliased Location -> TrackName")

    if "LapTimeSec" not in df.columns and "LapTime_s" in df.columns:
        df["LapTimeSec"] = df["LapTime_s"]

    # Ensure TireAgeSq exists
    if "TireAgeSq" not in df.columns and "TireAge" in df.columns:
        df["TireAgeSq"] = df["TireAge"] ** 2
        print("   Created TireAgeSq from TireAge")

    # Ensure RaceLapNumber exists
    if "RaceLapNumber" not in df.columns and "LapNumber" in df.columns:
        df["RaceLapNumber"] = df["LapNumber"]
        print("   Aliased LapNumber -> RaceLapNumber")

    # Ensure GapToCarAhead exists (physics dataset may not have it)
    if "GapToCarAhead" not in df.columns:
        df["GapToCarAhead"] = 10.0  # default: no car close ahead
        print("   GapToCarAhead not found, defaulting to 10.0")

    # Ensure DRS_Available exists
    if "DRS_Available" not in df.columns:
        if "DRSDetected" in df.columns:
            df["DRS_Available"] = df["DRSDetected"].fillna(0).astype(int)
            print("   Created DRS_Available from DRSDetected")
        else:
            df["DRS_Available"] = 0
            print("   DRS_Available not found, defaulting to 0")

    # Fill NaN in physics features with sensible defaults
    physics_fill = {
        "DamageState": 0.0,
        "TyreTemp_C": 50.0,
        "SlidingProxy": 0.5,
        "Fz_N": 15000.0,
    }
    for col, default in physics_fill.items():
        if col in df.columns:
            n_nan = df[col].isna().sum()
            if n_nan > 0:
                df[col] = df[col].fillna(default)
                print(f"   Filled {n_nan} NaN in {col} with {default}")

    # Check for required columns (excluding engineered ones)
    base_required = [
        c for c in ALL_FEATURES
        if c != "RaceLapFraction"   # engineered later
    ] + [RAW_TARGET, GROUP_KEY]
    missing = [c for c in base_required if c not in df.columns]
    if missing:
        raise ValueError(f"Missing columns: {missing}")

    for col in CATEGORICAL_FEATURES:
        df[col] = df[col].astype(str)

    print(f"\n   Raw target ({RAW_TARGET}):")
    print(f"     Mean:  {df[RAW_TARGET].mean():.2f}s  "
          f"Std: {df[RAW_TARGET].std():.2f}s")
    print(f"     Range: {df[RAW_TARGET].min():.2f}s – "
          f"{df[RAW_TARGET].max():.2f}s")
    print(f"   Groups ({GROUP_KEY}): "
          f"{df[GROUP_KEY].nunique()} races")

    # Print physics feature summary
    print(f"\n   Physics features:")
    for col in ["DamageState", "TyreTemp_C", "SlidingProxy", "Fz_N"]:
        if col in df.columns:
            print(f"     {col}: mean={df[col].mean():.4f}  "
                  f"std={df[col].std():.4f}  "
                  f"range=[{df[col].min():.4f}, {df[col].max():.4f}]")

    return df


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add engineered features that improve generalisation.

    RaceLapFraction (v2):
        Normalises RaceLapNumber to [0, 1] within each race.
        This generalises across tracks with different total
        laps (e.g. Monaco ~78 laps vs Spa ~44 laps).
        Raw RaceLapNumber conflates "lap 30" at Monaco
        (mid-race) with "lap 30" at Spa (late-race).

    Season (for baseline computation):
        Extracted from RaceID or date if available.
    """
    df = df.copy()

    # ── RaceLapFraction ───────────────────────────────────────
    if "RaceLapFraction" not in df.columns:
        max_laps_per_race = df.groupby(GROUP_KEY)[
            "RaceLapNumber"
        ].transform("max")
        df["RaceLapFraction"] = (
            df["RaceLapNumber"] / max_laps_per_race
        )
        df["RaceLapFraction"] = df["RaceLapFraction"].clip(
            0.0, 1.0
        )
        print(f"\n🔧 Engineered features:")
        print(f"   RaceLapFraction: "
              f"mean={df['RaceLapFraction'].mean():.3f}, "
              f"std={df['RaceLapFraction'].std():.3f}")

    # ── Season ────────────────────────────────────────────────
    if "Season" not in df.columns:
        if "RaceDate" in df.columns:
            df["Season"] = pd.to_datetime(
                df["RaceDate"]
            ).dt.year
            print(f"   Season extracted from RaceDate: "
                  f"{sorted(df['Season'].unique())}")
        elif "Year" in df.columns:
            df["Season"] = df["Year"].astype(int)
            print(f"   Season from Year column: "
                  f"{sorted(df['Season'].unique())}")
        else:
            # Attempt to extract from RaceID if it contains
            # a year pattern (e.g. "2023_Bahrain_R")
            try:
                df["Season"] = (
                    df[GROUP_KEY]
                    .astype(str)
                    .str.extract(r"(20[2-9]\d)", expand=False)
                    .astype(float)
                    .fillna(0)
                    .astype(int)
                )
                seasons = sorted(
                    df.loc[df["Season"] > 0, "Season"].unique()
                )
                if len(seasons) > 0:
                    print(f"   Season extracted from RaceID: "
                          f"{seasons}")
                else:
                    raise ValueError("No year found in RaceID")
            except Exception:
                df["Season"] = 2024  # fallback: single season
                print(f"      Could not determine season. "
                      f"Falling back to single season.")

    return df


def compute_baselines(
    df: pd.DataFrame,
    mode: str = BASELINE_MODE,
) -> tuple[pd.DataFrame, dict[str, float], float, str]:
    """
    Compute per-track baseline lap times and normalise the
    target to a delta.

    Two modes:
      "track"        — per-TrackName median (22 groups)
      "track_season" — per-TrackName-per-Season median

    Per-track-per-season baselines are preferred for multi-
    season datasets because regulation changes can shift
    baseline lap times by several seconds at the same circuit.

    Returns
    -------
    df : pd.DataFrame
        With DELTA_TARGET column added.
    track_baselines : dict
        {key: median_lap_time} mapping.
    global_baseline : float
        Global median (fallback for unseen tracks).
    actual_mode : str
        The mode actually used (may fall back to "track"
        if Season data is unavailable).
    """
    print(f"\n  Computing per-track baselines "
          f"(mode: {mode})…")

    global_baseline = float(df[RAW_TARGET].median())
    actual_mode = mode

    if mode == "track_season" and "Season" in df.columns:
        n_seasons = df["Season"].nunique()
        if n_seasons > 1:
            # Build composite key
            df = df.copy()
            df["_BaselineKey"] = (
                df["TrackName"] + "_" +
                df["Season"].astype(str)
            )
            track_baselines_raw = (
                df.groupby("_BaselineKey")[RAW_TARGET]
                .median()
                .to_dict()
            )

            # Also need a per-track-only fallback for
            # prediction on new seasons
            track_only_baselines = (
                df.groupby("TrackName")[RAW_TARGET]
                .median()
                .to_dict()
            )

            df["_TrackBaseline"] = df["_BaselineKey"].map(
                track_baselines_raw
            )
            df[DELTA_TARGET] = (
                df[RAW_TARGET] - df["_TrackBaseline"]
            )

            track_baselines = track_baselines_raw

            print(f"   Baseline groups: "
                  f"{len(track_baselines)} "
                  f"(track × season)")
            print(f"   Seasons: {n_seasons} — "
                  f"{sorted(df['Season'].unique())}")
        else:
            print(f"      Only 1 season found. "
                  f"Falling back to 'track' mode.")
            actual_mode = "track"
    else:
        if mode == "track_season":
            print(f"      Season column not available. "
                  f"Falling back to 'track' mode.")
        actual_mode = "track"

    if actual_mode == "track":
        df = df.copy()
        track_baselines = (
            df.groupby("TrackName")[RAW_TARGET]
            .median()
            .to_dict()
        )
        track_only_baselines = track_baselines

        df["_TrackBaseline"] = df["TrackName"].map(
            track_baselines
        )
        df[DELTA_TARGET] = (
            df[RAW_TARGET] - df["_TrackBaseline"]
        )

        print(f"   Baseline groups: "
              f"{len(track_baselines)} (track only)")

    print(f"   Global median: {global_baseline:.2f}s")
    print(f"\n   Delta target ({DELTA_TARGET}):")
    print(f"     Mean:  {df[DELTA_TARGET].mean():.2f}s  "
          f"Std: {df[DELTA_TARGET].std():.2f}s")
    print(f"     Range: {df[DELTA_TARGET].min():.2f}s – "
          f"{df[DELTA_TARGET].max():.2f}s")

    # Show baselines
    print(f"\n   Baselines:")
    for key in sorted(track_baselines):
        if actual_mode == "track_season":
            mask = df["_BaselineKey"] == key
        else:
            mask = df["TrackName"] == key
        n_laps = mask.sum()
        print(
            f"     {key:35s} "
            f"{track_baselines[key]:7.2f}s "
            f"({n_laps:,} laps)"
        )

    # Store track-only baselines for prediction fallback
    if actual_mode == "track_season":
        df.attrs["track_only_baselines"] = track_only_baselines

    return df, track_baselines, global_baseline, actual_mode


def filter_outlier_deltas(
    df: pd.DataFrame,
    config: Optional[dict] = None,
) -> pd.DataFrame:
    """
    Remove laps with extreme deltas that the model cannot
    learn from (safety cars, VSC, pit anomalies, red flags).

    The v1 model had deltas from -6.4s to +46.2s. The +46s
    tails inflate RMSE and distort tree splits, wasting model
    capacity on unpredictable events.

    Uses both quantile-based and hard-cap filtering (whichever
    is tighter), applied symmetrically.
    """
    if config is None:
        config = OUTLIER_CONFIG

    n_before = len(df)

    q_low = df[DELTA_TARGET].quantile(
        config["lower_quantile"]
    )
    q_high = df[DELTA_TARGET].quantile(
        config["upper_quantile"]
    )
    hard_cap = config["hard_cap_seconds"]

    # Apply the tighter of quantile vs hard cap
    lower_bound = max(q_low, -hard_cap)
    upper_bound = min(q_high, hard_cap)

    mask = (
        (df[DELTA_TARGET] >= lower_bound)
        & (df[DELTA_TARGET] <= upper_bound)
    )
    df_clean = df[mask].copy()
    n_after = len(df_clean)
    n_removed = n_before - n_after

    print(f"\n🧹 Outlier filtering:")
    print(f"   Quantile bounds: "
          f"[{q_low:+.2f}s, {q_high:+.2f}s]")
    print(f"   Hard cap:        ±{hard_cap:.1f}s")
    print(f"   Effective:       "
          f"[{lower_bound:+.2f}s, {upper_bound:+.2f}s]")
    print(f"   Removed: {n_removed:,} laps "
          f"({n_removed / n_before:.1%})")
    print(f"   Remaining: {n_after:,} laps")
    print(f"   New delta std: "
          f"{df_clean[DELTA_TARGET].std():.2f}s")
    print(f"   New delta range: "
          f"[{df_clean[DELTA_TARGET].min():+.2f}s, "
          f"{df_clean[DELTA_TARGET].max():+.2f}s]")

    # Show per-quantile removal
    n_low = ((df[DELTA_TARGET] < lower_bound)).sum()
    n_high = ((df[DELTA_TARGET] > upper_bound)).sum()
    print(f"   Removed below: {n_low:,} laps "
          f"(< {lower_bound:+.2f}s)")
    print(f"   Removed above: {n_high:,} laps "
          f"(> {upper_bound:+.2f}s)")

    return df_clean


def prepare_features(df: pd.DataFrame):
    """Extract X, y (delta), groups, and cat indices."""
    X = df[ALL_FEATURES].copy()
    y = df[DELTA_TARGET].copy()
    groups = df[GROUP_KEY].copy()
    cat_indices = [
        ALL_FEATURES.index(c) for c in CATEGORICAL_FEATURES
    ]

    print(f"\n  Feature matrix: {X.shape}")
    print(f"   Target: {DELTA_TARGET} (not raw {RAW_TARGET})")
    print(f"   Categorical ({len(cat_indices)}): "
          f"{CATEGORICAL_FEATURES}")
    print(f"   Numerical ({len(NUMERICAL_FEATURES)}): "
          f"{NUMERICAL_FEATURES}")
    print(f"   Device: {TASK_TYPE}")

    return X, y, groups, cat_indices


# ═══════════════════════════════════════════════════════════════
# 4. CROSS-VALIDATION
# ═══════════════════════════════════════════════════════════════

def cross_validate(
    X: pd.DataFrame,
    y: pd.Series,
    groups: pd.Series,
    cat_indices: list,
    df_full: pd.DataFrame,
    track_baselines: dict[str, float],
    baseline_mode: str = "track",
    params: Optional[dict] = None,
    n_splits: Optional[int] = None,
) -> list:
    """
    GroupKFold cross-validation.

    Reports both delta RMSE (what the model optimises) and
    absolute RMSE (for comparison with the raw-target version).

    v2: Added min_iterations sanity check logging.
    """
    if params is None:
        params = DEFAULT_PARAMS.copy()
    if n_splits is None:
        n_splits = CV_CONFIG["n_splits"]

    cv_params = _make_cv_params(params, use_gpu=True)
    es_rounds = CV_CONFIG["early_stopping_rounds"]
    gkf = GroupKFold(n_splits=n_splits)
    fold_metrics: list[dict] = []

    device_label = (
        "GPU  "
        if cv_params.get("task_type") == "GPU"
        else "CPU"
    )

    print(f"\n{'═' * 60}")
    print(f"  CROSS-VALIDATION ({n_splits}-Fold GroupKFold)")
    print(f"  Device: {device_label}")
    print(f"  Target: {DELTA_TARGET}")
    print(f"  Baseline mode: {baseline_mode}")
    print(f"{'═' * 60}")

    cv_start = time.perf_counter()

    for fold_i, (train_idx, test_idx) in enumerate(
        gkf.split(X, y, groups=groups)
    ):
        fold_start = time.perf_counter()
        print(f"\n── Fold {fold_i + 1}/{n_splits} ──")

        X_train = X.iloc[train_idx]
        X_test = X.iloc[test_idx]
        y_train = y.iloc[train_idx]
        y_test = y.iloc[test_idx]

        test_races = groups.iloc[test_idx].unique()
        print(
            f"   Train: {len(train_idx):,} | "
            f"Test: {len(test_idx):,} "
            f"({len(test_races)} races)"
        )

        train_pool = Pool(
            X_train, y_train, cat_features=cat_indices
        )
        test_pool = Pool(
            X_test, y_test, cat_features=cat_indices
        )

        model = CatBoostRegressor(**cv_params)
        model.fit(
            train_pool,
            eval_set=test_pool,
            early_stopping_rounds=es_rounds,
            verbose=0,
        )

        best_iter = model.get_best_iteration()
        delta_preds = model.predict(test_pool)

        # ── Delta metrics (what the model optimises) ──────────
        delta_rmse = np.sqrt(
            mean_squared_error(y_test, delta_preds)
        )
        delta_mae = mean_absolute_error(y_test, delta_preds)
        delta_r2 = r2_score(y_test, delta_preds)

        # ── Absolute metrics (for comparison) ─────────────────
        if baseline_mode == "track_season":
            # Reconstruct the baseline key for test rows
            if "Season" in df_full.columns:
                test_keys = (
                    df_full.iloc[test_idx]["TrackName"]
                    + "_"
                    + df_full.iloc[test_idx]["Season"]
                    .astype(str)
                )
                test_baselines = test_keys.map(
                    track_baselines
                ).values
                # Fallback for missing keys
                track_only = df_full.attrs.get(
                    "track_only_baselines", {}
                )
                nan_mask = np.isnan(
                    test_baselines.astype(float)
                )
                if nan_mask.any():
                    fallback = (
                        df_full.iloc[test_idx]
                        .loc[nan_mask, "TrackName"]
                        .map(track_only)
                        .values
                    )
                    test_baselines[nan_mask] = fallback
            else:
                test_baselines = (
                    X_test["TrackName"]
                    .map(track_baselines)
                    .values
                )
        else:
            test_baselines = (
                X_test["TrackName"]
                .map(track_baselines)
                .values
            )

        abs_preds = delta_preds + test_baselines
        abs_actual = df_full.iloc[test_idx][RAW_TARGET].values
        abs_rmse = np.sqrt(
            mean_squared_error(abs_actual, abs_preds)
        )

        fold_elapsed = time.perf_counter() - fold_start

        print(
            f"   Delta RMSE: {delta_rmse:.4f}s | "
            f"MAE: {delta_mae:.4f}s | "
            f"R²: {delta_r2:.4f}"
        )
        print(
            f"   Abs   RMSE: {abs_rmse:.4f}s "
            f"(for comparison with raw-target model)"
        )
        print(f"     {fold_elapsed:.1f}s | "
              f"best_iter: {best_iter}")

        # ── Sanity check: best_iteration ──────────────────────
        if best_iter < 50:
            print(
                f"      best_iter={best_iter} is very low — "
                f"model may be underfitting or overfitting "
                f"immediately"
            )

        # Per-compound breakdown (on deltas)
        test_df = X_test.copy()
        test_df["_actual"] = y_test.values
        test_df["_pred"] = delta_preds

        for compound in sorted(
            test_df["Compound"].unique()
        ):
            mask = test_df["Compound"] == compound
            comp_rmse = np.sqrt(
                mean_squared_error(
                    test_df.loc[mask, "_actual"],
                    test_df.loc[mask, "_pred"],
                )
            )
            print(
                f"     {compound:8s}: "
                f"RMSE={comp_rmse:.4f}s "
                f"({mask.sum()} laps)"
            )

        fold_metrics.append({
            "fold": fold_i,
            "delta_rmse": delta_rmse,
            "delta_mae": delta_mae,
            "delta_r2": delta_r2,
            "abs_rmse": abs_rmse,
            "test_races": list(test_races),
            "n_test_laps": len(test_idx),
            "best_iteration": best_iter,
            "elapsed_sec": fold_elapsed,
        })

    # ── Summary ───────────────────────────────────────────────
    cv_elapsed = time.perf_counter() - cv_start

    avg = {
        k: np.mean([m[k] for m in fold_metrics])
        for k in [
            "delta_rmse", "delta_mae", "delta_r2", "abs_rmse",
        ]
    }
    std_rmse = np.std(
        [m["delta_rmse"] for m in fold_metrics]
    )
    avg_best_iter = np.mean(
        [m["best_iteration"] for m in fold_metrics]
    )

    print(f"\n{'═' * 60}")
    print(f"  CV RESULTS ({device_label})")
    print(f"{'═' * 60}")
    print(f"  Delta RMSE: {avg['delta_rmse']:.4f}s "
          f"± {std_rmse:.4f}s")
    print(f"  Delta MAE:  {avg['delta_mae']:.4f}s")
    print(f"  Delta R²:   {avg['delta_r2']:.4f}")
    print(f"  Abs   RMSE: {avg['abs_rmse']:.4f}s "
          f"(cf. raw-target model)")
    print(f"  Avg best_iter: {avg_best_iter:.0f}")
    print(f"  CV time:    {cv_elapsed:.1f}s")

    # ── Health checks ─────────────────────────────────────────
    if avg["delta_r2"] < 0:
        print(f"     Negative R² — model worse than "
              f"predicting the mean")
    if avg_best_iter < 50:
        print(f"     Average best_iter < 50 — "
              f"investigate regularisation")

    print(f"{'═' * 60}")

    return fold_metrics


# ═══════════════════════════════════════════════════════════════
# 5. HYPERPARAMETER TUNING
# ═══════════════════════════════════════════════════════════════

def tune_hyperparameters(
    X: pd.DataFrame,
    y: pd.Series,
    groups: pd.Series,
    cat_indices: list,
    n_trials: Optional[int] = None,
    n_splits: Optional[int] = None,
    subsample_frac: Optional[float] = None,
) -> dict:
    """
    Optuna hyperparameter search (pruned + warm-started).

    v2 changes:
      - Search space forces conservative configurations:
        * learning_rate: [0.01, 0.06] (was [0.02, 0.12])
        * depth: [4, 7] (was [5, 9])
        * l2_leaf_reg: [3, 30] (was [1, 20])
        * min_data_in_leaf: [20, 100] (new)
        * iterations: [800, 2000] (was [500, 1500])
      - subsample_frac defaults to 1.0 (was 0.6)
      - Warm-start uses conservative defaults
    """
    if not OPTUNA_AVAILABLE:
        print("   Optuna not available. Using defaults.")
        return DEFAULT_PARAMS.copy()

    if n_trials is None:
        n_trials = TUNING_CONFIG["n_trials"]
    if n_splits is None:
        n_splits = TUNING_CONFIG["n_splits"]
    if subsample_frac is None:
        subsample_frac = TUNING_CONFIG["subsample_frac"]

    max_iters = TUNING_CONFIG["max_iterations"]
    es_rounds = TUNING_CONFIG["early_stopping_rounds"]
    device_label = "GPU  " if GPU_AVAILABLE else "CPU"
    max_fits = n_trials * n_splits

    print(f"\n{'═' * 60}")
    print(f"  HYPERPARAMETER TUNING ({n_trials} trials)")
    print(f"{'═' * 60}")
    print(f"  Device:          {device_label}")
    print(f"  Max fits:        {max_fits}")
    print(f"  Train subsample: {subsample_frac:.0%}")
    print(f"  Iter cap:        {max_iters}")
    print(f"  Early stop:      {es_rounds}")
    print(f"  Target:          {DELTA_TARGET}")
    print(f"{'═' * 60}")

    gkf = GroupKFold(n_splits=n_splits)
    fold_indices = list(gkf.split(X, y, groups=groups))
    tuning_start = time.perf_counter()

    def objective(trial: "optuna.Trial") -> float:
        params: dict[str, Any] = {
            # v2: Conservative search space
            "iterations": trial.suggest_int(
                "iterations", 800, max_iters,
            ),
            "learning_rate": trial.suggest_float(
                "learning_rate", 0.01, 0.06, log=True,
            ),
            "depth": trial.suggest_int("depth", 4, 7),
            "l2_leaf_reg": trial.suggest_float(
                "l2_leaf_reg", 3.0, 30.0, log=True,
            ),
            "min_data_in_leaf": trial.suggest_int(
                "min_data_in_leaf", 20, 100,
            ),
            "bagging_temperature": trial.suggest_float(
                "bagging_temperature", 0.5, 2.0,
            ),
            "random_strength": trial.suggest_float(
                "random_strength", 0.5, 2.0,
            ),
            "border_count": 128,
            "loss_function": "RMSE",
            "verbose": 0,
            "random_seed": 42,
        }

        if GPU_AVAILABLE:
            params["task_type"] = "GPU"
            params.update(GPU_OVERRIDES)
        else:
            params["task_type"] = "CPU"

        fold_rmses: list[float] = []
        rng = np.random.RandomState(42 + trial.number)

        for fold_i, (train_idx, test_idx) in enumerate(
            fold_indices
        ):
            if subsample_frac < 1.0:
                n_sub = max(
                    200,
                    int(len(train_idx) * subsample_frac),
                )
                train_idx_sub = rng.choice(
                    train_idx, size=n_sub, replace=False,
                )
            else:
                train_idx_sub = train_idx

            train_pool = Pool(
                X.iloc[train_idx_sub],
                y.iloc[train_idx_sub],
                cat_features=cat_indices,
            )
            test_pool = Pool(
                X.iloc[test_idx],
                y.iloc[test_idx],
                cat_features=cat_indices,
            )

            model = CatBoostRegressor(**params)
            model.fit(
                train_pool,
                eval_set=test_pool,
                early_stopping_rounds=es_rounds,
                verbose=0,
            )

            preds = model.predict(test_pool)
            rmse = np.sqrt(
                mean_squared_error(y.iloc[test_idx], preds)
            )
            fold_rmses.append(rmse)

            trial.report(
                float(np.mean(fold_rmses)), step=fold_i
            )
            if trial.should_prune():
                raise optuna.TrialPruned()

        return float(np.mean(fold_rmses))

    optuna.logging.set_verbosity(optuna.logging.WARNING)

    study = optuna.create_study(
        direction="minimize",
        pruner=optuna.pruners.MedianPruner(
            n_startup_trials=5,
            n_warmup_steps=0,
        ),
    )

    # Warm-start with conservative defaults
    study.enqueue_trial({
        "iterations": min(
            DEFAULT_PARAMS["iterations"], max_iters
        ),
        "learning_rate": DEFAULT_PARAMS["learning_rate"],
        "depth": DEFAULT_PARAMS["depth"],
        "l2_leaf_reg": DEFAULT_PARAMS["l2_leaf_reg"],
        "min_data_in_leaf": DEFAULT_PARAMS["min_data_in_leaf"],
        "bagging_temperature": 1.0,
        "random_strength": 1.0,
    })

    study.optimize(
        cast(Any, objective),
        n_trials=n_trials,
        show_progress_bar=True,
    )

    tuning_elapsed = time.perf_counter() - tuning_start

    n_complete = len([
        t for t in study.trials
        if t.state == optuna.trial.TrialState.COMPLETE
    ])
    n_pruned = len([
        t for t in study.trials
        if t.state == optuna.trial.TrialState.PRUNED
    ])
    actual_fits = sum(
        len(t.intermediate_values) for t in study.trials
    )

    best = study.best_params
    print(f"\n    Best RMSE:   {study.best_value:.4f}s")
    print(f"     Time:        {tuning_elapsed:.1f}s "
          f"({tuning_elapsed / 60:.1f} min)")
    print(f"    Trials:       {n_complete} complete, "
          f"{n_pruned} pruned")
    print(f"    Fold fits:    {actual_fits} "
          f"(vs {max_fits} max)")
    print(f"  Best parameters:")
    for k, v in sorted(best.items()):
        print(f"    {k}: {v}")

    # ── Sanity check on tuned iterations ──────────────────────
    if "iterations" in best and best["iterations"] < 200:
        print(
            f"     Tuned iterations={best['iterations']} "
            f"is very low. Clamping to 500."
        )
        best["iterations"] = 500

    best["loss_function"] = "RMSE"
    best["random_seed"] = 42
    best["verbose"] = 200
    best.setdefault("bagging_temperature", 1.0)
    best.setdefault("random_strength", 1.0)
    best.setdefault("border_count", 128)

    return best


# ═══════════════════════════════════════════════════════════════
# 6. FINAL MODEL TRAINING
# ═══════════════════════════════════════════════════════════════

def train_final_model(
    X: pd.DataFrame,
    y: pd.Series,
    cat_indices: list,
    params: Optional[dict] = None,
    cv_best_iterations: Optional[list[int]] = None,
) -> CatBoostRegressor:
    """
    Train the final model on ALL data.

    Uses adaptive iteration count from CV best_iterations.
    Always CPU (posterior_sampling + langevin require it).

    v2: Minimum iteration floor of 500 to prevent the
    5-iteration collapse seen in v1.
    """
    if params is None:
        params = DEFAULT_PARAMS.copy()

    final_params = {**params}
    final_params["posterior_sampling"] = True
    final_params["langevin"] = True
    final_params["task_type"] = "CPU"
    final_params.pop("devices", None)

    # ── Adaptive iteration count ──────────────────────────────
    # v2: Floor of 500 iterations minimum.  The v1 model
    # collapsed to 5-11 iterations because of aggressive
    # hyperparameters.  With conservative params this should
    # not happen, but the floor is a safety net.
    MIN_FINAL_ITERATIONS = 500

    if cv_best_iterations and all(
        b is not None and b > 0 for b in cv_best_iterations
    ):
        adaptive_iters = int(
            np.median(cv_best_iterations) * 1.15
        )
        adaptive_iters = max(
            MIN_FINAL_ITERATIONS,
            min(adaptive_iters, 4000),
        )
        original_iters = final_params.get("iterations", 2500)

        if adaptive_iters < original_iters:
            print(
                f"\n    Adaptive iterations: "
                f"{original_iters} → {adaptive_iters} "
                f"(CV best: {cv_best_iterations}, "
                f"floor: {MIN_FINAL_ITERATIONS})"
            )
            final_params["iterations"] = adaptive_iters
        else:
            print(
                f"\n     Keeping {original_iters} iterations "
                f"(CV best: {cv_best_iterations}, "
                f"×1.15 = {adaptive_iters})"
            )
    else:
        print(
            f"\n     No valid CV best_iterations. "
            f"Using {final_params.get('iterations', 2500)}."
        )

    print(f"\n{'═' * 60}")
    print(f"  TRAINING FINAL MODEL")
    print(f"{'═' * 60}")
    print(f"  Training on {len(X):,} laps (all data)")
    print(f"  Target: {DELTA_TARGET}")
    print(f"  Device: CPU (required for virtual ensembles)")
    if GPU_AVAILABLE:
        print(f"     GPU was used for CV & tuning")
    print(f"  Parameters:")
    for k, v in sorted(final_params.items()):
        if k != "verbose":
            print(f"    {k}: {v}")

    pool = Pool(X, y, cat_features=cat_indices)

    train_start = time.perf_counter()
    model = CatBoostRegressor(**final_params)
    model.fit(pool)
    train_elapsed = time.perf_counter() - train_start

    # ── Training loss sanity check ────────────────────────────
    train_preds = model.predict(pool)
    train_rmse = np.sqrt(mean_squared_error(y, train_preds))
    train_r2 = r2_score(y, train_preds)
    print(f"\n  Train RMSE: {train_rmse:.4f}s | "
          f"R²: {train_r2:.4f}")

    model.save_model(str(MODEL_PATH))
    print(f"    Model saved to: {MODEL_PATH}")
    print(f"     Training time: {train_elapsed:.1f}s "
          f"({train_elapsed / 60:.1f} min)")

    return model


# ═══════════════════════════════════════════════════════════════
# 7. FEATURE ANALYSIS
# ═══════════════════════════════════════════════════════════════

def analyse_features(
    model: CatBoostRegressor,
    X: pd.DataFrame,
    y: pd.Series,
    cat_indices: list,
    compute_interactions: bool = False,
) -> pd.DataFrame:
    """
    Analyse feature importance and (optionally) interactions.

    With the delta target + outlier filtering + conservative
    hyperparameters, we expect:
      - TireAge/TireAgeSq, Compound, RaceLapNumber in top 5
      - Weather/traffic features contributing non-zero amounts
      - No single feature > 35%
      - TrackName reduced from 24% (v1) if using track_season
        baselines

    v2: Added zero-importance feature warning.
    """
    pool = Pool(X, y, cat_features=cat_indices)

    importance_raw = model.get_feature_importance(
        pool, type=cast(Any, "PredictionValuesChange")
    )
    importance = np.asarray(
        importance_raw, dtype=float
    ).reshape(-1)
    importance_sum = (
        float(importance.sum()) if importance.size else 1.0
    )

    feat_imp = pd.DataFrame({
        "Feature": ALL_FEATURES,
        "Importance": importance,
        "Importance_Pct": importance / importance_sum * 100,
    }).sort_values("Importance", ascending=False)

    print(f"\n{'═' * 60}")
    print(f"  FEATURE IMPORTANCE (delta target)")
    print(f"{'═' * 60}")
    for _, row in feat_imp.iterrows():
        bar_len = int(row["Importance_Pct"] / 2)
        bar = "█" * bar_len
        print(
            f"  {row['Feature']:20s} "
            f"{row['Importance_Pct']:5.1f}% {bar}"
        )

    # ── Sanity checks ─────────────────────────────────────────
    print(f"\n  Physical validation:")

    top_5 = set(feat_imp.head(5)["Feature"])
    expected_top = {"TireAge", "TireAgeSq", "Compound",
                    "RaceLapNumber", "RaceLapFraction"}
    found = top_5 & expected_top

    if len(found) >= 2:
        print(
            f"      Race dynamics in top 5: {found}"
        )
    else:
        print(
            f"       Expected race dynamics "
            f"(TireAge/Compound/RaceLapNumber) not "
            f"dominant. Top 5: {top_5}"
        )

    # Check max importance < 35%
    max_pct = feat_imp["Importance_Pct"].iloc[0]
    max_feat = feat_imp["Feature"].iloc[0]
    if max_pct > 35:
        print(
            f"       {max_feat} dominates at "
            f"{max_pct:.1f}% — investigate possible "
            f"shortcut"
        )
    else:
        print(
            f"      Importance well-distributed "
            f"(max: {max_feat} at {max_pct:.1f}%)"
        )

    # Check for zero-importance features (v2)
    zero_features = feat_imp[
        feat_imp["Importance_Pct"] < 0.1
    ]["Feature"].tolist()
    if zero_features:
        print(
            f"       Near-zero importance ({len(zero_features)} "
            f"features): {zero_features}"
        )
        print(
            f"       These features may need more data "
            f"variation or different encoding"
        )
    else:
        print(
            f"      All features contributing "
            f"(no zero-importance)"
        )

    # Check TrackName importance (v2)
    track_pct = feat_imp.loc[
        feat_imp["Feature"] == "TrackName",
        "Importance_Pct",
    ].iloc[0]
    if track_pct > 20:
        print(
            f"       TrackName at {track_pct:.1f}% — "
            f"baseline may not be capturing all "
            f"track-level variance"
        )
    else:
        print(
            f"      TrackName at {track_pct:.1f}% "
            f"(baseline working well)"
        )

    # ── Optional: Interactions ────────────────────────────────
    if compute_interactions:
        print(f"\n{'═' * 60}")
        print(f"  TOP FEATURE INTERACTIONS")
        print(f"{'═' * 60}")

        try:
            interactions_raw = model.get_feature_importance(
                pool, type=cast(Any, "Interaction")
            )
            interactions = (
                np.asarray(interactions_raw, dtype=float)
                if interactions_raw is not None
                else np.empty((0, 3), dtype=float)
            )

            if interactions.size > 0:
                if interactions.ndim == 1:
                    interactions = interactions.reshape(-1, 3)
                inter_df = pd.DataFrame(
                    interactions,
                    columns=[
                        "Feature1_idx",
                        "Feature2_idx",
                        "Strength",
                    ],
                )
                inter_df["Feature1"] = (
                    inter_df["Feature1_idx"]
                    .astype(int)
                    .map(lambda i: ALL_FEATURES[i])
                )
                inter_df["Feature2"] = (
                    inter_df["Feature2_idx"]
                    .astype(int)
                    .map(lambda i: ALL_FEATURES[i])
                )
                inter_df = inter_df.sort_values(
                    "Strength", ascending=False
                )

                for _, row in inter_df.head(10).iterrows():
                    print(
                        f"  {row['Feature1']:20s} × "
                        f"{row['Feature2']:20s} "
                        f"Strength: {row['Strength']:.2f}"
                    )

                # Physical validation
                print(f"\n  Interaction validation:")

                compound_tire = inter_df[
                    (
                        (inter_df["Feature1"] == "Compound")
                        & (inter_df["Feature2"].isin(
                            ["TireAge", "TireAgeSq"]
                        ))
                    )
                    | (
                        (inter_df["Feature2"] == "Compound")
                        & (inter_df["Feature1"].isin(
                            ["TireAge", "TireAgeSq"]
                        ))
                    )
                ]
                print(
                    "      Compound × TireAge detected"
                    if not compound_tire.empty
                    else "     Compound × TireAge "
                    "NOT found"
                )

                temp_compound = inter_df[
                    (
                        (inter_df["Feature1"] == "TrackTemp")
                        & (inter_df["Feature2"] == "Compound")
                    )
                    | (
                        (inter_df["Feature2"] == "TrackTemp")
                        & (inter_df["Feature1"] == "Compound")
                    )
                ]
                print(
                    "     TrackTemp × Compound detected"
                    if not temp_compound.empty
                    else "      TrackTemp × Compound "
                    "NOT found"
                )
        except Exception as e:
            print(
                f"    Could not compute interactions: {e}"
            )
    else:
        print(
            f"\n   Interaction analysis skipped "
            f"(compute_interactions=True to enable)"
        )

    return feat_imp


# ═══════════════════════════════════════════════════════════════
# 8. TRACKNAME ABLATION TEST
# ═══════════════════════════════════════════════════════════════

def run_trackname_ablation(
    X: pd.DataFrame,
    y: pd.Series,
    groups: pd.Series,
    cat_indices: list,
    params: dict,
) -> dict:
    """
    Quick 2-fold ablation: train with and without TrackName
    to check if it is helping or hurting generalisation.

    If removing TrackName improves R², the model was using
    it as a memorisation shortcut even after delta normalisation.

    Returns
    -------
    dict with keys "with_track" and "without_track", each
    containing {"rmse": float, "r2": float}.
    """
    print(f"\n{'═' * 60}")
    print(f"  TRACKNAME ABLATION TEST (2-fold quick check)")
    print(f"{'═' * 60}")

    gkf = GroupKFold(n_splits=2)
    cv_params = _make_cv_params(params, use_gpu=True)
    es_rounds = CV_CONFIG["early_stopping_rounds"]

    results: dict[str, dict] = {}

    for label, drop_track in [
        ("with_track", False),
        ("without_track", True),
    ]:
        if drop_track:
            features_subset = [
                f for f in ALL_FEATURES if f != "TrackName"
            ]
            cat_subset = [
                features_subset.index(c)
                for c in CATEGORICAL_FEATURES
                if c != "TrackName"
            ]
            X_sub = X[features_subset]
        else:
            features_subset = ALL_FEATURES
            cat_subset = cat_indices
            X_sub = X

        fold_rmses = []
        fold_r2s = []

        for train_idx, test_idx in gkf.split(
            X_sub, y, groups=groups
        ):
            train_pool = Pool(
                X_sub.iloc[train_idx],
                y.iloc[train_idx],
                cat_features=cat_subset,
            )
            test_pool = Pool(
                X_sub.iloc[test_idx],
                y.iloc[test_idx],
                cat_features=cat_subset,
            )

            model = CatBoostRegressor(**cv_params)
            model.fit(
                train_pool,
                eval_set=test_pool,
                early_stopping_rounds=es_rounds,
                verbose=0,
            )

            preds = model.predict(test_pool)
            fold_rmses.append(np.sqrt(
                mean_squared_error(y.iloc[test_idx], preds)
            ))
            fold_r2s.append(
                r2_score(y.iloc[test_idx], preds)
            )

        avg_rmse = float(np.mean(fold_rmses))
        avg_r2 = float(np.mean(fold_r2s))
        results[label] = {"rmse": avg_rmse, "r2": avg_r2}

        print(f"  {label:20s}: "
              f"RMSE={avg_rmse:.4f}s  R²={avg_r2:.4f}")

    # Recommendation
    with_r2 = results["with_track"]["r2"]
    without_r2 = results["without_track"]["r2"]

    if without_r2 > with_r2 + 0.01:
        print(
            f"\n    RECOMMENDATION: Remove TrackName — "
            f"R² improves by {without_r2 - with_r2:.4f}"
        )
    elif abs(with_r2 - without_r2) < 0.01:
        print(
            f"\n   TrackName has marginal impact "
            f"(ΔR² = {with_r2 - without_r2:+.4f}). "
            f"Keeping it is fine."
        )
    else:
        print(
            f"\n  TrackName is helping "
            f"(ΔR² = {with_r2 - without_r2:+.4f})"
        )

    return results


# ═══════════════════════════════════════════════════════════════
# 9. SAVE TRAINING REPORT
# ═══════════════════════════════════════════════════════════════

def save_report(
    fold_metrics: list,
    feat_imp: pd.DataFrame,
    params: dict,
    track_baselines: dict[str, float],
    baseline_mode: str,
    n_laps: int,
    n_races: int,
    n_laps_after_filter: int = 0,
    ablation_results: Optional[dict] = None,
    total_time: float = 0.0,
):
    """Save a text report of the training results."""
    avg = {
        k: np.mean([m[k] for m in fold_metrics])
        for k in [
            "delta_rmse", "delta_mae", "delta_r2", "abs_rmse",
        ]
    }
    std_rmse = np.std(
        [m["delta_rmse"] for m in fold_metrics]
    )
    avg_best_iter = np.mean(
        [m["best_iteration"] for m in fold_metrics]
    )
    total_cv_time = sum(
        m.get("elapsed_sec", 0) for m in fold_metrics
    )

    lines = [
        "F1 VIRTUAL RACE STRATEGIST — TRAINING REPORT (v2)",
        "=" * 55,
        "",
        f"Dataset: {n_laps:,} laps from {n_races} races",
        f"After outlier filtering: {n_laps_after_filter:,} laps"
        if n_laps_after_filter > 0
        else f"After outlier filtering: N/A",
        f"Features: {len(ALL_FEATURES)} "
        f"({len(CATEGORICAL_FEATURES)} cat, "
        f"{len(NUMERICAL_FEATURES)} num)",
        f"Model: CatBoost Regressor",
        f"Target: {DELTA_TARGET} (delta from per-track median)",
        f"Baseline mode: {baseline_mode}",
        f"GPU: {'Yes' if GPU_AVAILABLE else 'No'}",
        f"Pipeline time: {total_time:.1f}s "
        f"({total_time / 60:.1f} min)",
        "",
        "CROSS-VALIDATION (GroupKFold by RaceID)",
        "-" * 55,
        f"Folds: {CV_CONFIG['n_splits']}",
        f"Delta RMSE: {avg['delta_rmse']:.4f}s "
        f"± {std_rmse:.4f}s",
        f"Delta MAE:  {avg['delta_mae']:.4f}s",
        f"Delta R²:   {avg['delta_r2']:.4f}",
        f"Abs   RMSE: {avg['abs_rmse']:.4f}s",
        f"Avg best_iter: {avg_best_iter:.0f}",
        f"CV time:    {total_cv_time:.1f}s",
        "",
    ]

    for m in fold_metrics:
        elapsed_str = (
            f", {m['elapsed_sec']:.1f}s"
            if "elapsed_sec" in m
            else ""
        )
        lines.append(
            f"  Fold {m['fold']}: "
            f"Δ RMSE={m['delta_rmse']:.4f}s  "
            f"Abs RMSE={m['abs_rmse']:.4f}s  "
            f"best_iter={m['best_iteration']}  "
            f"({m['n_test_laps']} laps, "
            f"{len(m['test_races'])} races{elapsed_str})"
        )

    lines.extend(["", "HYPERPARAMETERS", "-" * 55])
    for k, v in sorted(params.items()):
        lines.append(f"  {k}: {v}")

    lines.extend(["", "TUNING CONFIG", "-" * 55])
    for k, v in sorted(TUNING_CONFIG.items()):
        lines.append(f"  {k}: {v}")

    lines.extend(["", "OUTLIER CONFIG", "-" * 55])
    for k, v in sorted(OUTLIER_CONFIG.items()):
        lines.append(f"  {k}: {v}")

    lines.extend(["", "FEATURE IMPORTANCE", "-" * 55])
    for _, row in feat_imp.iterrows():
        lines.append(
            f"  {row['Feature']:20s} "
            f"{row['Importance_Pct']:5.1f}%"
        )

    if ablation_results:
        lines.extend(
            ["", "TRACKNAME ABLATION", "-" * 55]
        )
        for label, metrics in ablation_results.items():
            lines.append(
                f"  {label:20s}: "
                f"RMSE={metrics['rmse']:.4f}s  "
                f"R²={metrics['r2']:.4f}"
            )

    lines.extend(
        ["", "PER-TRACK BASELINES", "-" * 55]
    )
    lines.append(f"  Mode: {baseline_mode}")
    for key in sorted(track_baselines):
        lines.append(
            f"  {key:35s} {track_baselines[key]:.2f}s"
        )

    REPORT_PATH.write_text(
        "\n".join(lines), encoding="utf-8"
    )
    print(f"\n   Report saved to: {REPORT_PATH}")


def save_baselines(
    track_baselines: dict[str, float],
    global_baseline: float,
    baseline_mode: str = "track",
    track_only_baselines: Optional[dict[str, float]] = None,
):
    """
    Save track baselines to JSON for use by the prediction
    interface and the strategy engine.

    The prediction interface needs these to convert delta
    predictions back to absolute lap times.

    v2: Supports both "track" and "track_season" modes.
    When using track_season baselines, also saves track-only
    fallbacks for unseen seasons.
    """
    payload = {
        "track_baselines": track_baselines,
        "global_baseline": global_baseline,
        "baseline_mode": baseline_mode,
        "description": (
            "Per-track median lap times (seconds). "
            "Add to model delta prediction to get "
            "absolute lap time."
        ),
    }

    if track_only_baselines is not None:
        payload["track_only_baselines"] = track_only_baselines

    BASELINES_PATH.write_text(
        json.dumps(payload, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    print(f"   Baselines saved to: {BASELINES_PATH}")


# ═══════════════════════════════════════════════════════════════
# 10. PREDICTION INTERFACE
# ═══════════════════════════════════════════════════════════════

class PacePredictor:
    """
    Wrapper around the trained CatBoost model for the
    Strategy Engine.

    The model predicts DELTA from a per-track baseline.
    This class handles the conversion back to absolute
    lap times transparently.

    v2 changes:
      - Supports both "track" and "track_season" baseline modes
      - Falls back gracefully: track_season → track → global
      - RaceLapFraction computed automatically if not provided
      - Batch predictions handle baseline mode correctly

    Provides:
      - Point predictions (absolute lap time)
      - Delta predictions (for same-track comparisons)
      - Uncertainty estimates (mean ± std via virtual ensembles)
      - Batch predictions
    """

    def __init__(
        self,
        model_path: Optional[str] = None,
        baselines_path: Optional[str] = None,
    ):
        """
        Load model and track baselines from disk.

        Parameters
        ----------
        model_path : str, optional
            Path to .cbm file. Defaults to MODEL_PATH.
        baselines_path : str, optional
            Path to track_baselines.json.
            Defaults to BASELINES_PATH.
        """
        if model_path is None:
            model_path = str(MODEL_PATH)
        if baselines_path is None:
            baselines_path = str(BASELINES_PATH)

        self.model = CatBoostRegressor()
        self.model.load_model(model_path)

        with open(baselines_path, "r") as f:
            payload = json.load(f)

        self._track_baselines: dict[str, float] = (
            payload["track_baselines"]
        )
        self._global_baseline: float = (
            payload["global_baseline"]
        )
        self._baseline_mode: str = payload.get(
            "baseline_mode", "track"
        )
        self._track_only_baselines: dict[str, float] = (
            payload.get("track_only_baselines", {})
        )

        self._cat_indices = [
            ALL_FEATURES.index(c)
            for c in CATEGORICAL_FEATURES
        ]

        print(
            f"🏎️  PacePredictor loaded "
            f"({len(self._track_baselines)} baseline groups, "
            f"mode: {self._baseline_mode})"
        )

    def _get_baseline(
        self,
        track_name: str,
        season: Optional[int] = None,
    ) -> float:
        """
        Look up baseline for a track (and optionally season).

        Fallback chain:
          1. track_season key (if mode is track_season)
          2. track-only baseline
          3. global median

        Parameters
        ----------
        track_name : str
            Circuit name.
        season : int, optional
            Year/season. Used only in track_season mode.
        """
        if (
            self._baseline_mode == "track_season"
            and season is not None
        ):
            key = f"{track_name}_{season}"
            if key in self._track_baselines:
                return self._track_baselines[key]
            # Fallback to track-only
            if track_name in self._track_only_baselines:
                return self._track_only_baselines[track_name]
        elif track_name in self._track_baselines:
            return self._track_baselines[track_name]

        # Final fallback
        return self._global_baseline

    def predict(
        self,
        compound: str,
        track_name: str,
        driver: str,
        team: str,
        tire_age: int,
        tire_age_sq: Optional[int] = None,
        race_lap_number: int = 1,
        race_lap_fraction: Optional[float] = None,
        total_race_laps: Optional[int] = None,
        track_temp: float = 35.0,
        air_temp: float = 28.0,
        humidity: float = 50.0,
        wind_speed: float = 2.0,
        gap_to_car_ahead: float = 5.0,
        drs_available: int = 0,
        season: Optional[int] = None,
        damage_state: float = 0.0,
        tyre_temp_c: float = 50.0,
        sliding_proxy: float = 0.5,
        fz_n: float = 15000.0,
    ) -> float:
        """
        Predict ABSOLUTE lap time for a single set of
        conditions.

        Internally predicts delta, then adds track baseline.

        Parameters
        ----------
        race_lap_fraction : float, optional
            If not provided, computed from race_lap_number /
            total_race_laps. Defaults to 0.5 if neither is
            available.
        total_race_laps : int, optional
            Total laps in the race. Used to compute
            race_lap_fraction if not provided directly.
        season : int, optional
            Year/season for track_season baseline lookup.

        Returns
        -------
        float
            Predicted lap time in seconds.
        """
        row = self._build_row(
            compound, track_name, driver, team,
            tire_age, tire_age_sq, race_lap_number,
            race_lap_fraction, total_race_laps,
            track_temp, air_temp, humidity, wind_speed,
            gap_to_car_ahead, drs_available,
            damage_state, tyre_temp_c, sliding_proxy, fz_n,
        )
        pool = Pool(row, cat_features=self._cat_indices)
        delta = float(self.model.predict(pool)[0])
        baseline = self._get_baseline(track_name, season)
        return baseline + delta

    def predict_delta(
        self,
        compound: str,
        track_name: str,
        driver: str,
        team: str,
        tire_age: int,
        tire_age_sq: Optional[int] = None,
        race_lap_number: int = 1,
        race_lap_fraction: Optional[float] = None,
        total_race_laps: Optional[int] = None,
        track_temp: float = 35.0,
        air_temp: float = 28.0,
        humidity: float = 50.0,
        wind_speed: float = 2.0,
        gap_to_car_ahead: float = 5.0,
        drs_available: int = 0,
    ) -> float:
        """
        Predict the raw DELTA (deviation from track baseline).

        Useful for the strategy engine when comparing stint
        options on the same track (baseline cancels out).
        """
        row = self._build_row(
            compound, track_name, driver, team,
            tire_age, tire_age_sq, race_lap_number,
            race_lap_fraction, total_race_laps,
            track_temp, air_temp, humidity, wind_speed,
            gap_to_car_ahead, drs_available,
        )
        pool = Pool(row, cat_features=self._cat_indices)
        return float(self.model.predict(pool)[0])

    def predict_with_uncertainty(
        self,
        compound: str,
        track_name: str,
        driver: str,
        team: str,
        tire_age: int,
        tire_age_sq: Optional[int] = None,
        race_lap_number: int = 1,
        race_lap_fraction: Optional[float] = None,
        total_race_laps: Optional[int] = None,
        track_temp: float = 35.0,
        air_temp: float = 28.0,
        humidity: float = 50.0,
        wind_speed: float = 2.0,
        gap_to_car_ahead: float = 5.0,
        drs_available: int = 0,
        n_ensembles: int = 10,
        season: Optional[int] = None,
    ) -> tuple[float, float]:
        """
        Predict ABSOLUTE lap time with uncertainty.

        Returns (mean_prediction, std_uncertainty) in seconds.
        Uncertainty is on the delta; baseline is deterministic.
        """
        row = self._build_row(
            compound, track_name, driver, team,
            tire_age, tire_age_sq, race_lap_number,
            race_lap_fraction, total_race_laps,
            track_temp, air_temp, humidity, wind_speed,
            gap_to_car_ahead, drs_available,
        )
        pool = Pool(row, cat_features=self._cat_indices)

        result = self.model.virtual_ensembles_predict(
            pool,
            prediction_type="TotalUncertainty",
            virtual_ensembles_count=n_ensembles,
        )

        delta_mean = result[0][0]
        variance = max(result[0][1], 0)
        std_dev = np.sqrt(variance)
        baseline = self._get_baseline(track_name, season)

        return float(baseline + delta_mean), float(std_dev)

    def predict_batch(
        self,
        df: pd.DataFrame,
        season: Optional[int] = None,
    ) -> np.ndarray:
        """
        Predict ABSOLUTE lap times for a DataFrame.

        The DataFrame must contain all ALL_FEATURES columns
        (including TrackName for baseline lookup).

        v2: Handles RaceLapFraction computation and both
        baseline modes.

        Parameters
        ----------
        df : pd.DataFrame
            Must contain ALL_FEATURES columns.
        season : int, optional
            Season for track_season baseline lookup.
            If not provided and baseline_mode is track_season,
            attempts to use a "Season" column in the DataFrame.

        Returns
        -------
        np.ndarray
            Absolute predicted lap times.
        """
        df = df.copy()
        for col in CATEGORICAL_FEATURES:
            df[col] = df[col].astype(str)

        # Compute RaceLapFraction if missing
        if "RaceLapFraction" not in df.columns:
            if "TotalRaceLaps" in df.columns:
                df["RaceLapFraction"] = (
                    df["RaceLapNumber"]
                    / df["TotalRaceLaps"]
                ).clip(0.0, 1.0)
            elif GROUP_KEY in df.columns:
                max_laps = df.groupby(GROUP_KEY)[
                    "RaceLapNumber"
                ].transform("max")
                df["RaceLapFraction"] = (
                    df["RaceLapNumber"] / max_laps
                ).clip(0.0, 1.0)
            else:
                df["RaceLapFraction"] = 0.5

        pool = Pool(
            df[ALL_FEATURES],
            cat_features=self._cat_indices,
        )
        deltas = self.model.predict(pool)

        # Resolve baselines per row
        if (
            self._baseline_mode == "track_season"
            and (
                season is not None
                or "Season" in df.columns
            )
        ):
            if season is not None:
                baselines = df["TrackName"].map(
                    lambda t: self._get_baseline(t, season)
                ).values
            else:
                baselines = df.apply(
                    lambda row: self._get_baseline(
                        row["TrackName"],
                        int(row["Season"]),
                    ),
                    axis=1,
                ).values
        else:
            baselines = df["TrackName"].map(
                lambda t: self._get_baseline(t)
            ).values

        return baselines + deltas

    def _build_row(
        self,
        compound,
        track_name,
        driver,
        team,
        tire_age,
        tire_age_sq,
        race_lap_number,
        race_lap_fraction,
        total_race_laps,
        track_temp,
        air_temp,
        humidity,
        wind_speed,
        gap_to_car_ahead,
        drs_available,
        damage_state=0.0,
        tyre_temp_c=50.0,
        sliding_proxy=0.5,
        fz_n=15000.0,
    ) -> pd.DataFrame:
        """Build a single-row DataFrame matching ALL_FEATURES."""
        if tire_age_sq is None:
            tire_age_sq = tire_age ** 2

        if race_lap_fraction is None:
            if total_race_laps is not None and total_race_laps > 0:
                race_lap_fraction = min(1.0, race_lap_number / total_race_laps)
            else:
                race_lap_fraction = 0.5

        return pd.DataFrame([{
            "Compound":        str(compound).upper(),
            "TrackName":       str(track_name),
            "Driver":          str(driver),
            "Team":            str(team),
            "TireAge":         int(tire_age),
            "TireAgeSq":       int(tire_age_sq),
            "RaceLapNumber":   int(race_lap_number),
            "RaceLapFraction": float(race_lap_fraction),
            "TrackTemp":       float(track_temp),
            "AirTemp":         float(air_temp),
            "Humidity":        float(humidity),
            "WindSpeed":       float(wind_speed),
            "GapToCarAhead":   float(gap_to_car_ahead),
            "DRS_Available":   int(drs_available),
            "DamageState":     float(damage_state),
            "TyreTemp_C":      float(tyre_temp_c),
            "SlidingProxy":    float(sliding_proxy),
            "Fz_N":            float(fz_n),
        }])[ALL_FEATURES]


# ═══════════════════════════════════════════════════════════════
# 11. CLI ENTRY POINT
# ═══════════════════════════════════════════════════════════════

if __name__ == "__main__":
    print("  F1 Virtual Race Strategist — Physics-Enhanced Pace Model")
    print(f"   Input:  {DATA_PATH}")
    print(f"   Output: {MODEL_PATH}")
    print(f"   Target: {DELTA_TARGET} (delta from track median)")
    print(f"   Baseline mode: {BASELINE_MODE}")
    print(f"   Device: {TASK_TYPE}")
    print(f"   Physics features: DamageState, TyreTemp_C, SlidingProxy, Fz_N")
    print()

    pipeline_start = time.perf_counter()

    # ── Step 1: Load Data ─────────────────────────────────────
    df = load_data(DATA_PATH)
    n_laps_raw = len(df)

    # ── Step 2: Engineer Features ─────────────────────────────
    df = engineer_features(df)

    # ── Step 3: Compute Baselines & Normalise ─────────────────
    df, track_baselines, global_baseline, actual_mode = (
        compute_baselines(df, mode=BASELINE_MODE)
    )

    # ── Step 4: Filter Outliers ───────────────────────────────
    df = filter_outlier_deltas(df)
    n_laps_filtered = len(df)

    # ── Step 5: Prepare Features ──────────────────────────────
    X, y, groups, cat_indices = prepare_features(df)

    # ── Step 6: Hyperparameter Tuning ─────────────────────────
    if OPTUNA_AVAILABLE:
        best_params = tune_hyperparameters(
            X, y, groups, cat_indices,
        )
    else:
        best_params = DEFAULT_PARAMS.copy()

    # ── Step 7: Cross-Validation ──────────────────────────────
    fold_metrics = cross_validate(
        X, y, groups, cat_indices,
        df_full=df,
        track_baselines=track_baselines,
        baseline_mode=actual_mode,
        params=best_params,
    )

    # ── Step 8: TrackName Ablation ────────────────────────────
    ablation_results = run_trackname_ablation(
        X, y, groups, cat_indices, best_params,
    )

    # ── Step 9: Train Final Model ─────────────────────────────
    cv_best_iters = [
        m["best_iteration"]
        for m in fold_metrics
        if m.get("best_iteration") is not None
    ]
    model = train_final_model(
        X, y, cat_indices,
        params=best_params,
        cv_best_iterations=cv_best_iters,
    )

    # ── Step 10: Save Baselines ───────────────────────────────
    track_only_baselines = df.attrs.get(
        "track_only_baselines", None
    )
    save_baselines(
        track_baselines,
        global_baseline,
        baseline_mode=actual_mode,
        track_only_baselines=track_only_baselines,
    )

    # ── Step 11: Feature Analysis ─────────────────────────────
    feat_imp = analyse_features(
        model, X, y, cat_indices,
        compute_interactions=False,
    )

    # ── Step 12: Prediction Test ──────────────────────────────
    print(f"\n{'═' * 60}")
    print(f"  PREDICTION TEST")
    print(f"{'═' * 60}")

    predictor = PacePredictor(
        str(MODEL_PATH), str(BASELINES_PATH)
    )

    sample = df.iloc[0]
    actual_abs = sample[RAW_TARGET]
    actual_delta = sample[DELTA_TARGET]
    track = sample["TrackName"]

    # Determine season for baseline lookup
    sample_season = (
        int(sample["Season"])
        if "Season" in sample.index
        else None
    )
    baseline = predictor._get_baseline(track, sample_season)

    # Determine total race laps for RaceLapFraction
    sample_race_id = sample[GROUP_KEY]
    total_laps = int(
        df.loc[
            df[GROUP_KEY] == sample_race_id,
            "RaceLapNumber",
        ].max()
    )

    predicted_abs = predictor.predict(
        compound=sample["Compound"],
        track_name=track,
        driver=sample["Driver"],
        team=sample["Team"],
        tire_age=int(sample["TireAge"]),
        race_lap_number=int(sample["RaceLapNumber"]),
        race_lap_fraction=float(sample["RaceLapFraction"]),
        total_race_laps=total_laps,
        track_temp=float(sample["TrackTemp"]),
        air_temp=float(sample["AirTemp"]),
        humidity=float(sample["Humidity"]),
        wind_speed=float(sample["WindSpeed"]),
        gap_to_car_ahead=float(sample["GapToCarAhead"]),
        drs_available=int(sample["DRS_Available"]),
        damage_state=float(sample.get("DamageState")),
        tyre_temp_c=float(sample.get("TyreTemp_C")),
        sliding_proxy=float(sample.get("SlidingProxy")),
        fz_n=float(sample.get("Fz_N")),
        season=sample_season,
    )

    predicted_unc, std_unc = predictor.predict_with_uncertainty(
        compound=sample["Compound"],
        track_name=track,
        driver=sample["Driver"],
        team=sample["Team"],
        tire_age=int(sample["TireAge"]),
        race_lap_number=int(sample["RaceLapNumber"]),
        race_lap_fraction=float(sample["RaceLapFraction"]),
        total_race_laps=total_laps,
        track_temp=float(sample["TrackTemp"]),
        air_temp=float(sample["AirTemp"]),
        humidity=float(sample["Humidity"]),
        wind_speed=float(sample["WindSpeed"]),
        gap_to_car_ahead=float(sample["GapToCarAhead"]),
        drs_available=int(sample["DRS_Available"]),
        season=sample_season,
    )

    predicted_delta = predictor.predict_delta(
        compound=sample["Compound"],
        track_name=track,
        driver=sample["Driver"],
        team=sample["Team"],
        tire_age=int(sample["TireAge"]),
        race_lap_number=int(sample["RaceLapNumber"]),
        race_lap_fraction=float(sample["RaceLapFraction"]),
        total_race_laps=total_laps,
        track_temp=float(sample["TrackTemp"]),
        air_temp=float(sample["AirTemp"]),
        humidity=float(sample["Humidity"]),
        wind_speed=float(sample["WindSpeed"]),
        gap_to_car_ahead=float(sample["GapToCarAhead"]),
        drs_available=int(sample["DRS_Available"]),
    )

    pipeline_elapsed = time.perf_counter() - pipeline_start

    print(f"\n  Sample prediction:")
    print(f"    Driver:         {sample['Driver']} "
          f"({sample['Team']})")
    print(f"    Track:          {track}")
    if sample_season:
        print(f"    Season:         {sample_season}")
    print(f"    Track baseline: {baseline:.2f}s "
          f"(mode: {predictor._baseline_mode})")
    print(f"    Compound:       {sample['Compound']} "
          f"(Age: {int(sample['TireAge'])})")
    print(f"    Race lap:       {int(sample['RaceLapNumber'])} "
          f"/ {total_laps} "
          f"(frac: {sample['RaceLapFraction']:.3f})")
    print(f"    ---")
    print(f"    Actual abs:     {actual_abs:.3f}s")
    print(f"    Predicted abs:  {predicted_abs:.3f}s")
    print(f"    Abs error:      "
          f"{predicted_abs - actual_abs:+.3f}s")
    print(f"    ---")
    print(f"    Actual delta:   {actual_delta:+.3f}s")
    print(f"    Predicted delta:{predicted_delta:+.3f}s")
    print(f"    Delta error:    "
          f"{predicted_delta - actual_delta:+.3f}s")
    print(f"    ---")
    print(f"    Uncertainty:    ±{std_unc:.3f}s")

    # ── Step 13: Save Report ──────────────────────────────────
    save_report(
        fold_metrics=fold_metrics,
        feat_imp=feat_imp,
        params=best_params,
        track_baselines=track_baselines,
        baseline_mode=actual_mode,
        n_laps=n_laps_raw,
        n_races=df[GROUP_KEY].nunique(),
        n_laps_after_filter=n_laps_filtered,
        ablation_results=ablation_results,
        total_time=pipeline_elapsed,
    )

    # ── Summary ───────────────────────────────────────────────
    avg_r2 = np.mean(
        [m["delta_r2"] for m in fold_metrics]
    )
    avg_rmse = np.mean(
        [m["delta_rmse"] for m in fold_metrics]
    )
    avg_best_iter = np.mean(
        [m["best_iteration"] for m in fold_metrics]
    )

    print(f"\n{'═' * 60}")
    print(f"  PIPELINE COMPLETE (v2)")
    print(f"{'═' * 60}")
    print(f"  Total: {pipeline_elapsed:.1f}s "
          f"({pipeline_elapsed / 60:.1f} min)")
    print(f"  GPU:   {'Yes' if GPU_AVAILABLE else 'No'}")
    print(f"  Baseline mode:   {actual_mode}")
    print(f"  Laps:  {n_laps_raw:,} raw → "
          f"{n_laps_filtered:,} after filtering")
    print(f"  Delta RMSE:      {avg_rmse:.4f}s")
    print(f"  Delta R²:        {avg_r2:.4f}")
    print(f"  Avg best_iter:   {avg_best_iter:.0f}")
    print(f"  Model predicts:  delta from track baseline")
    print(f"  PacePredictor converts back to absolute times")

    # ── Health assessment ─────────────────────────────────────
    print(f"\n  Health Assessment:")
    issues = []

    if avg_r2 < 0:
        issues.append(
            " Negative R² — model has no predictive power"
        )
    elif avg_r2 < 0.15:
        issues.append(
            f"  Low R² ({avg_r2:.4f}) — "
            f"model explains < 15% of variance"
        )
    else:
        print(f"     R² = {avg_r2:.4f}")

    if avg_best_iter < 50:
        issues.append(
            f"  Low best_iter ({avg_best_iter:.0f}) — "
            f"model may be overfitting immediately"
        )
    elif avg_best_iter < 200:
        issues.append(
            f"  Moderate best_iter ({avg_best_iter:.0f}) — "
            f"consider lowering learning_rate further"
        )
    else:
        print(
            f"     Avg best_iter = {avg_best_iter:.0f}"
        )

    if avg_rmse > 1.5:
        issues.append(
            f"  RMSE ({avg_rmse:.4f}s) may be too high "
            f"for reliable strategy decisions (1-stop vs "
            f"2-stop typically differs by 1-3s/lap)"
        )
    else:
        print(
            f"    RMSE = {avg_rmse:.4f}s"
        )

    if issues:
        for issue in issues:
            print(f"    {issue}")
        print(
            f"\n   If issues persist, consider:"
        )
        print(
            f"     - Checking data_pipeline.py for "
            f"better SC/VSC filtering"
        )
        print(
            f"     - Adding stint-level features "
            f"(StintLap, StintNumber)"
        )
        print(
            f"     - Trying MAE loss instead of RMSE "
            f"(more robust to outliers)"
        )
        print(
            f"     - Reducing learning_rate to 0.01 "
            f"with more iterations"
        )
    else:
        print(
            f"\n All health checks passed. "
            f"Ready for strategy engine."
        )

    print(
        f"\n Model training complete (v2)."
    )