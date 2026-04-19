"""
Predict F1 lap-time DELTAS
using a two-tier feature architecture:

    STRATEGY  — lap N state used to predict lap N+1 time
                            (Compound, TrackName, Driver, Team,
                             StintNumber, RaceLapNumber)
    PHYSICS   — modelled car / tyre state, projectable forward
                via physics models the strategy engine maintains
                (FuelProxy, TyreHealth, TyreTemp_C)

Target is shifted LapTimeSec from lap N+1.
Features are observed values from lap N.

Input:
    training_data_with_physics_shifted.parquet  — from build_v3_features.py
    models/v3_feature_config.json  (optional)

Output:
    models/pace_model.cbm
    models/track_baselines.json
    models/training_report.txt
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
        "  Optuna not installed. Hyperparameter tuning "
        "will be skipped. Install with: pip install optuna"
    )


# ═══════════════════════════════════════════════════════════════
# 1. GPU DETECTION
# ═══════════════════════════════════════════════════════════════


def detect_gpu() -> bool:
    """Detect CUDA GPU by attempting a tiny CatBoost fit."""
    print(" Detecting GPU availability…")

    try:
        from catboost.utils import get_gpu_device_count

        gpu_count = int(get_gpu_device_count())
    except Exception as e:
        print(f"GPU query failed: {e}")
        return False

    if gpu_count <= 0:
        print("    No CUDA GPU detected by CatBoost")
        return False

    print(f"   CatBoost detected {gpu_count} CUDA device(s)")

    try:
        CatBoostRegressor(
            iterations=1,
            task_type="GPU",
            devices="0",
            verbose=0,
        ).fit(
            np.array([[1, 2], [3, 4], [5, 6]]),
            np.array([1, 2, 3]),
        )
        print("GPU training verified")
        return True
    except Exception as e:
        print(f"GPU fit failed: {e}")
        return False


# ═══════════════════════════════════════════════════════════════
# 2. CONFIGURATION
# ═══════════════════════════════════════════════════════════════

# ── Paths ─────────────────────────────────────────────────────
MODEL_DIR = pathlib.Path("models")
MODEL_DIR.mkdir(exist_ok=True)
MODEL_PATH = MODEL_DIR / "pace_model.cbm"
BASELINES_PATH = MODEL_DIR / "track_baselines.json"
REPORT_PATH = MODEL_DIR / "training_report.txt"
V3_CONFIG_PATH = MODEL_DIR / "v3_feature_config.json"

# ── Target ────────────────────────────────────────────────────
RAW_TARGET = "LapTimeSec"
DELTA_TARGET = "LapTimeDelta"

# ── Group key ─────────────────────────────────────────────────
GROUP_KEY = "RaceID"

# ── Feature Tiers ─────────────────────────────────────────────
#
# STRATEGY: deterministic — the strategy engine computes these
# directly from lap N race state.
STRATEGY_CATEGORICAL = [
    "Compound",
    "TrackName",
    "Driver",
    "Team",
]

STRATEGY_NUMERICAL = [
    "StintNumber",
    "RaceLapNumber",
]

# PHYSICS: projectable — the strategy engine computes these
# via physics sub-models (fuel model, degradation model,
# thermal model).  Optional: None → NaN → CatBoost ignores.
PHYSICS_FEATURES = [
    "FuelProxy",    # fuel model: race_lap_number / total_laps (normalised fuel burn)
    "TyreHealth",   # degradation model: f(compound, tire_age, track)
    "TyreTemp_C",   # thermal model: f(compound, tire_age, track, ambient)
]

# ── Composite feature lists ──────────────────────────────────
CATEGORICAL_FEATURES = STRATEGY_CATEGORICAL.copy()

NUMERICAL_FEATURES = STRATEGY_NUMERICAL + PHYSICS_FEATURES

ALL_FEATURES = CATEGORICAL_FEATURES + NUMERICAL_FEATURES


DATA_PATH = "training_data_with_physics_shifted.parquet"

if V3_CONFIG_PATH.exists():
    try:
        with open(V3_CONFIG_PATH, "r") as _f:
            _v3_config = json.load(_f)
        _config_data_path = _v3_config.get("data_path", "")
        if (
            _config_data_path
            and pathlib.Path(_config_data_path).exists()
        ):
            DATA_PATH = _config_data_path

        _cfg_cat = _v3_config.get("all_categorical", [])
        _cfg_num = _v3_config.get("all_numerical", [])
        if _cfg_cat and _cfg_num:
            _missing_cat = [
                c for c in CATEGORICAL_FEATURES
                if c not in _cfg_cat
            ]
            _missing_num = [
                c for c in NUMERICAL_FEATURES
                if c not in _cfg_num
            ]
            if _missing_cat or _missing_num:
                print(
                    f"Feature mismatch with config. "
                    f"Missing cat: {_missing_cat}, "
                    f"Missing num: {_missing_num}"
                )

        print(f"V3 config loaded: {V3_CONFIG_PATH}")
        print(f"   Data: {DATA_PATH}")
        print(f"   Features: {_v3_config.get('n_features', '?')}")
    except Exception as _e:
        print(f"Could not load V3 config: {_e}")

# ── Outlier Filtering ─────────────────────────────────────────
# Removes around 0.6% of laps  with extreme deltas.
# Asymmetric quantiles reflect the right-skewed delta distribution:
#   - Lower 0.1%  timing errors, anomalous fast laps
#   - Upper 0.5%   pit in/out
OUTLIER_CONFIG: dict[str, Any] = {
    "lower_quantile": 0.001,
    "upper_quantile": 0.995,
    "hard_cap_seconds": 15.0,
}

# ── Baseline Mode ─────────────────────────────────────────────
BASELINE_MODE = "track_season"

# ── GPU ───────────────────────────────────────────────────────
GPU_AVAILABLE = detect_gpu()
TASK_TYPE = "GPU" if GPU_AVAILABLE else "CPU"

# ── CatBoost Defaults ────────────────────────────────────────
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
TUNING_CONFIG: dict[str, Any] = {
    "n_trials": 35,
    "n_splits": 3,
    "subsample_frac": 1.0,
    "max_iterations": 2500,
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
    print(f"\nLoading data from: {path}")
    df = (
        pd.read_parquet(path)
        if path.endswith(".parquet")
        else pd.read_csv(path)
    )
    print(f"   Rows: {len(df):,}")
    print(f"   Columns: {len(df.columns)}")

    # Backward-compatible aliasing for stint feature naming.
    if "StintNumber" not in df.columns and "Stint" in df.columns:
        df["StintNumber"] = df["Stint"]
        print("Aliased Stint -> StintNumber")

    # Rename RaceLapFraction → FuelProxy if source data uses old name
    if "FuelProxy" not in df.columns and "RaceLapFraction" in df.columns:
        df["FuelProxy"] = df["RaceLapFraction"]
        print("Aliased RaceLapFraction -> FuelProxy")

    skip_check = {"TireAgeSq", "FuelProxy"}
    required = [
        c for c in ALL_FEATURES if c not in skip_check
    ] + [RAW_TARGET, GROUP_KEY]

    missing = [c for c in required if c not in df.columns]

    if missing:
        print(f"Missing columns: {missing}")
        critical_missing = [
            c for c in missing
            if c in (
                STRATEGY_CATEGORICAL
                + STRATEGY_NUMERICAL
                + [RAW_TARGET, GROUP_KEY]
            )
        ]
        if critical_missing:
            raise ValueError(
                f"Critical columns missing: {critical_missing}"
            )
        print("   Continuing with available features…")

    for col in CATEGORICAL_FEATURES:
        if col in df.columns:
            df[col] = df[col].astype(str)

    print(f"\n   Feature availability:")
    print(
        f"     Strategy (cat):  "
        f"{[c for c in STRATEGY_CATEGORICAL if c in df.columns]}"
    )
    print(
        f"     Strategy (num):  "
        f"{[c for c in STRATEGY_NUMERICAL if c in df.columns]}"
    )
    print(
        f"     Physics:         "
        f"{[c for c in PHYSICS_FEATURES if c in df.columns]}"
    )

    print(f"\n   Raw target ({RAW_TARGET}):")
    print(
        f"     Mean:  {df[RAW_TARGET].mean():.2f}s  "
        f"Std: {df[RAW_TARGET].std():.2f}s"
    )
    print(
        f"     Range: {df[RAW_TARGET].min():.2f}s – "
        f"{df[RAW_TARGET].max():.2f}s"
    )
    print(
        f"   Groups ({GROUP_KEY}): "
        f"{df[GROUP_KEY].nunique()} races"
    )

    return df


def ensure_engineered_features(
    df: pd.DataFrame,
) -> pd.DataFrame:
    """Ensure engineered features exist."""
    df = df.copy()

    if "TireAgeSq" not in df.columns and "TireAge" in df.columns:
        df["TireAgeSq"] = df["TireAge"] ** 2
        print("Engineered TireAgeSq")

    if "RaceID" in df.columns:
        df["RaceID"] = df["RaceID"].astype(str)

    # Engineer FuelProxy (normalised race progress as fuel burn proxy)
    if "FuelProxy" not in df.columns:
        if "RaceLapFraction" in df.columns:
            df["FuelProxy"] = df["RaceLapFraction"]
            print("Engineered FuelProxy from RaceLapFraction")
        elif "RaceLapNumber" in df.columns:
            max_laps = df.groupby(GROUP_KEY)[
                "RaceLapNumber"
            ].transform("max")
            df["FuelProxy"] = (
                df["RaceLapNumber"] / max_laps
            ).clip(0.0, 1.0)
            print("Engineered FuelProxy from RaceLapNumber / TotalLaps")

    if "TotalRaceLaps" not in df.columns and "RaceLapNumber" in df.columns:
        df["TotalRaceLaps"] = df.groupby(GROUP_KEY)["RaceLapNumber"].transform("max")
        print("Engineered TotalRaceLaps")

    if "Season" not in df.columns:
        if "RaceDate" in df.columns:
            df["Season"] = pd.to_datetime(
                df["RaceDate"]
            ).dt.year
        else:
            try:
                df["Season"] = (
                    df[GROUP_KEY]
                    .astype(str)
                    .str.extract(r"(20[2-9]\d)", expand=False)
                    .astype(float)
                    .fillna(0)
                    .astype(int)
                )
            except Exception:
                df["Season"] = 2024
        print(
            f"   🔧 Extracted Season: "
            f"{sorted(df['Season'].unique())}"
        )

    return df


def compute_baselines(
    df: pd.DataFrame,
    mode: str = BASELINE_MODE,
) -> tuple[pd.DataFrame, dict[str, float], float, str, dict[str, float]]:
    """
    Compute per-track baselines and normalise target to delta.

    Returns
    -------
    df : DataFrame with DELTA_TARGET column added
    track_baselines : primary baseline dict
    global_baseline : fallback median
    actual_mode : "track_season" or "track"
    track_only_baselines : track-level fallbacks (for unseen seasons)
    """
    print(f"\n Computing per-track baselines (mode: {mode})…")

    global_baseline = float(df[RAW_TARGET].median())
    track_baselines: dict[str, float] = {}
    track_only_baselines: dict[str, float] = {}
    actual_mode = mode

    if mode == "track_season" and "Season" in df.columns:
        n_seasons = df["Season"].nunique()
        if n_seasons > 1:
            df = df.copy()
            baseline_key = (
                df["TrackName"] + "_"
                + df["Season"].astype(str)
            )
            track_baselines = {
                str(k): float(v)
                for k, v in df.groupby(baseline_key)[
                    RAW_TARGET
                ].median().to_dict().items()
            }
            track_only_baselines = {
                str(k): float(v)
                for k, v in df.groupby("TrackName")[
                    RAW_TARGET
                ].median().to_dict().items()
            }
            df[DELTA_TARGET] = (
                df[RAW_TARGET] - baseline_key.map(track_baselines)
            )
            print(
                f"   Baseline groups: "
                f"{len(track_baselines)} (track × season)"
            )
            print(f"   Seasons: {n_seasons}")
        else:
            actual_mode = "track"
    else:
        actual_mode = "track"

    if actual_mode == "track":
        df = df.copy()
        track_baselines = {
            str(k): float(v)
            for k, v in df.groupby("TrackName")[
                RAW_TARGET
            ].median().to_dict().items()
        }
        track_only_baselines = dict(track_baselines)
        df[DELTA_TARGET] = (
            df[RAW_TARGET]
            - df["TrackName"].map(track_baselines)
        )
        print(
            f"   Baseline groups: "
            f"{len(track_baselines)} (track only)"
        )

    print(f"   Global median: {global_baseline:.2f}s")
    print(f"\n   Delta target ({DELTA_TARGET}):")
    print(
        f"     Mean:  {df[DELTA_TARGET].mean():.2f}s  "
        f"Std: {df[DELTA_TARGET].std():.2f}s"
    )
    print(
        f"     Range: {df[DELTA_TARGET].min():.2f}s – "
        f"{df[DELTA_TARGET].max():.2f}s"
    )

    print(f"\n   Baselines ({len(track_baselines)} groups):")
    for key in sorted(track_baselines)[:10]:
        print(
            f"     {key:35s} {track_baselines[key]:7.2f}s"
        )
    if len(track_baselines) > 10:
        print(f"     … and {len(track_baselines) - 10} more")

    return (
        df,
        track_baselines,
        global_baseline,
        actual_mode,
        track_only_baselines,
    )


def filter_outlier_deltas(
    df: pd.DataFrame,
    config: Optional[dict] = None,
) -> pd.DataFrame:
    """Remove laps with extreme deltas."""
    if config is None:
        config = OUTLIER_CONFIG

    n_before = len(df)

    q_low = df[DELTA_TARGET].quantile(config["lower_quantile"])
    q_high = df[DELTA_TARGET].quantile(
        config["upper_quantile"]
    )
    hard_cap = config["hard_cap_seconds"]

    lower_bound = max(q_low, -hard_cap)
    upper_bound = min(q_high, hard_cap)

    mask = (df[DELTA_TARGET] >= lower_bound) & (
        df[DELTA_TARGET] <= upper_bound
    )
    df_clean = df[mask].copy()
    n_removed = n_before - len(df_clean)

    print(f"\n Outlier filtering:")
    print(
        f"   Effective: [{lower_bound:+.2f}s, "
        f"{upper_bound:+.2f}s]"
    )
    print(
        f"   Removed: {n_removed:,} laps "
        f"({n_removed / n_before:.1%})"
    )
    print(f"   Remaining: {len(df_clean):,} laps")
    print(
        f"   New delta std: "
        f"{df_clean[DELTA_TARGET].std():.2f}s"
    )

    return df_clean


def prepare_features(
    df: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.Series, pd.Series, list[int]]:
    """Extract X, y, groups, cat indices."""
    available_features = [
        c for c in ALL_FEATURES if c in df.columns
    ]
    available_cat = [
        c for c in CATEGORICAL_FEATURES if c in df.columns
    ]

    X = df[available_features].copy()
    y = df[DELTA_TARGET].copy()
    groups = df[GROUP_KEY].copy()
    cat_indices = [
        available_features.index(c) for c in available_cat
    ]

    physics_present = [
        c for c in PHYSICS_FEATURES if c in X.columns
    ]
    if physics_present:
        phys_nan = X[physics_present].isna().sum().sum()
        phys_cells = len(X) * len(physics_present)
        phys_nan_pct = phys_nan / phys_cells * 100
    else:
        phys_nan_pct = 0.0

    strategy_num_present = [
        c for c in STRATEGY_NUMERICAL if c in X.columns
    ]

    print(f"\n Feature matrix: {X.shape}")
    print(f"   Target: {DELTA_TARGET}")
    print(
        f"   Categorical ({len(available_cat)}): "
        f"{available_cat}"
    )
    print(
        f"   Strategy num ({len(strategy_num_present)}): "
        f"{strategy_num_present}"
    )
    print(
        f"   Physics ({len(physics_present)}): "
        f"{physics_present}"
    )
    print(f"   Physics NaN: {phys_nan_pct:.1f}%")
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
    global_baseline: float,
    track_only_baselines: dict[str, float],
    baseline_mode: str = "track",
    params: Optional[dict] = None,
    n_splits: Optional[int] = None,
) -> list:
    """GroupKFold CV with delta + absolute metrics."""
    if params is None:
        params = DEFAULT_PARAMS.copy()
    n_splits = (
        int(CV_CONFIG["n_splits"])
        if n_splits is None
        else int(n_splits)
    )

    cv_params = _make_cv_params(params, use_gpu=True)
    es_rounds = CV_CONFIG["early_stopping_rounds"]
    gkf = GroupKFold(n_splits=n_splits)
    fold_metrics: list[dict] = []

    device_label = (
        "GPU  "
        if cv_params.get("task_type") == "GPU"
        else "CPU"
    )

    print(f"\n{'═' * 65}")
    print(f"  CROSS-VALIDATION ({n_splits}-Fold GroupKFold)")
    print(f"  Device: {device_label}")
    print(f"  Target: {DELTA_TARGET}")
    print(f"  Features: {X.shape[1]}")
    print(f"{'═' * 65}")

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

        delta_rmse = np.sqrt(
            mean_squared_error(y_test, delta_preds)
        )
        delta_mae = mean_absolute_error(y_test, delta_preds)
        delta_r2 = r2_score(y_test, delta_preds)

        # ── Absolute metrics ──────────────────────────────────
        if baseline_mode == "track_season" and "Season" in df_full.columns:
            test_keys = (
                df_full.iloc[test_idx]["TrackName"]
                + "_"
                + df_full.iloc[test_idx]["Season"].astype(str)
            )
            test_baselines = test_keys.map(
                track_baselines
            ).values.astype(float)

            # Fallback: track-only → global
            nan_mask = np.isnan(test_baselines)
            if nan_mask.any() and track_only_baselines:
                fb = (
                    df_full.iloc[test_idx]
                    .loc[nan_mask, "TrackName"]
                    .map(track_only_baselines)
                    .values
                )
                test_baselines[nan_mask] = fb

            nan_mask = np.isnan(test_baselines)
            if nan_mask.any():
                test_baselines[nan_mask] = global_baseline
        else:
            test_baselines = (
                X_test["TrackName"]
                .map(track_baselines)
                .values
                .astype(float)
            )
            nan_mask = np.isnan(test_baselines)
            if nan_mask.any():
                test_baselines[nan_mask] = global_baseline

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
        print(f"   Abs   RMSE: {abs_rmse:.4f}s")
        print(
            f"{fold_elapsed:.1f}s | "
            f"best_iter: {best_iter}"
        )

        if best_iter is not None and best_iter < 50:
            print(f" best_iter={best_iter} very low")

        # Per-compound breakdown
        test_df = X_test.copy()
        test_df["_actual"] = y_test.values
        test_df["_pred"] = delta_preds

        for compound in sorted(test_df["Compound"].unique()):
            mask = test_df["Compound"] == compound
            comp_rmse = np.sqrt(
                mean_squared_error(
                    test_df.loc[mask, "_actual"],
                    test_df.loc[mask, "_pred"],
                )
            )
            print(
                f"     {compound:8s}: RMSE={comp_rmse:.4f}s "
                f"({mask.sum()} laps)"
            )

        fold_metrics.append(
            {
                "fold": fold_i,
                "delta_rmse": delta_rmse,
                "delta_mae": delta_mae,
                "delta_r2": delta_r2,
                "abs_rmse": abs_rmse,
                "test_races": list(test_races),
                "n_test_laps": len(test_idx),
                "best_iteration": best_iter,
                "elapsed_sec": fold_elapsed,
            }
        )

    # ── Summary ───────────────────────────────────────────────
    cv_elapsed = time.perf_counter() - cv_start
    avg = {
        k: np.mean([m[k] for m in fold_metrics])
        for k in [
            "delta_rmse",
            "delta_mae",
            "delta_r2",
            "abs_rmse",
        ]
    }
    std_rmse = np.std([m["delta_rmse"] for m in fold_metrics])
    avg_best_iter = np.mean(
        [m["best_iteration"] for m in fold_metrics]
    )

    print(f"\n{'═' * 65}")
    print(f"  CV RESULTS ({device_label})")
    print(f"{'═' * 65}")
    print(
        f"  Delta RMSE: {avg['delta_rmse']:.4f}s "
        f"± {std_rmse:.4f}s"
    )
    print(f"  Delta MAE:  {avg['delta_mae']:.4f}s")
    print(f"  Delta R²:   {avg['delta_r2']:.4f}")
    print(f"  Abs   RMSE: {avg['abs_rmse']:.4f}s")
    print(f"  Avg best_iter: {avg_best_iter:.0f}")
    print(f"  CV time:    {cv_elapsed:.1f}s")

    if avg["delta_r2"] < 0:
        print("Negative R²")
    if avg_best_iter < 50:
        print("Average best_iter < 50")

    print(f"{'═' * 65}")

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
    """Optuna hyperparameter search."""
    if not OPTUNA_AVAILABLE:
        print("Optuna not available. Using defaults.")
        return DEFAULT_PARAMS.copy()

    n_trials = (
        int(TUNING_CONFIG["n_trials"])
        if n_trials is None
        else int(n_trials)
    )
    n_splits = (
        int(TUNING_CONFIG["n_splits"])
        if n_splits is None
        else int(n_splits)
    )
    subsample_frac = (
        float(TUNING_CONFIG["subsample_frac"])
        if subsample_frac is None
        else float(subsample_frac)
    )

    max_iters = int(TUNING_CONFIG["max_iterations"])
    es_rounds = int(TUNING_CONFIG["early_stopping_rounds"])
    device_label = "GPU " if GPU_AVAILABLE else "CPU"
    max_fits = n_trials * n_splits

    print(f"\n{'═' * 65}")
    print(f"  HYPERPARAMETER TUNING ({n_trials} trials)")
    print(f"{'═' * 65}")
    print(f"  Device:          {device_label}")
    print(f"  Max fits:        {max_fits}")
    print(f"  Train subsample: {subsample_frac:.0%}")
    print(f"  Iter cap:        {max_iters}")
    print(f"  Early stop:      {es_rounds}")
    print(f"  Features:        {X.shape[1]}")
    print(f"{'═' * 65}")

    gkf = GroupKFold(n_splits=n_splits)
    fold_indices = list(gkf.split(X, y, groups=groups))
    tuning_start = time.perf_counter()

    def objective(trial: "optuna.Trial") -> float:
        params: dict[str, Any] = {
            "iterations": trial.suggest_int(
                "iterations",
                800,
                max_iters,
            ),
            "learning_rate": trial.suggest_float(
                "learning_rate",
                0.01,
                0.06,
                log=True,
            ),
            "depth": trial.suggest_int("depth", 4, 8),
            "l2_leaf_reg": trial.suggest_float(
                "l2_leaf_reg",
                3.0,
                30.0,
                log=True,
            ),
            "min_data_in_leaf": trial.suggest_int(
                "min_data_in_leaf",
                20,
                100,
            ),
            "bagging_temperature": trial.suggest_float(
                "bagging_temperature",
                0.5,
                2.0,
            ),
            "random_strength": trial.suggest_float(
                "random_strength",
                0.5,
                2.0,
            ),
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
                    train_idx,
                    size=n_sub,
                    replace=False,
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
    study.enqueue_trial(
        {
            "iterations": min(
                DEFAULT_PARAMS["iterations"], max_iters
            ),
            "learning_rate": DEFAULT_PARAMS["learning_rate"],
            "depth": DEFAULT_PARAMS["depth"],
            "l2_leaf_reg": DEFAULT_PARAMS["l2_leaf_reg"],
            "min_data_in_leaf": DEFAULT_PARAMS[
                "min_data_in_leaf"
            ],
            "bagging_temperature": 1.0,
            "random_strength": 1.0,
        }
    )

    study.optimize(
        cast(Any, objective),
        n_trials=n_trials,
        show_progress_bar=True,
    )

    tuning_elapsed = time.perf_counter() - tuning_start

    n_complete = len(
        [
            t
            for t in study.trials
            if t.state == optuna.trial.TrialState.COMPLETE
        ]
    )
    n_pruned = len(
        [
            t
            for t in study.trials
            if t.state == optuna.trial.TrialState.PRUNED
        ]
    )
    actual_fits = sum(
        len(t.intermediate_values) for t in study.trials
    )

    best = study.best_params
    print(f"\n Best RMSE:   {study.best_value:.4f}s")
    print(
        f"Time:        {tuning_elapsed:.1f}s "
        f"({tuning_elapsed / 60:.1f} min)"
    )
    print(
        f" Trials:       {n_complete} complete, "
        f"{n_pruned} pruned"
    )
    print(
        f"Fold fits:    {actual_fits} "
        f"(vs {max_fits} max)"
    )
    print("  Best parameters:")
    for k, v in sorted(best.items()):
        print(f"    {k}: {v}")

    if best.get("iterations", 999) < 200:
        print("Tuned iterations very low. Clamping.")
        best["iterations"] = 500

    best["loss_function"] = "RMSE"
    best["random_seed"] = 42
    best["verbose"] = 200
    best.setdefault("bagging_temperature", 1.0)
    best.setdefault("random_strength", 1.0)

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
    """Train final model on ALL data."""
    if params is None:
        params = DEFAULT_PARAMS.copy()

    final_params = {**params}
    final_params["posterior_sampling"] = True
    final_params["langevin"] = True
    final_params["task_type"] = "CPU"
    final_params.pop("devices", None)
    # Remove GPU-only keys that conflict with CPU training
    final_params.pop("border_count", None)

    MIN_FINAL_ITERATIONS = 200

    if cv_best_iterations and all(
        b is not None and b > 0 for b in cv_best_iterations
    ):
        adaptive_iters = int(
            np.median(cv_best_iterations) * 1.15
        )
        adaptive_iters = max(
            MIN_FINAL_ITERATIONS,
            min(adaptive_iters, 5000),
        )
        original_iters = final_params.get("iterations", 2500)

        if adaptive_iters < original_iters:
            print(
                f"\n Adaptive iterations: "
                f"{original_iters} → {adaptive_iters} "
                f"(CV best: {cv_best_iterations})"
            )
            final_params["iterations"] = adaptive_iters
        else:
            print(
                f"\n Keeping {original_iters} iterations "
                f"(CV best: {cv_best_iterations})"
            )

    print(f"\n{'═' * 65}")
    print("  TRAINING FINAL MODEL")
    print(f"{'═' * 65}")
    print(f"  Training on {len(X):,} laps (all data)")
    print(f"  Features: {X.shape[1]}")
    print(f"  Target: {DELTA_TARGET}")
    print("  Device: CPU (required for virtual ensembles)")
    if GPU_AVAILABLE:
        print("GPU was used for CV & tuning")
    print("  Parameters:")
    for k, v in sorted(final_params.items()):
        if k != "verbose":
            print(f"    {k}: {v}")

    pool = Pool(X, y, cat_features=cat_indices)

    train_start = time.perf_counter()
    model = CatBoostRegressor(**final_params)
    model.fit(pool)
    train_elapsed = time.perf_counter() - train_start

    train_preds = model.predict(pool)
    train_rmse = np.sqrt(mean_squared_error(y, train_preds))
    train_r2 = r2_score(y, train_preds)
    print(
        f"\n  Train RMSE: {train_rmse:.4f}s | "
        f"R²: {train_r2:.4f}"
    )

    model.save_model(str(MODEL_PATH))
    print(f"Model saved to: {MODEL_PATH}")
    print(
        f"Training time: {train_elapsed:.1f}s "
        f"({train_elapsed / 60:.1f} min)"
    )

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
    """Feature importance with per-tier breakdown."""
    pool = Pool(X, y, cat_features=cat_indices)
    features_used = list(X.columns)

    importance_raw = model.get_feature_importance(
        pool, type=cast(Any, "PredictionValuesChange")
    )
    importance = np.asarray(
        importance_raw, dtype=float
    ).reshape(-1)
    importance_sum = (
        float(importance.sum()) if importance.size else 1.0
    )

    feat_imp = pd.DataFrame(
        {
            "Feature": features_used,
            "Importance": importance,
            "Importance_Pct": importance / importance_sum * 100,
        }
    ).sort_values("Importance", ascending=False)

    def _get_tier(feat: str) -> str:
        if feat in STRATEGY_CATEGORICAL + STRATEGY_NUMERICAL:
            return "strategy"
        elif feat in PHYSICS_FEATURES:
            return "physics"
        return "other"

    feat_imp["Tier"] = feat_imp["Feature"].map(_get_tier)

    print(f"\n{'═' * 65}")
    print("  FEATURE IMPORTANCE")
    print(f"{'═' * 65}")
    for _, row in feat_imp.iterrows():
        bar_len = int(row["Importance_Pct"] / 2)
        bar = "█" * bar_len
        tier_tag = f"[{row['Tier'][:4]:4s}]"
        print(
            f"  {tier_tag} {row['Feature']:30s} "
            f"{row['Importance_Pct']:5.1f}% {bar}"
        )

    print(f"\n  Per-tier breakdown:")
    for tier in ["strategy", "physics"]:
        tier_df = feat_imp[feat_imp["Tier"] == tier]
        tier_pct = tier_df["Importance_Pct"].sum()
        n_feats = len(tier_df)
        n_active = (tier_df["Importance_Pct"] > 0.1).sum()
        print(
            f"    {tier:10s}: {tier_pct:5.1f}% total "
            f"({n_active}/{n_feats} features active)"
        )

    # ── Physical validation ───────────────────────────────────
    print(f"\n  Physical validation:")

    top_5 = set(feat_imp.head(5)["Feature"])
    expected_dynamics = {
        "TireAge",
        "TireAgeSq",
        "Compound",
        "RaceLapNumber",
        "FuelProxy",
        "StintNumber",
        "TyreHealth",
    }
    found = top_5 & expected_dynamics
    if len(found) >= 2:
        print(f"Race dynamics in top 5: {found}")
    else:
        print(
            f"Expected dynamics not dominant. "
            f"Top 5: {top_5}"
        )

    max_pct = feat_imp["Importance_Pct"].iloc[0]
    max_feat = feat_imp["Feature"].iloc[0]
    if max_pct > 35:
        print(
            f"{max_feat} dominates at {max_pct:.1f}%"
        )
    else:
        print(
            f"Well-distributed "
            f"(max: {max_feat} at {max_pct:.1f}%)"
        )

    zero = feat_imp[feat_imp["Importance_Pct"] < 0.1][
        "Feature"
    ].tolist()
    if zero:
        print(f"Near-zero importance: {zero}")
    else:
        print("All features contributing")

    # TyreHealth vs TireAge — physics model vs linear proxy
    tyre_health_pct = feat_imp.loc[
        feat_imp["Feature"] == "TyreHealth", "Importance_Pct"
    ]
    tire_age_pct = feat_imp.loc[
        feat_imp["Feature"] == "TireAge", "Importance_Pct"
    ]
    if not tyre_health_pct.empty and not tire_age_pct.empty:
        th = tyre_health_pct.iloc[0]
        ta = tire_age_pct.iloc[0]
        if th > ta:
            print(
                f"TyreHealth ({th:.1f}%) > "
                f"TireAge ({ta:.1f}%) — physics model "
                f"captures degradation beyond linear age"
            )
        elif th > 1.0:
            print(
                f"TireAge ({ta:.1f}%) > "
                f"TyreHealth ({th:.1f}%) — both "
                f"contributing, physics adds marginal signal"
            )
        else:
            print(
                f"TyreHealth ({th:.1f}%) near zero — "
                f"check if feature varies within stints"
            )

    # ── Interactions ──────────────────────────────────────────
    if compute_interactions:
        print(f"\n{'═' * 65}")
        print("  TOP FEATURE INTERACTIONS")
        print(f"{'═' * 65}")

        try:
            interactions_raw = model.get_feature_importance(
                pool, type=cast(Any, "Interaction")
            )
            interactions = (
                np.asarray(interactions_raw, dtype=float)
                if interactions_raw is not None
                else np.empty((0, 3))
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
                    .map(
                        lambda i: (
                            features_used[int(i)]
                            if 0 <= int(i) < len(features_used)
                            else "?"
                        )
                    )
                )
                inter_df["Feature2"] = (
                    inter_df["Feature2_idx"]
                    .astype(int)
                    .map(
                        lambda i: (
                            features_used[int(i)]
                            if 0 <= int(i) < len(features_used)
                            else "?"
                        )
                    )
                )
                inter_df = inter_df.sort_values(
                    "Strength", ascending=False
                )
                for _, row in inter_df.head(10).iterrows():
                    print(
                        f"  {row['Feature1']:25s} × "
                        f"{row['Feature2']:25s} "
                        f"Str: {row['Strength']:.2f}"
                    )
        except Exception as e:
            print(f"Interactions failed: {e}")
    else:
        print("\n Interaction analysis skipped")

    return feat_imp


# ═══════════════════════════════════════════════════════════════
# 8. SAVE REPORT & BASELINES
# ═══════════════════════════════════════════════════════════════


def save_report(
    fold_metrics: list,
    feat_imp: pd.DataFrame,
    params: dict,
    track_baselines: dict[str, float],
    baseline_mode: str,
    n_laps_raw: int,
    n_laps_filtered: int,
    n_races: int,
    total_time: float = 0.0,
):
    """Save training report."""
    avg = {
        k: np.mean([m[k] for m in fold_metrics])
        for k in [
            "delta_rmse",
            "delta_mae",
            "delta_r2",
            "abs_rmse",
        ]
    }
    std_rmse = np.std([m["delta_rmse"] for m in fold_metrics])
    avg_best_iter = np.mean(
        [m["best_iteration"] for m in fold_metrics]
    )

    lines = [
        "F1 VIRTUAL RACE STRATEGIST — TRAINING REPORT (V3)",
        "=" * 60,
        "",
        f"Dataset: {n_laps_raw:,} laps → "
        f"{n_laps_filtered:,} after filtering",
        f"Races: {n_races}",
        f"Features: {len(feat_imp)} "
        f"(strategy + physics)",
        "Model: CatBoost Regressor",
        f"Target: {DELTA_TARGET}",
        f"Baseline: {baseline_mode}",
        f"GPU: {'Yes' if GPU_AVAILABLE else 'No'}",
        f"Pipeline time: {total_time:.1f}s",
        "",
        "FEATURE TIERS",
        "-" * 60,
        f"Strategy (cat):  {STRATEGY_CATEGORICAL}",
        f"Strategy (num):  {STRATEGY_NUMERICAL}",
        f"Physics:         {PHYSICS_FEATURES}",
        "",
        "CROSS-VALIDATION",
        "-" * 60,
        f"Folds: {CV_CONFIG['n_splits']}",
        f"Delta RMSE: {avg['delta_rmse']:.4f}s "
        f"± {std_rmse:.4f}s",
        f"Delta MAE:  {avg['delta_mae']:.4f}s",
        f"Delta R²:   {avg['delta_r2']:.4f}",
        f"Abs   RMSE: {avg['abs_rmse']:.4f}s",
        f"Avg best_iter: {avg_best_iter:.0f}",
        "",
    ]

    for m in fold_metrics:
        lines.append(
            f"  Fold {m['fold']}: "
            f"ΔRMSE={m['delta_rmse']:.4f}s "
            f"AbsRMSE={m['abs_rmse']:.4f}s "
            f"iter={m['best_iteration']} "
            f"({m['n_test_laps']} laps)"
        )

    lines.extend(["", "HYPERPARAMETERS", "-" * 60])
    for k, v in sorted(params.items()):
        lines.append(f"  {k}: {v}")

    lines.extend(["", "FEATURE IMPORTANCE", "-" * 60])
    for _, row in feat_imp.iterrows():
        lines.append(
            f"  [{row.get('Tier', '?'):8s}] "
            f"{row['Feature']:30s} "
            f"{row['Importance_Pct']:5.1f}%"
        )

    lines.extend(["", "TIER SUMMARY", "-" * 60])
    for tier in ["strategy", "physics"]:
        tier_df = feat_imp[feat_imp["Tier"] == tier]
        tier_pct = tier_df["Importance_Pct"].sum()
        n_active = (tier_df["Importance_Pct"] > 0.1).sum()
        lines.append(
            f"  {tier:10s}: {tier_pct:.1f}% "
            f"({n_active}/{len(tier_df)} active)"
        )

    lines.extend(
        [
            "",
            "TRACK BASELINES",
            "-" * 60,
            f"Mode: {baseline_mode}",
            f"Groups: {len(track_baselines)}",
        ]
    )
    for key in sorted(track_baselines):
        lines.append(
            f"  {key:35s} {track_baselines[key]:7.2f}s"
        )

    REPORT_PATH.write_text(
        "\n".join(lines), encoding="utf-8"
    )
    print(f"\n Report saved to: {REPORT_PATH}")


def save_baselines(
    track_baselines: dict[str, float],
    global_baseline: float,
    baseline_mode: str = "track",
    track_only_baselines: Optional[dict[str, float]] = None,
):
    """Save track baselines to JSON."""
    payload: dict[str, Any] = {
        "track_baselines": track_baselines,
        "global_baseline": global_baseline,
        "baseline_mode": baseline_mode,
        "model_version": "v3_strategy_only",
        "description": (
            "Per-track median lap times. "
            "Add to delta prediction for absolute time."
        ),
    }
    if track_only_baselines is not None:
        payload["track_only_baselines"] = track_only_baselines

    BASELINES_PATH.write_text(
        json.dumps(payload, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    print(f"Baselines saved to: {BASELINES_PATH}")


# ═══════════════════════════════════════════════════════════════
# 9. PREDICTION INTERFACE
# ═══════════════════════════════════════════════════════════════


class PacePredictor:
    """
    Prediction interface for the Strategy Engine.

    Uses strategy + projectable physics features.
    Every feature is computable for any hypothetical future
    lap — physics features are optional (None → NaN → CatBoost
    falls back to strategy tier).

    Returns absolute lap times by adding the per-track baseline
    to the model's delta prediction.
    """

    def __init__(
        self,
        model_path: Optional[str] = None,
        baselines_path: Optional[str] = None,
    ):
        if model_path is None:
            model_path = str(MODEL_PATH)
        if baselines_path is None:
            baselines_path = str(BASELINES_PATH)

        self.model = CatBoostRegressor()
        self.model.load_model(model_path)

        with open(baselines_path, "r") as f:
            payload = json.load(f)

        self._track_baselines: dict[str, float] = payload[
            "track_baselines"
        ]
        self._global_baseline: float = payload[
            "global_baseline"
        ]
        self._baseline_mode: str = payload.get(
            "baseline_mode", "track"
        )
        self._track_only_baselines: dict[str, float] = (
            payload.get("track_only_baselines", {})
        )

        raw_features = cast(Any, self.model.feature_names_)
        if isinstance(raw_features, list):
            self._model_features: list[str] = [
                str(f) for f in raw_features
            ]
        else:
            self._model_features = list(ALL_FEATURES)
        self._cat_indices = [
            i
            for i, name in enumerate(self._model_features)
            if name in CATEGORICAL_FEATURES
        ]

        print("PacePredictor loaded")
        print(
            f"   Baselines: {len(self._track_baselines)} "
            f"groups (mode: {self._baseline_mode})"
        )
        print(f"   Features: {len(self._model_features)}")
        print(
            f"   Strategy: "
            f"{[f for f in self._model_features if f in STRATEGY_CATEGORICAL + STRATEGY_NUMERICAL]}"
        )
        print(
            f"   Physics:  "
            f"{[f for f in self._model_features if f in PHYSICS_FEATURES]}"
        )

    def _get_baseline(
        self,
        track_name: str,
        season: Optional[int] = None,
    ) -> float:
        """Baseline lookup with fallback chain:
        track_season → track_only → global."""
        if (
            self._baseline_mode == "track_season"
            and season is not None
        ):
            key = f"{track_name}_{season}"
            if key in self._track_baselines:
                return self._track_baselines[key]
            if track_name in self._track_only_baselines:
                return self._track_only_baselines[track_name]
        elif track_name in self._track_baselines:
            return self._track_baselines[track_name]
        return self._global_baseline

    # ── Strategy Prediction ──────────────────────────────────

    def predict(
        self,
        compound: str,
        track_name: str,
        driver: str,
        team: str,
        tire_age: int,
        stint: int = 1,
        race_lap_number: int = 1,
        total_race_laps: Optional[int] = None,
        tyre_health: Optional[float] = None,
        tyre_temp_c: Optional[float] = None,
        fuel_proxy: Optional[float] = None,
        season: Optional[int] = None,
    ) -> float:
        """
        Predict absolute lap time for a hypothetical lap.

        Parameters
        ----------
        compound : str
            Tyre compound (SOFT, MEDIUM, HARD).
        track_name : str
            Circuit name matching training data.
        driver : str
            Driver abbreviation (e.g. "VER").
        team : str
            Team name (e.g. "Red Bull Racing").
        tire_age : int
            Laps since last pit stop.
        stint : int
            Stint number (1, 2, 3…).
        race_lap_number : int
            Current lap in the race.
        total_race_laps : int, optional
            Total laps in race (for FuelProxy computation).
        tyre_health : float, optional
            Projected tyre health from degradation model.
        tyre_temp_c : float, optional
            Projected tyre temperature from thermal model.
        fuel_proxy : float, optional
            Normalised race progress (0.0–1.0). If None,
            computed from race_lap_number / total_race_laps.
        season : int, optional
            Season year for baseline lookup.

        Returns
        -------
        float
            Predicted absolute lap time in seconds.
        """
        row = self._build_row(
            compound=compound,
            track_name=track_name,
            driver=driver,
            team=team,
            tire_age=tire_age,
            stint=stint,
            race_lap_number=race_lap_number,
            total_race_laps=total_race_laps,
            tyre_health=tyre_health,
            tyre_temp_c=tyre_temp_c,
            fuel_proxy=fuel_proxy,
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
        stint: int = 1,
        race_lap_number: int = 1,
        total_race_laps: Optional[int] = None,
        tyre_health: Optional[float] = None,
        tyre_temp_c: Optional[float] = None,
        fuel_proxy: Optional[float] = None,
    ) -> float:
        """
        Predict raw delta only (no baseline added).

        Useful for comparing stint options on the same track
        where the baseline cancels out.
        """
        row = self._build_row(
            compound=compound,
            track_name=track_name,
            driver=driver,
            team=team,
            tire_age=tire_age,
            stint=stint,
            race_lap_number=race_lap_number,
            total_race_laps=total_race_laps,
            tyre_health=tyre_health,
            tyre_temp_c=tyre_temp_c,
            fuel_proxy=fuel_proxy,
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
        stint: int = 1,
        race_lap_number: int = 1,
        total_race_laps: Optional[int] = None,
        tyre_health: Optional[float] = None,
        tyre_temp_c: Optional[float] = None,
        fuel_proxy: Optional[float] = None,
        season: Optional[int] = None,
        n_ensembles: int = 10,
    ) -> tuple[float, float]:
        """
        Predict absolute time + uncertainty estimate.

        Returns (mean_prediction, std_uncertainty) in seconds.
        Uses virtual ensembles (posterior_sampling / langevin).
        """
        row = self._build_row(
            compound=compound,
            track_name=track_name,
            driver=driver,
            team=team,
            tire_age=tire_age,
            stint=stint,
            race_lap_number=race_lap_number,
            total_race_laps=total_race_laps,
            tyre_health=tyre_health,
            tyre_temp_c=tyre_temp_c,
            fuel_proxy=fuel_proxy,
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

    # ── Batch Predictions ────────────────────────────────────

    def predict_batch(
        self,
        df: pd.DataFrame,
        season: Optional[int] = None,
    ) -> np.ndarray:
        """
        Batch predict absolute lap times from a DataFrame.

        Parameters
        ----------
        df : pd.DataFrame
            Must contain model feature columns.
        season : int, optional
            Season for baseline lookup (overrides per-row).

        Returns
        -------
        np.ndarray
            Absolute predicted lap times.
        """
        df = df.copy()
        for col in CATEGORICAL_FEATURES:
            if col in df.columns:
                df[col] = df[col].astype(str)

        if (
            "TireAgeSq" not in df.columns
            and "TireAge" in df.columns
        ):
            df["TireAgeSq"] = df["TireAge"] ** 2

        # Engineer FuelProxy from available columns
        if "FuelProxy" not in df.columns:
            if "RaceLapFraction" in df.columns:
                df["FuelProxy"] = df["RaceLapFraction"]
            elif "RaceLapNumber" in df.columns:
                if "TotalRaceLaps" in df.columns:
                    df["FuelProxy"] = (
                        df["RaceLapNumber"]
                        / df["TotalRaceLaps"]
                    ).clip(0.0, 1.0)
                elif GROUP_KEY in df.columns:
                    max_laps = df.groupby(GROUP_KEY)[
                        "RaceLapNumber"
                    ].transform("max")
                    df["FuelProxy"] = (
                        df["RaceLapNumber"] / max_laps
                    ).clip(0.0, 1.0)
                else:
                    df["FuelProxy"] = 0.5

        # Stint feature naming compatibility for batch inference.
        if "StintNumber" not in df.columns and "Stint" in df.columns:
            df["StintNumber"] = df["Stint"]
        if "Stint" not in df.columns and "StintNumber" in df.columns:
            df["Stint"] = df["StintNumber"]

        for col in self._model_features:
            if col not in df.columns:
                if col in CATEGORICAL_FEATURES:
                    df[col] = "UNKNOWN"
                else:
                    df[col] = np.nan

        pool = Pool(
            df[self._model_features],
            cat_features=self._cat_indices,
        )
        deltas = self.model.predict(pool)

        # Resolve baselines per row
        if (
            self._baseline_mode == "track_season"
            and (
                season is not None or "Season" in df.columns
            )
        ):
            if season is not None:
                baselines = (
                    df["TrackName"]
                    .map(
                        lambda t, s=season: self._get_baseline(
                            str(t), s
                        )
                    )
                    .values
                )
            else:
                baselines = (
                    df.apply(
                        lambda row: self._get_baseline(
                            str(row["TrackName"]),
                            int(row["Season"]),
                        ),
                        axis=1,
                    )
                    .values
                )
        else:
            baselines = (
                df["TrackName"]
                .map(lambda t: self._get_baseline(str(t)))
                .values
            )

        return baselines + deltas

    # ── Strategy Simulation ──────────────────────────────────

    def simulate_strategy(
        self,
        driver: str,
        team: str,
        track_name: str,
        current_lap: int,
        total_laps: int,
        initial_compound: str,
        pit_plan: list[dict],
        initial_tire_age: int = 1,
        season: Optional[int] = None,
        pit_loss_sec: float = 22.0,
        physics_projector: Optional[Any] = None,
    ) -> dict:
        """
        Simulate an entire pit strategy from current_lap
        to the end of the race.

        Parameters
        ----------
        driver, team, track_name : str
            Identity parameters.
        current_lap : int
            First lap to simulate.
        total_laps : int
            Total race laps.
        initial_compound : str
            Compound at the start of simulation.
        pit_plan : list of dict
            Each dict: {"pit_lap": int, "compound": str}
            — pit ON that lap, switch TO that compound.
        initial_tire_age : int
            Tyre age at the start of simulation (default 1).
        season : int, optional
            For baseline lookup.
        pit_loss_sec : float
            Time penalty per pit stop (default 22s).
        physics_projector : callable, optional
            Called as physics_projector(
                compound, tire_age, stint,
                race_lap_number, total_laps, track_name
            ) → dict with optional keys:
                "tyre_health", "tyre_temp_c", "fuel_proxy"
            Returns physics projections for each lap.
            If None, physics features are NaN (CatBoost
            falls back to strategy tier) except FuelProxy
            which is computed from race progress.

        Returns
        -------
        dict with:
            total_time : float — sum of lap times + pit losses
            lap_times : list[float] — per-lap predictions
            pit_stops : int
            compounds_used : list[str]
            schedule : list[dict] — per-lap details
            laps_simulated : int
        """
        sorted_pits = sorted(
            pit_plan, key=lambda p: p["pit_lap"]
        )
        pit_laps = {p["pit_lap"] for p in sorted_pits}
        pit_compound_map = {
            p["pit_lap"]: p["compound"]
            for p in sorted_pits
        }

        # Build per-lap schedule
        current_compound = initial_compound
        current_stint = 1
        tire_age = initial_tire_age
        schedule: list[dict] = []

        for lap in range(current_lap, total_laps + 1):
            if lap in pit_laps:
                current_stint += 1
                current_compound = pit_compound_map[lap]
                tire_age = 1

            schedule.append(
                {
                    "lap": lap,
                    "compound": current_compound,
                    "tire_age": tire_age,
                    "stint": current_stint,
                }
            )
            tire_age += 1

        # Predict each lap
        lap_times: list[float] = []
        compounds_used: set[str] = set()

        for entry in schedule:
            # Project physics if model provided
            tyre_health = None
            tyre_temp_c = None
            fuel_proxy = None

            if physics_projector is not None:
                try:
                    physics = physics_projector(
                        compound=entry["compound"],
                        tire_age=entry["tire_age"],
                        stint=entry["stint"],
                        race_lap_number=entry["lap"],
                        total_laps=total_laps,
                        track_name=track_name,
                    )
                    tyre_health = physics.get("tyre_health")
                    tyre_temp_c = physics.get("tyre_temp_c")
                    fuel_proxy = physics.get("fuel_proxy")
                except Exception:
                    pass

            # Default FuelProxy from race progress if not
            # provided by physics projector
            if fuel_proxy is None:
                fuel_proxy = min(
                    1.0, entry["lap"] / total_laps
                )

            predicted = self.predict(
                compound=entry["compound"],
                track_name=track_name,
                driver=driver,
                team=team,
                tire_age=entry["tire_age"],
                stint=entry["stint"],
                race_lap_number=entry["lap"],
                total_race_laps=total_laps,
                tyre_health=tyre_health,
                tyre_temp_c=tyre_temp_c,
                fuel_proxy=fuel_proxy,
                season=season,
            )
            lap_times.append(predicted)
            compounds_used.add(entry["compound"])

        n_pits = len(sorted_pits)
        total_time = sum(lap_times) + (n_pits * pit_loss_sec)

        return {
            "total_time": total_time,
            "lap_times": lap_times,
            "pit_stops": n_pits,
            "pit_loss_sec": pit_loss_sec,
            "compounds_used": sorted(compounds_used),
            "schedule": schedule,
            "laps_simulated": len(schedule),
        }

    # ── Internal Row Builder ─────────────────────────────────

    def _build_row(
        self,
        compound: str,
        track_name: str,
        driver: str,
        team: str,
        tire_age: int,
        stint: int = 1,
        race_lap_number: int = 1,
        total_race_laps: Optional[int] = None,
        tyre_health: Optional[float] = None,
        tyre_temp_c: Optional[float] = None,
        fuel_proxy: Optional[float] = None,
    ) -> pd.DataFrame:
        """
        Build a single-row DataFrame matching the model's
        trained feature set.

        None → NaN — CatBoost handles natively via default
        split directions learned during training.
        """
        tire_age_sq = tire_age ** 2

        # Compute FuelProxy: use explicit value, derive from
        # race progress, or fall back to 0.5
        if fuel_proxy is not None:
            fuel_proxy_val = float(fuel_proxy)
        elif total_race_laps and total_race_laps > 0:
            fuel_proxy_val = min(
                1.0, race_lap_number / total_race_laps
            )
        else:
            fuel_proxy_val = 0.5

        row_data: dict[str, Any] = {
            # Strategy categorical
            "Compound": str(compound).upper(),
            "TrackName": str(track_name),
            "Driver": str(driver),
            "Team": str(team),
            # Strategy numerical
            "TireAge": int(tire_age),
            "TireAgeSq": int(tire_age_sq),
            "StintNumber": int(stint),
            # Backward compatibility for models trained with legacy naming.
            "Stint": int(stint),
            "RaceLapNumber": int(race_lap_number),
            # Physics
            "FuelProxy": float(fuel_proxy_val),
            "TyreHealth": (
                float(tyre_health)
                if tyre_health is not None
                else np.nan
            ),
            "TyreTemp_C": (
                float(tyre_temp_c)
                if tyre_temp_c is not None
                else np.nan
            ),
        }

        # Only include features the model was trained on
        row_filtered = {
            k: v
            for k, v in row_data.items()
            if k in self._model_features
        }

        # Fill any model features not in row_data
        for feat in self._model_features:
            if feat not in row_filtered:
                if feat in CATEGORICAL_FEATURES:
                    row_filtered[feat] = "UNKNOWN"
                else:
                    row_filtered[feat] = np.nan

        row_df = pd.DataFrame([row_filtered])
        return row_df[self._model_features]


# ═══════════════════════════════════════════════════════════════
# 10. CLI ENTRY POINT
# ═══════════════════════════════════════════════════════════════

if __name__ == "__main__":
    print("F1 Virtual Race Strategist — Pace Model V3")
    print(f"   Input:    {DATA_PATH}")
    print(f"   Output:   {MODEL_PATH}")
    print(f"   Target:   {DELTA_TARGET}")
    print(f"   Baseline: {BASELINE_MODE}")
    print(f"   Device:   {TASK_TYPE}")
    print("   Features: strategy + projectable physics")
    print()

    pipeline_start = time.perf_counter()

    # ── Step 1: Load Data ─────────────────────────────────────
    df = load_data(DATA_PATH)
    n_laps_raw = len(df)

    # ── Step 2: Ensure Engineered Features ────────────────────
    df = ensure_engineered_features(df)

    # ── Step 3: Compute Baselines & Normalise ─────────────────
    (
        df,
        track_baselines,
        global_baseline,
        actual_mode,
        track_only_baselines,
    ) = compute_baselines(df, mode=BASELINE_MODE)

    # ── Step 4: Filter Outliers ───────────────────────────────
    df = filter_outlier_deltas(df)
    n_laps_filtered = len(df)

    # ── Step 5: Prepare Features ──────────────────────────────
    X, y, groups, cat_indices = prepare_features(df)

    # ── Step 6: Hyperparameter Tuning ─────────────────────────
    if OPTUNA_AVAILABLE:
        best_params = tune_hyperparameters(
            X, y, groups, cat_indices
        )
    else:
        best_params = DEFAULT_PARAMS.copy()

    # ── Step 7: Cross-Validation ──────────────────────────────
    fold_metrics = cross_validate(
        X,
        y,
        groups,
        cat_indices,
        df_full=df,
        track_baselines=track_baselines,
        global_baseline=global_baseline,
        track_only_baselines=track_only_baselines,
        baseline_mode=actual_mode,
        params=best_params,
    )

    # ── Step 8: Train Final Model ─────────────────────────────
    cv_best_iters = [
        m["best_iteration"]
        for m in fold_metrics
        if m.get("best_iteration") is not None
    ]
    model = train_final_model(
        X,
        y,
        cat_indices,
        params=best_params,
        cv_best_iterations=cv_best_iters,
    )

    # ── Step 9: Save Baselines ────────────────────────────────
    save_baselines(
        track_baselines,
        global_baseline,
        baseline_mode=actual_mode,
        track_only_baselines=track_only_baselines,
    )

    # ── Step 10: Feature Analysis ─────────────────────────────
    feat_imp = analyse_features(
        model,
        X,
        y,
        cat_indices,
        compute_interactions=True,
    )

    # ── Step 11: Prediction Tests ─────────────────────────────
    print(f"\n{'═' * 65}")
    print("  PREDICTION TESTS")
    print(f"{'═' * 65}")

    predictor = PacePredictor(
        str(MODEL_PATH), str(BASELINES_PATH)
    )

    sample = df.iloc[0]
    actual_abs = sample[RAW_TARGET]
    actual_delta = sample[DELTA_TARGET]
    track = sample["TrackName"]

    sample_season = (
        int(sample["Season"])
        if "Season" in sample.index
        else None
    )
    baseline = predictor._get_baseline(track, sample_season)

    sample_race = sample[GROUP_KEY]
    race_laps_series = cast(
        pd.Series,
        df.loc[df[GROUP_KEY] == sample_race, "RaceLapNumber"],
    )
    total_laps = int(race_laps_series.max())

    # ── Test 1: Strategy prediction ───────────────────────────
    strategy_pred = predictor.predict(
        compound=sample["Compound"],
        track_name=track,
        driver=sample["Driver"],
        team=sample["Team"],
        tire_age=int(sample["TireAge"]),
        stint=int(sample.get("StintNumber", sample.get("Stint", 1))),
        race_lap_number=int(sample["RaceLapNumber"]),
        total_race_laps=total_laps,
        tyre_health=(
            float(sample["TyreHealth"])
            if "TyreHealth" in sample.index
            and pd.notna(sample.get("TyreHealth"))
            else None
        ),
        tyre_temp_c=(
            float(sample["TyreTemp_C"])
            if "TyreTemp_C" in sample.index
            and pd.notna(sample.get("TyreTemp_C"))
            else None
        ),
        fuel_proxy=(
            float(sample["FuelProxy"])
            if "FuelProxy" in sample.index
            and pd.notna(sample.get("FuelProxy"))
            else None
        ),
        season=sample_season,
    )

    # ── Test 2: Uncertainty ───────────────────────────────────
    unc_pred, std_unc = predictor.predict_with_uncertainty(
        compound=sample["Compound"],
        track_name=track,
        driver=sample["Driver"],
        team=sample["Team"],
        tire_age=int(sample["TireAge"]),
        stint=int(sample.get("StintNumber", sample.get("Stint", 1))),
        race_lap_number=int(sample["RaceLapNumber"]),
        total_race_laps=total_laps,
        tyre_health=(
            float(sample["TyreHealth"])
            if "TyreHealth" in sample.index
            and pd.notna(sample.get("TyreHealth"))
            else None
        ),
        tyre_temp_c=(
            float(sample["TyreTemp_C"])
            if "TyreTemp_C" in sample.index
            and pd.notna(sample.get("TyreTemp_C"))
            else None
        ),
        fuel_proxy=(
            float(sample["FuelProxy"])
            if "FuelProxy" in sample.index
            and pd.notna(sample.get("FuelProxy"))
            else None
        ),
        season=sample_season,
    )

    pipeline_elapsed = time.perf_counter() - pipeline_start

    print(f"\n  Sample prediction:")
    print(
        f"    Driver:         {sample['Driver']} "
        f"({sample['Team']})"
    )
    print(f"    Track:          {track}")
    if sample_season:
        print(f"    Season:         {sample_season}")
    print(f"    Baseline:       {baseline:.2f}s")
    print(
        f"    Compound:       {sample['Compound']} "
        f"(Age: {int(sample['TireAge'])}, "
        f"Stint: {int(sample.get('StintNumber', sample.get('Stint', 1)))})"
    )
    print(
        f"    Race lap:       "
        f"{int(sample['RaceLapNumber'])}/{total_laps}"
    )

    for feat in PHYSICS_FEATURES:
        if (
            feat in sample.index
            and pd.notna(sample.get(feat))
        ):
            print(
                f"    {feat:16s}: {float(sample[feat]):.4f}"
            )

    print("    ---")
    print(f"    Actual abs:     {actual_abs:.3f}s")
    print(
        f"    Strategy pred:  {strategy_pred:.3f}s "
        f"(error: {strategy_pred - actual_abs:+.3f}s)"
    )
    print(f"    Uncertainty:    ±{std_unc:.3f}s")

    # ── Test 3: Strategy Simulation Demo ──────────────────────
    print(f"\n  Strategy Simulation Demo:")
    print("    Simulating 1-stop vs 2-stop from lap 1")

    result_1stop = predictor.simulate_strategy(
        driver=sample["Driver"],
        team=sample["Team"],
        track_name=track,
        current_lap=1,
        total_laps=total_laps,
        initial_compound="MEDIUM",
        pit_plan=[
            {
                "pit_lap": total_laps // 2,
                "compound": "HARD",
            }
        ],
        season=sample_season,
        pit_loss_sec=22.0,
    )

    result_2stop = predictor.simulate_strategy(
        driver=sample["Driver"],
        team=sample["Team"],
        track_name=track,
        current_lap=1,
        total_laps=total_laps,
        initial_compound="SOFT",
        pit_plan=[
            {
                "pit_lap": total_laps // 3,
                "compound": "MEDIUM",
            },
            {
                "pit_lap": 2 * total_laps // 3,
                "compound": "HARD",
            },
        ],
        season=sample_season,
        pit_loss_sec=22.0,
    )

    print(
        f"    1-stop ({result_1stop['compounds_used']}): "
        f"{result_1stop['total_time']:.1f}s total "
        f"({result_1stop['pit_stops']} pit × 22s)"
    )
    print(
        f"    2-stop ({result_2stop['compounds_used']}): "
        f"{result_2stop['total_time']:.1f}s total "
        f"({result_2stop['pit_stops']} pits × 22s)"
    )
    diff = (
        result_1stop["total_time"]
        - result_2stop["total_time"]
    )
    winner = "2-stop" if diff > 0 else "1-stop"
    print(f"    Δ = {abs(diff):.1f}s → {winner} preferred")

    # ── Step 12: Save Report ──────────────────────────────────
    save_report(
        fold_metrics=fold_metrics,
        feat_imp=feat_imp,
        params=best_params,
        track_baselines=track_baselines,
        baseline_mode=actual_mode,
        n_laps_raw=n_laps_raw,
        n_laps_filtered=n_laps_filtered,
        n_races=df[GROUP_KEY].nunique(),
        total_time=pipeline_elapsed,
    )

    # ── Summary ───────────────────────────────────────────────
    avg_r2 = np.mean([m["delta_r2"] for m in fold_metrics])
    avg_rmse = np.mean(
        [m["delta_rmse"] for m in fold_metrics]
    )
    avg_best_iter = np.mean(
        [m["best_iteration"] for m in fold_metrics]
    )

    print(f"\n{'═' * 65}")
    print("  PIPELINE COMPLETE")
    print(f"{'═' * 65}")
    print(
        f"  Total:       {pipeline_elapsed:.1f}s "
        f"({pipeline_elapsed / 60:.1f} min)"
    )
    print(f"  GPU:         {'Yes' if GPU_AVAILABLE else 'No'}")
    print(f"  Baseline:    {actual_mode}")
    print(
        f"  Laps:        {n_laps_raw:,} raw → "
        f"{n_laps_filtered:,} filtered"
    )
    print(f"  Delta RMSE:  {avg_rmse:.4f}s")
    print(f"  Delta R²:    {avg_r2:.4f}")
    print(f"  Best iter:   {avg_best_iter:.0f}")

    # ── Health Assessment ─────────────────────────────────────
    print(f"\n  Health Assessment:")
    issues: list[str] = []

    if avg_r2 < 0:
        issues.append("Negative R²")
    elif avg_r2 < 0.15:
        issues.append(f"Low R² ({avg_r2:.4f})")
    else:
        print(f"R² = {avg_r2:.4f}")

    if avg_best_iter < 50:
        issues.append(
            f"Low best_iter ({avg_best_iter:.0f})"
        )
    else:
        print(f"Avg best_iter = {avg_best_iter:.0f}")

    if avg_rmse > 1.5:
        issues.append(f"High RMSE ({avg_rmse:.4f}s)")
    else:
        print(f"RMSE = {avg_rmse:.4f}s")

    # Per-physics-feature check
    physics_total_pct = 0.0
    for feat in PHYSICS_FEATURES:
        row = feat_imp[feat_imp["Feature"] == feat]
        if not row.empty:
            pct = row["Importance_Pct"].iloc[0]
            physics_total_pct += pct
            if pct < 0.1:
                issues.append(
                    f"{feat} at {pct:.1f}% — "
                    f"check within-stint variance"
                )

    if physics_total_pct < 1.0 and any(
        f in feat_imp["Feature"].values
        for f in PHYSICS_FEATURES
    ):
        issues.append(
            f"Physics tier total: {physics_total_pct:.1f}%"
        )
    elif physics_total_pct >= 1.0:
        print(
            f"Physics tier: {physics_total_pct:.1f}%"
        )

    # Overfit gap
    train_pool = Pool(X, y, cat_features=cat_indices)
    train_preds = model.predict(train_pool)
    train_rmse = np.sqrt(mean_squared_error(y, train_preds))
    overfit_ratio = train_rmse / avg_rmse if avg_rmse > 0 else 0
    if overfit_ratio < 0.7:
        issues.append(
            f"Overfit gap: train {train_rmse:.4f}s "
            f"vs CV {avg_rmse:.4f}s "
            f"(ratio {overfit_ratio:.2f})"
        )
    else:
        print(
            f"Overfit ratio: {overfit_ratio:.2f} "
            f"(train {train_rmse:.4f}s / "
            f"CV {avg_rmse:.4f}s)"
        )

    if issues:
        for issue in issues:
            print(f"    {issue}")
        print(f"\n Suggestions:")
        if any("Physics" in i or "TyreHealth" in i or "FuelProxy" in i or "TyreTemp" in i for i in issues):
            print(
                "     - Check physics feature distributions "
                "with: df[PHYSICS_FEATURES].describe()"
            )
            print(
                "     - Verify TyreHealth varies within "
                "stints (not just across stints)"
            )
        if any("Overfit" in i for i in issues):
            print(
                "     - Increase l2_leaf_reg or "
                "min_data_in_leaf"
            )
            print(
                "     - Reduce depth in tuning search space"
            )
        if any("R²" in i for i in issues):
            print(
                "     - Review outlier filtering thresholds"
            )
            print(
                "     - Check target distribution: "
                "df[DELTA_TARGET].hist()"
            )
    else:
        print(f"\n All health checks passed.")

    print(f"\n  Prediction modes:")
    print("    Strategy:    predictor.predict(...)")
    print("    Batch:       predictor.predict_batch(df)")
    print("    Uncertainty: predictor.predict_with_uncertainty(...)")
    print("    Simulate:    predictor.simulate_strategy(...)")

    print(
        f"\n V3 model training complete. "
        f"Ready for strategy engine."
    )