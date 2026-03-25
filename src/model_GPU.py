"""
pace_model.py
=============
Train a CatBoost Regressor to predict F1 lap-time DELTAS
(deviation from per-track baseline) using features extracted
by data_pipeline.py.

The delta-target design forces the model to learn within-race
dynamics (tire degradation, fuel burn, weather, traffic) instead
of simply memorising which circuit is which.

Key design decisions:
  - Per-track baseline normalisation (removes inter-track variance)
  - GroupKFold by RaceID (no data leakage)
  - Native categorical handling (no manual encoding)
  - Optuna hyperparameter tuning (pruned + warm-started)
  - Virtual ensembles for uncertainty estimation
  - GPU acceleration where supported (auto-detected)

Usage:
    python pace_model.py

Input:
    training_data.parquet — from data_pipeline.py

Output:
    models/pace_model.cbm           — trained CatBoost model
    models/track_baselines.json     — per-track median lap times
    models/training_report.txt      — feature importance + metrics
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
        "⚠️  Optuna not installed. Hyperparameter tuning "
        "will be skipped. Install with: pip install optuna"
    )


# ═══════════════════════════════════════════════════════════════
# 1. GPU DETECTION
# ═══════════════════════════════════════════════════════════════

def detect_gpu() -> bool:
    """Detect CUDA GPU by attempting a tiny CatBoost fit."""
    print("🔍 Detecting GPU availability…")

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
            print("   ❌ No NVIDIA GPU detected")
            return False
    except (FileNotFoundError, subprocess.TimeoutExpired):
        print("   ❌ nvidia-smi not found")
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
        print("   ✅ GPU training verified")
        return True
    except Exception as e:
        print(f"   ⚠️  GPU fit failed: {e}")
        return False


# ═══════════════════════════════════════════════════════════════
# 2. CONFIGURATION
# ═══════════════════════════════════════════════════════════════

# ── Paths ─────────────────────────────────────────────────────
DATA_PATH = "training_data.parquet"
MODEL_DIR = pathlib.Path("models")
MODEL_DIR.mkdir(exist_ok=True)
MODEL_PATH = MODEL_DIR / "pace_model.cbm"
BASELINES_PATH = MODEL_DIR / "track_baselines.json"
REPORT_PATH = MODEL_DIR / "training_report.txt"

# ── Target ────────────────────────────────────────────────────
RAW_TARGET = "LapTimeSec"
# The model predicts the DELTA from a per-track baseline.
# This removes inter-track variance (~9 s std) and forces
# the model to learn within-race dynamics (~1–3 s std).
DELTA_TARGET = "LapTimeDelta"

# ── Features ──────────────────────────────────────────────────
CATEGORICAL_FEATURES = [
    "Compound",
    "TrackName",
    "Driver",
    "Team",
]

# CircuitLength and NumberOfCorners are REMOVED.
# They are constant per circuit and perfectly redundant with
# TrackName (which CatBoost handles natively).  With a raw
# lap-time target they dominated importance at 87 % — the
# model used them as a numeric shortcut to identify the
# circuit instead of learning tyre degradation, fuel burn,
# and weather effects.
NUMERICAL_FEATURES = [
    "TireAge",
    "TireAgeSq",
    "RaceLapNumber",
    "TrackTemp",
    "AirTemp",
    "Humidity",
    "WindSpeed",
    "GapToCarAhead",
    "DRS_Available",
]

ALL_FEATURES = CATEGORICAL_FEATURES + NUMERICAL_FEATURES

GROUP_KEY = "RaceID"

# ── GPU ───────────────────────────────────────────────────────
GPU_AVAILABLE = detect_gpu()
TASK_TYPE = "GPU" if GPU_AVAILABLE else "CPU"

# ── CatBoost Defaults ────────────────────────────────────────
# With the delta target the learning task is harder (subtler
# signal), so we allow more iterations than the raw-target
# version.  Virtual-ensemble params (posterior_sampling,
# langevin) are CPU-only and applied only to the final model.
DEFAULT_PARAMS: dict[str, Any] = {
    "iterations": 2500,
    "learning_rate": 0.05,
    "depth": 8,
    "loss_function": "RMSE",
    "verbose": 200,
    "random_seed": 42,
    "posterior_sampling": True,
    "langevin": True,
}

GPU_OVERRIDES: dict[str, Any] = {"border_count": 128}

# ── Tuning Configuration ─────────────────────────────────────
# Slightly larger budget than the raw-target version because
# the delta-prediction task has a subtler signal and benefits
# from more exploration.
TUNING_CONFIG: dict[str, Any] = {
    "n_trials": 25,
    "n_splits": 2,
    "subsample_frac": 0.6,
    "max_iterations": 1500,
    "early_stopping_rounds": 50,
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
    print(f"\n Loading data from: {path}")
    df = (
        pd.read_parquet(path)
        if path.endswith(".parquet")
        else pd.read_csv(path)
    )
    print(f"   Rows: {len(df):,}")

    required = ALL_FEATURES + [RAW_TARGET, GROUP_KEY]
    missing = [c for c in required if c not in df.columns]
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

    return df


def compute_baselines(
    df: pd.DataFrame,
) -> tuple[pd.DataFrame, dict[str, float], float]:
    """
    Compute per-track baseline lap times and normalise the
    target to a delta.

    The baseline is the per-TrackName MEDIAN lap time across
    all races in the training set.  Median is preferred over
    mean because it is robust to outlier laps (safety cars,
    in-laps, out-laps that survived filtering, etc.).

    This is NOT data leakage:
      - The per-track median is a property of the circuit
        geometry (like circuit length), not of specific
        race outcomes.
      - It is very stable (std across races at the same
        track is typically < 1 s for dry conditions).
      - GroupKFold holds out entire RACES, not tracks.
        The model cannot exploit the baseline to predict
        individual lap outcomes.

    Parameters
    ----------
    df : pd.DataFrame
        Training data with RAW_TARGET and "TrackName".

    Returns
    -------
    df : pd.DataFrame
        Input DataFrame with DELTA_TARGET column added.
    track_baselines : dict
        {track_name: median_lap_time} mapping.
    global_baseline : float
        Global median (fallback for unseen tracks).
    """
    print(f"\n Computing per-track baselines…")

    track_baselines: dict[str, float] = (
        df.groupby("TrackName")[RAW_TARGET]
        .median()
        .to_dict()
    )
    global_baseline = float(df[RAW_TARGET].median())

    # Normalise target
    df = df.copy()
    df["_TrackBaseline"] = df["TrackName"].map(track_baselines)
    df[DELTA_TARGET] = df[RAW_TARGET] - df["_TrackBaseline"]

    print(f"   Tracks: {len(track_baselines)}")
    print(f"   Global median: {global_baseline:.2f}s")
    print(f"\n   Delta target ({DELTA_TARGET}):")
    print(f"     Mean:  {df[DELTA_TARGET].mean():.2f}s  "
          f"Std: {df[DELTA_TARGET].std():.2f}s")
    print(f"     Range: {df[DELTA_TARGET].min():.2f}s – "
          f"{df[DELTA_TARGET].max():.2f}s")

    # Show per-track baselines
    print(f"\n   Per-track baselines:")
    for track in sorted(track_baselines):
        n_laps = (df["TrackName"] == track).sum()
        print(
            f"     {track:25s} {track_baselines[track]:7.2f}s "
            f"({n_laps:,} laps)"
        )

    return df, track_baselines, global_baseline


def prepare_features(df: pd.DataFrame):
    """Extract X, y (delta), groups, and cat indices."""
    X = df[ALL_FEATURES].copy()
    y = df[DELTA_TARGET].copy()
    groups = df[GROUP_KEY].copy()
    cat_indices = [
        ALL_FEATURES.index(c) for c in CATEGORICAL_FEATURES
    ]

    print(f"\n Feature matrix: {X.shape}")
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
    params: Optional[dict] = None,
    n_splits: Optional[int] = None,
) -> list:
    """
    GroupKFold cross-validation.

    Reports both delta RMSE (what the model optimises) and
    absolute RMSE (for comparison with the raw-target version).

    Parameters
    ----------
    df_full : pd.DataFrame
        Full DataFrame with RAW_TARGET for absolute-error
        reporting.
    track_baselines : dict
        Per-track medians for converting deltas → absolutes.
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
        "GPU "
        if cv_params.get("task_type") == "GPU"
        else "CPU"
    )

    print(f"\n{'═' * 60}")
    print(f"  CROSS-VALIDATION ({n_splits}-Fold GroupKFold)")
    print(f"  Device: {device_label}")
    print(f"  Target: {DELTA_TARGET}")
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

        delta_preds = model.predict(test_pool)

        # ── Delta metrics (what the model optimises) ──────────
        delta_rmse = np.sqrt(
            mean_squared_error(y_test, delta_preds)
        )
        delta_mae = mean_absolute_error(y_test, delta_preds)
        delta_r2 = r2_score(y_test, delta_preds)

        # ── Absolute metrics (for comparison) ─────────────────
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
        print(f"    {fold_elapsed:.1f}s | "
              f"best_iter: {model.get_best_iteration()}")

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
            "best_iteration": model.get_best_iteration(),
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

    print(f"\n{'═' * 60}")
    print(f"  CV RESULTS ({device_label})")
    print(f"{'═' * 60}")
    print(f"  Delta RMSE: {avg['delta_rmse']:.4f}s "
          f"± {std_rmse:.4f}s")
    print(f"  Delta MAE:  {avg['delta_mae']:.4f}s")
    print(f"  Delta R²:   {avg['delta_r2']:.4f}")
    print(f"  Abs   RMSE: {avg['abs_rmse']:.4f}s "
          f"(cf. raw-target model)")
    print(f"  CV time:    {cv_elapsed:.1f}s")
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

    The search space is intentionally compact (4 tuned params).
    With the delta target the signal is subtler, so we allow
    a wider iteration range (500–1500) and slightly more trials
    than the raw-target version.
    """
    if not OPTUNA_AVAILABLE:
        print("  Optuna not available. Using defaults.")
        return DEFAULT_PARAMS.copy()

    if n_trials is None:
        n_trials = TUNING_CONFIG["n_trials"]
    if n_splits is None:
        n_splits = TUNING_CONFIG["n_splits"]
    if subsample_frac is None:
        subsample_frac = TUNING_CONFIG["subsample_frac"]

    max_iters = TUNING_CONFIG["max_iterations"]
    es_rounds = TUNING_CONFIG["early_stopping_rounds"]
    device_label = "GPU " if GPU_AVAILABLE else "CPU"
    max_fits = n_trials * n_splits

    print(f"\n{'═' * 60}")
    print(f"  HYPERPARAMETER TUNING ({n_trials} trials)")
    print(f"{'═' * 60}")
    print(f"  Device:          {device_label}")
    print(f"  Max fits:        {max_fits}")
    print(f"  Train subsample: {subsample_frac:.0%}")
    print(f"  Iter cap:        {max_iters}")
    print(f"  Target:          {DELTA_TARGET}")
    print(f"{'═' * 60}")

    gkf = GroupKFold(n_splits=n_splits)
    fold_indices = list(gkf.split(X, y, groups=groups))
    tuning_start = time.perf_counter()

    def objective(trial: "optuna.Trial") -> float:
        params: dict[str, Any] = {
            "iterations": trial.suggest_int(
                "iterations", 500, max_iters,
            ),
            "learning_rate": trial.suggest_float(
                "learning_rate", 0.02, 0.12, log=True,
            ),
            "depth": trial.suggest_int("depth", 5, 9),
            "l2_leaf_reg": trial.suggest_float(
                "l2_leaf_reg", 1.0, 20.0, log=True,
            ),
            "bagging_temperature": 1.0,
            "random_strength": 1.0,
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

    # Warm-start with defaults
    study.enqueue_trial({
        "iterations": min(
            DEFAULT_PARAMS["iterations"], max_iters
        ),
        "learning_rate": DEFAULT_PARAMS["learning_rate"],
        "depth": DEFAULT_PARAMS["depth"],
        "l2_leaf_reg": 3.0,
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
    print(f"\n   Best RMSE:   {study.best_value:.4f}s")
    print(f"   Time:        {tuning_elapsed:.1f}s "
          f"({tuning_elapsed / 60:.1f} min)")
    print(f"   Trials:       {n_complete} complete, "
          f"{n_pruned} pruned")
    print(f"   Fold fits:    {actual_fits} "
          f"(vs {max_fits} max)")
    print(f"   Best parameters:")
    for k, v in sorted(best.items()):
        print(f"    {k}: {v}")

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
    """
    if params is None:
        params = DEFAULT_PARAMS.copy()

    final_params = {**params}
    final_params["posterior_sampling"] = True
    final_params["langevin"] = True
    final_params["task_type"] = "CPU"
    final_params.pop("devices", None)

    # ── Adaptive iteration count ──────────────────────────────
    # Use CV best_iteration as a guide.  Add 15% margin because
    # the full dataset (all races) may support more iterations
    # than the ~67% training folds in 3-fold CV.
    if cv_best_iterations and all(
        b is not None and b > 0 for b in cv_best_iterations
    ):
        adaptive_iters = int(
            np.median(cv_best_iterations) * 1.15
        )
        adaptive_iters = max(500, min(adaptive_iters, 4000))
        original_iters = final_params.get("iterations", 2500)

        if adaptive_iters < original_iters:
            print(
                f"\n   Adaptive iterations: "
                f"{original_iters} → {adaptive_iters} "
                f"(CV best: {cv_best_iterations})"
            )
            final_params["iterations"] = adaptive_iters
        else:
            print(
                f"\n  ℹ  Keeping {original_iters} iterations "
                f"(CV best: {cv_best_iterations}, "
                f"×1.15 = {adaptive_iters})"
            )

    print(f"\n{'═' * 60}")
    print(f"  TRAINING FINAL MODEL")
    print(f"{'═' * 60}")
    print(f"  Training on {len(X):,} laps (all data)")
    print(f"  Target: {DELTA_TARGET}")
    print(f"  Device: CPU (required for virtual ensembles)")
    if GPU_AVAILABLE:
        print(f"   GPU was used for CV & tuning")
    print(f"  Parameters:")
    for k, v in sorted(final_params.items()):
        if k != "verbose":
            print(f"    {k}: {v}")

    pool = Pool(X, y, cat_features=cat_indices)

    train_start = time.perf_counter()
    model = CatBoostRegressor(**final_params)
    model.fit(pool)
    train_elapsed = time.perf_counter() - train_start

    model.save_model(str(MODEL_PATH))
    print(f"\n  Model saved to: {MODEL_PATH}")
    print(f"   Training time: {train_elapsed:.1f}s "
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

    With the delta target, we expect the ranking to shift
    dramatically from the raw-target version:
      BEFORE (raw target):  CircuitLength 44%, NumberOfCorners 43%
      AFTER  (delta target): TireAge, Compound, RaceLapNumber, …

    Parameters
    ----------
    compute_interactions : bool
        If True, compute pairwise interaction strengths.
        Adds ~3 min on T4. Default False.
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
                    "RaceLapNumber"}
    found = top_5 & expected_top

    if len(found) >= 2:
        print(
            f"     Race dynamics in top 5: {found}"
        )
    else:
        print(
            f"      Expected race dynamics "
            f"(TireAge/Compound/RaceLapNumber) not "
            f"dominant. Top 5: {top_5}"
        )

    # Check that no single feature exceeds 40%
    # (sign of a shortcut / degenerate learning)
    max_pct = feat_imp["Importance_Pct"].iloc[0]
    max_feat = feat_imp["Feature"].iloc[0]
    if max_pct > 40:
        print(
            f"      {max_feat} dominates at "
            f"{max_pct:.1f}% — investigate possible "
            f"data leakage or shortcut"
        )
    else:
        print(
            f"     Importance well-distributed "
            f"(max: {max_feat} at {max_pct:.1f}%)"
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
                    "     Compound × TireAge detected"
                    if not compound_tire.empty
                    else "      Compound × TireAge "
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
            f"\n    Interaction analysis skipped "
            f"(compute_interactions=True to enable)"
        )

    return feat_imp


# ═══════════════════════════════════════════════════════════════
# 8. SAVE TRAINING REPORT
# ═══════════════════════════════════════════════════════════════

def save_report(
    fold_metrics: list,
    feat_imp: pd.DataFrame,
    params: dict,
    track_baselines: dict[str, float],
    n_laps: int,
    n_races: int,
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
    total_cv_time = sum(
        m.get("elapsed_sec", 0) for m in fold_metrics
    )

    lines = [
        "F1 VIRTUAL RACE STRATEGIST — TRAINING REPORT",
        "=" * 55,
        "",
        f"Dataset: {n_laps:,} laps from {n_races} races",
        f"Features: {len(ALL_FEATURES)} "
        f"({len(CATEGORICAL_FEATURES)} cat, "
        f"{len(NUMERICAL_FEATURES)} num)",
        f"Model: CatBoost Regressor",
        f"Target: {DELTA_TARGET} (delta from per-track median)",
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
            f"({m['n_test_laps']} laps, "
            f"{len(m['test_races'])} races{elapsed_str})"
        )

    lines.extend(["", "HYPERPARAMETERS", "-" * 55])
    for k, v in sorted(params.items()):
        lines.append(f"  {k}: {v}")

    lines.extend(["", "TUNING CONFIG", "-" * 55])
    for k, v in sorted(TUNING_CONFIG.items()):
        lines.append(f"  {k}: {v}")

    lines.extend(["", "FEATURE IMPORTANCE", "-" * 55])
    for _, row in feat_imp.iterrows():
        lines.append(
            f"  {row['Feature']:20s} "
            f"{row['Importance_Pct']:5.1f}%"
        )

    lines.extend(
        ["", "PER-TRACK BASELINES (median lap time)", "-" * 55]
    )
    for track in sorted(track_baselines):
        lines.append(
            f"  {track:25s} {track_baselines[track]:.2f}s"
        )

    REPORT_PATH.write_text(
        "\n".join(lines), encoding="utf-8"
    )
    print(f"\n   Report saved to: {REPORT_PATH}")


def save_baselines(
    track_baselines: dict[str, float],
    global_baseline: float,
):
    """
    Save track baselines to JSON for use by the prediction
    interface and the strategy engine.

    The prediction interface needs these to convert delta
    predictions back to absolute lap times.
    """
    payload = {
        "track_baselines": track_baselines,
        "global_baseline": global_baseline,
        "description": (
            "Per-track median lap times (seconds). "
            "Add to model delta prediction to get "
            "absolute lap time."
        ),
    }
    BASELINES_PATH.write_text(
        json.dumps(payload, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    print(f"   Baselines saved to: {BASELINES_PATH}")


# ═══════════════════════════════════════════════════════════════
# 9. PREDICTION INTERFACE
# ═══════════════════════════════════════════════════════════════

class PacePredictor:
    """
    Wrapper around the trained CatBoost model for the
    Strategy Engine.

    The model predicts DELTA from a per-track baseline.
    This class handles the conversion back to absolute
    lap times transparently.

    Provides:
      - Point predictions (absolute lap time)
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

        self._cat_indices = [
            ALL_FEATURES.index(c)
            for c in CATEGORICAL_FEATURES
        ]

        print(
            f"  PacePredictor loaded "
            f"({len(self._track_baselines)} tracks)"
        )

    def _get_baseline(self, track_name: str) -> float:
        """
        Look up baseline for a track.  Falls back to the
        global median for unseen tracks (e.g. new circuits).
        """
        return self._track_baselines.get(
            track_name, self._global_baseline
        )

    def predict(
        self,
        compound: str,
        track_name: str,
        driver: str,
        team: str,
        tire_age: int,
        tire_age_sq: Optional[int] = None,
        race_lap_number: int = 1,
        track_temp: float = 35.0,
        air_temp: float = 28.0,
        humidity: float = 50.0,
        wind_speed: float = 2.0,
        gap_to_car_ahead: float = 5.0,
        drs_available: int = 0,
    ) -> float:
        """
        Predict ABSOLUTE lap time for a single set of
        conditions.

        Internally predicts delta, then adds track baseline.

        Returns
        -------
        float
            Predicted lap time in seconds.
        """
        row = self._build_row(
            compound, track_name, driver, team,
            tire_age, tire_age_sq, race_lap_number,
            track_temp, air_temp, humidity, wind_speed,
            gap_to_car_ahead, drs_available,
        )
        pool = Pool(row, cat_features=self._cat_indices)
        delta = float(self.model.predict(pool)[0])
        baseline = self._get_baseline(track_name)
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
        track_temp: float = 35.0,
        air_temp: float = 28.0,
        humidity: float = 50.0,
        wind_speed: float = 2.0,
        gap_to_car_ahead: float = 5.0,
        drs_available: int = 0,
        n_ensembles: int = 10,
    ) -> tuple[float, float]:
        """
        Predict ABSOLUTE lap time with uncertainty.

        Returns (mean_prediction, std_uncertainty) in seconds.
        Uncertainty is on the delta; baseline is deterministic.
        """
        row = self._build_row(
            compound, track_name, driver, team,
            tire_age, tire_age_sq, race_lap_number,
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
        baseline = self._get_baseline(track_name)

        return float(baseline + delta_mean), float(std_dev)

    def predict_batch(
        self, df: pd.DataFrame
    ) -> np.ndarray:
        """
        Predict ABSOLUTE lap times for a DataFrame.

        The DataFrame must contain all ALL_FEATURES columns
        (including TrackName for baseline lookup).

        Returns
        -------
        np.ndarray
            Absolute predicted lap times.
        """
        df = df.copy()
        for col in CATEGORICAL_FEATURES:
            df[col] = df[col].astype(str)

        pool = Pool(
            df[ALL_FEATURES],
            cat_features=self._cat_indices,
        )
        deltas = self.model.predict(pool)

        baselines = df["TrackName"].map(
            lambda t: self._get_baseline(t)
        ).values

        return baselines + deltas

    def _build_row(
        self,
        compound, track_name, driver, team,
        tire_age, tire_age_sq, race_lap_number,
        track_temp, air_temp, humidity, wind_speed,
        gap_to_car_ahead, drs_available,
    ) -> pd.DataFrame:
        """Build a single-row DataFrame matching ALL_FEATURES."""
        if tire_age_sq is None:
            tire_age_sq = tire_age ** 2

        return pd.DataFrame([{
            "Compound":        str(compound).upper(),
            "TrackName":       str(track_name),
            "Driver":          str(driver),
            "Team":            str(team),
            "TireAge":         int(tire_age),
            "TireAgeSq":       int(tire_age_sq),
            "RaceLapNumber":   int(race_lap_number),
            "TrackTemp":       float(track_temp),
            "AirTemp":         float(air_temp),
            "Humidity":        float(humidity),
            "WindSpeed":       float(wind_speed),
            "GapToCarAhead":   float(gap_to_car_ahead),
            "DRS_Available":   int(drs_available),
        }])


# ═══════════════════════════════════════════════════════════════
# 10. CLI ENTRY POINT
# ═══════════════════════════════════════════════════════════════

if __name__ == "__main__":
    print("🏎️  F1 Virtual Race Strategist — Pace Model")
    print(f"   Input:  {DATA_PATH}")
    print(f"   Output: {MODEL_PATH}")
    print(f"   Target: {DELTA_TARGET} (delta from track median)")
    print(f"   Device: {TASK_TYPE}")
    print()

    pipeline_start = time.perf_counter()

    # ── Step 1: Load Data ─────────────────────────────────────
    df = load_data(DATA_PATH)

    # ── Step 2: Compute Baselines & Normalise ─────────────────
    df, track_baselines, global_baseline = compute_baselines(df)
    X, y, groups, cat_indices = prepare_features(df)

    # ── Step 3: Hyperparameter Tuning ─────────────────────────
    if OPTUNA_AVAILABLE:
        best_params = tune_hyperparameters(
            X, y, groups, cat_indices,
        )
    else:
        best_params = DEFAULT_PARAMS.copy()

    # ── Step 4: Cross-Validation ──────────────────────────────
    fold_metrics = cross_validate(
        X, y, groups, cat_indices,
        df_full=df,
        track_baselines=track_baselines,
        params=best_params,
    )

    # ── Step 5: Train Final Model ─────────────────────────────
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

    # ── Step 6: Save Baselines ────────────────────────────────
    save_baselines(track_baselines, global_baseline)

    # ── Step 7: Feature Analysis ──────────────────────────────
    feat_imp = analyse_features(
        model, X, y, cat_indices,
        compute_interactions=False,
    )

    # ── Step 8: Prediction Test ───────────────────────────────
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
    baseline = track_baselines[track]

    predicted_abs = predictor.predict(
        compound=sample["Compound"],
        track_name=track,
        driver=sample["Driver"],
        team=sample["Team"],
        tire_age=int(sample["TireAge"]),
        race_lap_number=int(sample["RaceLapNumber"]),
        track_temp=float(sample["TrackTemp"]),
        air_temp=float(sample["AirTemp"]),
        humidity=float(sample["Humidity"]),
        wind_speed=float(sample["WindSpeed"]),
        gap_to_car_ahead=float(sample["GapToCarAhead"]),
        drs_available=int(sample["DRS_Available"]),
    )

    predicted_unc, std_unc = predictor.predict_with_uncertainty(
        compound=sample["Compound"],
        track_name=track,
        driver=sample["Driver"],
        team=sample["Team"],
        tire_age=int(sample["TireAge"]),
        race_lap_number=int(sample["RaceLapNumber"]),
        track_temp=float(sample["TrackTemp"]),
        air_temp=float(sample["AirTemp"]),
        humidity=float(sample["Humidity"]),
        wind_speed=float(sample["WindSpeed"]),
        gap_to_car_ahead=float(sample["GapToCarAhead"]),
        drs_available=int(sample["DRS_Available"]),
    )

    predicted_delta = predictor.predict_delta(
        compound=sample["Compound"],
        track_name=track,
        driver=sample["Driver"],
        team=sample["Team"],
        tire_age=int(sample["TireAge"]),
        race_lap_number=int(sample["RaceLapNumber"]),
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
    print(f"    Track baseline: {baseline:.2f}s")
    print(f"    Compound:       {sample['Compound']} "
          f"(Age: {int(sample['TireAge'])})")
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

    # ── Step 9: Save Report ───────────────────────────────────
    save_report(
        fold_metrics=fold_metrics,
        feat_imp=feat_imp,
        params=best_params,
        track_baselines=track_baselines,
        n_laps=len(df),
        n_races=df[GROUP_KEY].nunique(),
        total_time=pipeline_elapsed,
    )

    print(f"\n{'═' * 60}")
    print(f"  PIPELINE COMPLETE")
    print(f"{'═' * 60}")
    print(f"  Total: {pipeline_elapsed:.1f}s "
          f"({pipeline_elapsed / 60:.1f} min)")
    print(f"  GPU:   {'Yes' if GPU_AVAILABLE else 'No'}")
    print(f"  Model predicts: delta from track baseline")
    print(f"  PacePredictor converts back to absolute times")
    print(
        f"\n Model training complete. "
        f"Ready for strategy engine."
    )