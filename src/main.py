"""
pace_model_no_physics.py
========================
Configurable CatBoost pace-model runner without physics features.

Edit MODEL_RUNS below to define one or more experiments. Each model writes to
its own folder under:

    models/<model_name>/

Each run reuses the training, evaluation, reporting, and prediction pipeline
from model_GPU_physics.py, but only with the feature set you specify.

Usage:
    python model_no_physics.py

Input defaults:
    data/processed/training_data.parquet

Outputs per model:
    models/<model_name>/pace_model.cbm
    models/<model_name>/track_baselines.json
    models/<model_name>/training_report.txt
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

BASE_DIR = Path(__file__).resolve().parents[1]
SRC_DIR = BASE_DIR / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

import model_GPU_physics as base


# ═══════════════════════════════════════════════════════════════
# 1. MODEL CONFIGURATION
# ═══════════════════════════════════════════════════════════════

MODEL_RUNS: list[dict[str, Any]] = [
    {
        "model_name": "baseline_no_physics",
        "data_path": BASE_DIR / "data" / "processed" / "training_data_with_physics_shifted.parquet",
        "categorical_features": [
            "Compound",
            "TrackName",
            "Driver",
            "Team",
        ],
        "numerical_features": [
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
        ],
    },

    {"model_name": "no_driver_team_info",
     "data_path": BASE_DIR / "data" / "processed" / "training_data_with_physics_shifted.parquet",
     "categorical_features": [
         "Compound",
         "TrackName",
     ],
     "numerical_features": [
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
         "DamageState",
        "TyreTemp_C",
        "SlidingProxy", # NOTE: removed F_zN for now.
     ],
    },
]


# ═══════════════════════════════════════════════════════════════
# 2. CONFIGURATION HELPERS
# ═══════════════════════════════════════════════════════════════

def configure_model(config: dict[str, Any]) -> None:
    """Apply a model config to the shared training pipeline."""
    model_name = str(config["model_name"])
    data_path = Path(config.get("data_path", BASE_DIR / "data" / "processed" / "training_data.parquet"))
    categorical_features = list(config["categorical_features"])
    numerical_features = list(config["numerical_features"])

    output_dir = BASE_DIR / "models" / model_name
    output_dir.mkdir(parents=True, exist_ok=True)

    base.DATA_PATH = str(data_path)
    base.MODEL_DIR = output_dir
    base.MODEL_PATH = output_dir / "pace_model.cbm"
    base.BASELINES_PATH = output_dir / "track_baselines.json"
    base.REPORT_PATH = output_dir / "training_report.txt"

    base.CATEGORICAL_FEATURES = categorical_features
    base.NUMERICAL_FEATURES = numerical_features
    base.ALL_FEATURES = categorical_features + numerical_features


def run_single_model(config: dict[str, Any]) -> dict[str, Any]:
    """Train, evaluate, and save one configured model."""
    configure_model(config)

    model_name = str(config["model_name"])
    print(f"  F1 Virtual Race Strategist - Pace Model ({model_name})")
    print(f"   Input:  {base.DATA_PATH}")
    print(f"   Output: {base.MODEL_PATH}")
    print(f"   Target: {base.DELTA_TARGET} (delta from track median)")
    print(f"   Baseline mode: {base.BASELINE_MODE}")
    print(f"   Device: {base.TASK_TYPE}")
    print(f"   Features: {len(base.ALL_FEATURES)} total")
    print()

    

    try:
        # Step 1: Load data
        df = base.load_data(base.DATA_PATH)
        n_laps_raw = len(df)
      

        # Step 2: Engineer features
        df = base.engineer_features(df)
    

        # Step 3: Compute baselines and normalise
        df, track_baselines, global_baseline, actual_mode = base.compute_baselines(
            df,
            mode=base.BASELINE_MODE,
        )


        # Step 4: Filter outliers
        df = base.filter_outlier_deltas(df)
        n_laps_filtered = len(df)


        # Step 5: Prepare features
        X, y, groups, cat_indices = base.prepare_features(df)


        # Step 6: Hyperparameter tuning
        if base.OPTUNA_AVAILABLE:
            best_params = base.tune_hyperparameters(
                X,
                y,
                groups,
                cat_indices,
            )
        else:
            best_params = base.DEFAULT_PARAMS.copy()


        # Step 7: Cross-validation
        fold_metrics = base.cross_validate(
            X,
            y,
            groups,
            cat_indices,
            df_full=df,
            track_baselines=track_baselines,
            baseline_mode=actual_mode,
            params=best_params,
        )


        # Step 8: TrackName ablation
        ablation_results = base.run_trackname_ablation(
            X,
            y,
            groups,
            cat_indices,
            best_params,
        )


        # Step 9: Train final model
        cv_best_iters = [
            m["best_iteration"]
            for m in fold_metrics
            if m.get("best_iteration") is not None
        ]
        model = base.train_final_model(
            X,
            y,
            cat_indices,
            params=best_params,
            cv_best_iterations=cv_best_iters,
        )


        # Step 10: Save baselines
        track_only_baselines = df.attrs.get("track_only_baselines", None)
        base.save_baselines(
            track_baselines,
            global_baseline,
            baseline_mode=actual_mode,
            track_only_baselines=track_only_baselines,
        )


        # Step 11: Feature analysis
        feat_imp = base.analyse_features(
            model,
            X,
            y,
            cat_indices,
            compute_interactions=False,
        )
     

        # Step 12: Prediction test
        print(f"\n{'═' * 60}")
        print(f"  PREDICTION TEST")
        print(f"{'═' * 60}")

        predictor = base.PacePredictor(
            str(base.MODEL_PATH), str(base.BASELINES_PATH)
        )

        sample = df.iloc[0]
        actual_abs = sample[base.RAW_TARGET]
        actual_delta = sample[base.DELTA_TARGET]
        track = sample["TrackName"]

        sample_season = (
            int(sample["Season"])
            if "Season" in sample.index
            else None
        )
        baseline = predictor._get_baseline(track, sample_season)

        sample_race_id = sample[base.GROUP_KEY]
        total_laps = int(
            df.loc[
                df[base.GROUP_KEY] == sample_race_id,
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



        print(f"\n  Sample prediction:")
        print(f"    Driver:         {sample['Driver']} ({sample['Team']})")
        print(f"    Track:          {track}")
        if sample_season:
            print(f"    Season:         {sample_season}")
        print(f"    Track baseline: {baseline:.2f}s (mode: {predictor._baseline_mode})")
        print(f"    Compound:       {sample['Compound']} (Age: {int(sample['TireAge'])})")
        print(
            f"    Race lap:       {int(sample['RaceLapNumber'])} / {total_laps} "
            f"(frac: {sample['RaceLapFraction']:.3f})"
        )
        print(f"    ---")
        print(f"    Actual abs:     {actual_abs:.3f}s")
        print(f"    Predicted abs:   {predicted_abs:.3f}s")
        print(f"    Abs error:      {predicted_abs - actual_abs:+.3f}s")
        print(f"    ---")
        print(f"    Actual delta:   {actual_delta:+.3f}s")
        print(f"    Predicted delta:{predicted_delta:+.3f}s")
        print(f"    Delta error:    {predicted_delta - actual_delta:+.3f}s")
        print(f"    ---")
        print(f"    Uncertainty:    ±{std_unc:.3f}s")


        # Step 13: Save report
        base.save_report(
            fold_metrics=fold_metrics,
            feat_imp=feat_imp,
            params=best_params,
            track_baselines=track_baselines,
            baseline_mode=actual_mode,
            n_laps=n_laps_raw,
            n_races=df[base.GROUP_KEY].nunique(),
            n_laps_after_filter=n_laps_filtered,
            ablation_results=ablation_results
        )


        avg_r2 = base.np.mean([m["delta_r2"] for m in fold_metrics])
        avg_rmse = base.np.mean([m["delta_rmse"] for m in fold_metrics])
        avg_best_iter = base.np.mean([m["best_iteration"] for m in fold_metrics])

        print(f"\n{'═' * 60}")
        print(f"  PIPELINE COMPLETE ({model_name})")
        print(f"{'═' * 60}")
        
        print(f"  GPU:   {'Yes' if base.GPU_AVAILABLE else 'No'}")
        print(f"  Baseline mode:   {actual_mode}")
        print(f"  Laps:  {n_laps_raw:,} raw → {n_laps_filtered:,} after filtering")
        print(f"  Delta RMSE:      {avg_rmse:.4f}s")
        print(f"  Delta R²:        {avg_r2:.4f}")
        print(f"  Avg best_iter:   {avg_best_iter:.0f}")
        print(f"  Model predicts:  delta from track baseline")
        print(f"  PacePredictor converts back to absolute times")

        issues = []
        if avg_r2 < 0:
            issues.append(" Negative R² - model has no predictive power")
        elif avg_r2 < 0.15:
            issues.append(
                f"  Low R² ({avg_r2:.4f}) - model explains < 15% of variance"
            )

        if avg_best_iter < 50:
            issues.append("  Very low average best_iter - investigate regularisation")

        if issues:
            print(f"\n  Health Assessment:")
            for issue in issues:
                print(f"   - {issue}")
        else:
            print(f"\n  Health Assessment: OK")

        return {
            "model_name": model_name,
            "output_dir": str(base.MODEL_DIR),
            "n_laps_raw": n_laps_raw,
            "n_laps_filtered": n_laps_filtered,
            "delta_r2": avg_r2,
            "delta_rmse": avg_rmse,
            "best_iter": avg_best_iter,
        }
    except Exception as e:
        print(f"\n  ERROR during model run: {e}")
        return {
            "model_name": model_name,
            "output_dir": str(base.MODEL_DIR),
            "error": str(e),
        }

def main() -> None:
    if not MODEL_RUNS:
        raise ValueError("MODEL_RUNS is empty. Add at least one model config.")

    summaries: list[dict[str, Any]] = []
    for config in MODEL_RUNS:
        print("\n" + "=" * 72)
        summaries.append(run_single_model(config))

    if len(summaries) > 1:
        print("\n" + "=" * 72)
        print("  MODEL RUN SUMMARY")
        print("=" * 72)
        for summary in summaries:
            print(
                f"  {summary['model_name']}: "
                f"RMSE={summary['delta_rmse']:.4f}s, "
                f"R²={summary['delta_r2']:.4f}, "
                f"dir={summary['output_dir']}"
            )


if __name__ == "__main__":
    main()
