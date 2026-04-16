"""
pace_model_no_physics.py
========================
Baseline CatBoost pace model without the physics-informed tyre features.

This reuses the training, evaluation, reporting, and prediction pipeline from
model_GPU_physics.py, but limits the feature set to the baseline context
variables only:
  - Compound
  - TrackName
  - Driver
  - Team
  - TireAge
  - TireAgeSq
  - RaceLapNumber
  - RaceLapFraction
  - TrackTemp
  - AirTemp
  - Humidity
  - WindSpeed
  - GapToCarAhead
  - DRS_Available

Usage:
    python model_no_physics.py

Input:
    data/processed/training_data.parquet

Output:
    models/pace_model_no_physics.cbm
    models/track_baselines_no_physics.json
    models/training_report_no_physics.txt
"""

from __future__ import annotations

import pathlib
import pathlib
import sys
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parents[1]
SRC_DIR = BASE_DIR / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))



import model_GPU_physics as base


def configure_no_physics_model() -> None:
    """Repoint the physics training pipeline to the baseline feature set."""
    base.DATA_PATH = str(BASE_DIR / "data" / "processed" / "training_data.parquet")
    base.MODEL_DIR = BASE_DIR / "models"
    base.MODEL_DIR.mkdir(exist_ok=True)
    base.MODEL_PATH = base.MODEL_DIR / "pace_model_no_physics.cbm"
    base.BASELINES_PATH = base.MODEL_DIR / "track_baselines_no_physics.json"
    base.REPORT_PATH = base.MODEL_DIR / "training_report_no_physics.txt"

    base.CATEGORICAL_FEATURES = [
        "Compound",
        "TrackName",
        "Driver",
        "Team",
    ]

    base.NUMERICAL_FEATURES = [
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
    ]

    base.ALL_FEATURES = base.CATEGORICAL_FEATURES + base.NUMERICAL_FEATURES


def main() -> None:
    configure_no_physics_model()

    print("  F1 Virtual Race Strategist - Pace Model (no physics)")
    print(f"   Input:  {base.DATA_PATH}")
    print(f"   Output: {base.MODEL_PATH}")
    print(f"   Target: {base.DELTA_TARGET} (delta from track median)")
    print(f"   Baseline mode: {base.BASELINE_MODE}")
    print(f"   Device: {base.TASK_TYPE}")
    print()


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
        ablation_results=ablation_results,
    )


    avg_r2 = base.np.mean([m["delta_r2"] for m in fold_metrics])
    avg_rmse = base.np.mean([m["delta_rmse"] for m in fold_metrics])
    avg_best_iter = base.np.mean([m["best_iteration"] for m in fold_metrics])

    print(f"\n{'═' * 60}")
    print(f"  PIPELINE COMPLETE (no physics)")
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


if __name__ == "__main__":
    main()
