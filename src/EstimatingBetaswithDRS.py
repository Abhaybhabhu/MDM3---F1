import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
CSV_PATH = BASE_DIR / "data" / "processed" / "training_data.csv"

BASELINE_SOFT_BETA = 1.2e-9
MIN_RATIO_MEDIUM = 0.20
MIN_RATIO_HARD = 0.10
MIN_STINT_LAPS = 5

DRY_COMPOUNDS = ["SOFT", "MEDIUM", "HARD"]

SAVE_DATA = True # flag for if you want to overwrite the CSV files. 
# Going to set it to False while I test the drs addition.

print("Script folder:", BASE_DIR)
print("Looking for CSV at:", CSV_PATH)

if not CSV_PATH.exists():
    raise FileNotFoundError(f"Could not find file: {CSV_PATH}")

df = pd.read_csv(CSV_PATH)

# Ensure DRS feature exists and is numeric for regression.
if "DRS_Available" not in df.columns:
    df["DRS_Available"] = 0.0

df["DRS_Available"] = pd.to_numeric(df["DRS_Available"], errors="coerce").fillna(0.0)
df["DRS_Available"] = (df["DRS_Available"] > 0).astype(float)

df = df[df["Compound"].isin(DRY_COMPOUNDS)].copy()
df = df[
    (df["LapTimeSec"].notna()) &
    (df["TireAge"].notna()) &
    (df["RaceLapNumber"].notna())
].copy()

df = df[
    (df["LapTimeSec"] > 0) &
    (df["TireAge"] >= 1)
].copy()

df = df.sort_values(["RaceID", "Driver", "LapNumber"]).reset_index(drop=True)

group_keys = ["RaceID", "Driver"]

first_mask = df.groupby(group_keys).cumcount().eq(0)

compound_change = df.groupby(group_keys)["Compound"].transform(
    lambda s: s.ne(s.shift()).fillna(True)
)

age_reset = df.groupby(group_keys)["TireAge"].transform(
    lambda s: s.le(s.shift()).fillna(True)
)

lap_gap = df.groupby(group_keys)["LapNumber"].transform(
    lambda s: s.diff().fillna(1).ne(1)
)
# Create boolean flag for the start of a new stint (i.e. if any of the conditions are met)
new_stint = first_mask | compound_change | age_reset | lap_gap

df["StintIndex"] = new_stint.groupby([df["RaceID"], df["Driver"]]).cumsum()
df["StintID"] = (
    df["RaceID"].astype(str)
    + "__"
    + df["Driver"].astype(str)
    + "__"
    + df["StintIndex"].astype(str)
)


# REMOVE VERY SHORT / UNUSABLE STINTS
stint_stats = df.groupby("StintID").agg(
    n_laps=("LapTimeSec", "size"),
    age_std=("TireAge", "std"),
    compound=("Compound", "first")
)

valid_stints = stint_stats[
    (stint_stats["n_laps"] >= MIN_STINT_LAPS) &
    (stint_stats["age_std"] > 0)
].index

df = df[df["StintID"].isin(valid_stints)].copy().reset_index(drop=True)


keep_mask = np.ones(len(df), dtype=bool)

for stint_id, g in df.groupby("StintID"):
    idx = g.index.to_numpy()
    x = g[["TireAge", "DRS_Available"]].to_numpy()
    y = g["LapTimeSec"].to_numpy()

    model = LinearRegression().fit(x, y)
    resid = y - model.predict(x)

    med = np.median(resid) # median of residuals.
    mad = np.median(np.abs(resid - med)) # median absolute deviation.

    sigma = 1.4826 * mad if mad > 0 else 1.0
    local_keep = np.abs(resid - med) <= 3.0 * sigma

    keep_mask[idx] = local_keep

df = df[keep_mask].copy().reset_index(drop=True)

stint_stats = df.groupby("StintID").agg(
    n_laps=("LapTimeSec", "size"),
    age_std=("TireAge", "std"),
    compound=("Compound", "first")
)

valid_stints = stint_stats[
    (stint_stats["n_laps"] >= MIN_STINT_LAPS) &
    (stint_stats["age_std"] > 0)
].index

df = df[df["StintID"].isin(valid_stints)].copy().reset_index(drop=True)


means = df.groupby("StintID")[["LapTimeSec", "RaceLapNumber"]].transform("mean")
means_drs = df.groupby("StintID")["DRS_Available"].transform("mean")
# 'demeaning' takes place here, I think? It's a funny word ahah.
y_within = (df["LapTimeSec"] - means["LapTimeSec"]).to_numpy()
X_within = np.column_stack(
    [
        (df["RaceLapNumber"] - means["RaceLapNumber"]).to_numpy(),
        (df["DRS_Available"] - means_drs).to_numpy(),
    ]
)

race_model = LinearRegression(fit_intercept=False)
race_model.fit(X_within, y_within)

gamma_race = race_model.coef_[0]
gamma_drs = race_model.coef_[1]
print(f"Estimated common RaceLapNumber effect: {gamma_race:.6f} s/lap")
print(f"Estimated common DRS effect: {gamma_drs:.6f} s/lap")

# Adjust lap times
df["AdjLapTimeSec"] = (
    df["LapTimeSec"]
    - gamma_race * df["RaceLapNumber"]
    - gamma_drs * df["DRS_Available"]
)

slope_rows = []

for stint_id, g in df.groupby("StintID"):
    if len(g) < MIN_STINT_LAPS:
        continue
    if g["TireAge"].nunique() < 4:
        continue

    x = g[["TireAge"]].to_numpy()
    y = g["AdjLapTimeSec"].to_numpy()

    model = LinearRegression().fit(x, y)

    slope_rows.append({
        "StintID": stint_id,
        "RaceID": g["RaceID"].iloc[0],
        "Driver": g["Driver"].iloc[0],
        "Compound": g["Compound"].iloc[0],
        "NumLaps": len(g),
        "Slope_s_per_lap": model.coef_[0],
        "Intercept": model.intercept_,
    })

slopes_df = pd.DataFrame(slope_rows)

if slopes_df.empty:
    raise ValueError("No valid stint slopes were fitted. Try reducing MIN_STINT_LAPS.")

summary_rows = []

for compound in DRY_COMPOUNDS:
    sub = slopes_df[slopes_df["Compound"] == compound].copy()

    raw_median = sub["Slope_s_per_lap"].median()
    positive_median = sub.loc[sub["Slope_s_per_lap"] > 0, "Slope_s_per_lap"].median()
    weighted_mean = np.average(sub["Slope_s_per_lap"], weights=sub["NumLaps"])

    summary_rows.append({
        "Compound": compound,
        "NumStints": len(sub),
        "RawMedianSlope": raw_median,
        "PositiveMedianSlope": positive_median,
        "WeightedMeanSlope": weighted_mean,
    })

summary_df = pd.DataFrame(summary_rows)
print("\nCompound slope summary:")
print(summary_df.to_string(index=False))

soft_raw = summary_df.loc[
    summary_df["Compound"] == "SOFT", "RawMedianSlope"
].iloc[0]

soft_raw = max(soft_raw, 1e-12)

raw_ratios = {}
for _, row in summary_df.iterrows():
    raw_ratios[row["Compound"]] = max(row["RawMedianSlope"], 0.0) / soft_raw

print("\nRaw beta ratios from median slopes (Soft baseline = 1):")
for comp in DRY_COMPOUNDS:
    print(f"{comp}: {raw_ratios[comp]:.4f}")

beta_ratios = {
    "SOFT": 1.0,
    "MEDIUM": max(min(raw_ratios.get("MEDIUM", 0.0), 0.95), MIN_RATIO_MEDIUM),
    "HARD": max(min(raw_ratios.get("HARD", 0.0), raw_ratios.get("MEDIUM", 1.0)), MIN_RATIO_HARD),
}

beta_map = {
    comp: BASELINE_SOFT_BETA * ratio
    for comp, ratio in beta_ratios.items()
}

beta_params_df = pd.DataFrame(
    [
        {
            "Compound": comp,
            "BetaRatio": beta_ratios[comp],
            "BetaValue": beta_map[comp],
        }
        for comp in DRY_COMPOUNDS
    ]
)

print("\nFinal beta ratios used:")
for comp in DRY_COMPOUNDS:
    print(f"{comp}: {beta_ratios[comp]:.4f}")

print("\nSuggested beta_map:")
print(beta_map)
print("\nCalibrated beta parameters:")
print(beta_params_df.to_string(index=False))

if SAVE_DATA:  
    summary_df.to_csv("compound_slope_summary.csv", index=False)
    slopes_df.to_csv("stint_degradation_slopes.csv", index=False)
    beta_params_df.to_csv("compound_beta_parameters.csv", index=False)
    df.to_csv("cleaned_beta_calibration_data.csv", index=False)

    print("\nSaved:")
    print("- compound_slope_summary.csv")
    print("- stint_degradation_slopes.csv")
    print("- compound_beta_parameters.csv")
    print("- cleaned_beta_calibration_data.csv")

drs_counts = (
    df["DRS_Available"]
    .astype(float)
    .value_counts()
    .reindex([0.0, 1.0], fill_value=0)
)
drs_percent = (drs_counts / max(drs_counts.sum(), 1)) * 100.0

plt.figure(figsize=(7, 5))
bars = plt.bar(
    ["DRS Off (0)", "DRS On (1)"],
    drs_counts.values,
    color=["#4C78A8", "#F58518"]
)

for i, bar in enumerate(bars):
    count = int(drs_counts.iloc[i])
    pct = drs_percent.iloc[i]
    plt.text(
        bar.get_x() + bar.get_width() / 2,
        bar.get_height(),
        f"{count:,}\n({pct:.1f}%)",
        ha="center",
        va="bottom",
        fontsize=9,
    )

plt.ylabel("Lap count")
plt.title("DRS usage across dataset")
plt.grid(True, axis="y")
plt.tight_layout()
plt.show()

plt.figure(figsize=(10, 6))
for compound in DRY_COMPOUNDS:
    sub = slopes_df[slopes_df["Compound"] == compound]
    plt.hist(sub["Slope_s_per_lap"], bins=40, alpha=0.5, label=compound)

plt.axvline(0, color="black", linestyle="--")
plt.xlabel("Adjusted degradation slope (s/lap)")
plt.ylabel("Count")
plt.title("Distribution of stint degradation slopes by compound")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

plt.figure(figsize=(8, 5))
plot_order = ["SOFT", "MEDIUM", "HARD"]
plot_vals = [beta_ratios[c] for c in plot_order]

plt.bar(plot_order, plot_vals)
plt.ylabel("Relative beta (Soft = 1)")
plt.title("Calibrated compound beta ratios")
plt.grid(True, axis="y")
plt.tight_layout()
plt.show()