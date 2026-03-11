import os
import numpy as np
import pandas as pd
import fastf1
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt

YEAR = 2023
GP = "Spain"
SESSION = "R"
DRIVER = "VER"

CACHE_DIR = "fastf1_cache"
os.makedirs(CACHE_DIR, exist_ok=True)
fastf1.Cache.enable_cache(CACHE_DIR)

session = fastf1.get_session(YEAR, GP, SESSION)
session.load()

laps = session.laps.pick_drivers([DRIVER]).copy()

# Keep only laps with valid lap times
laps = laps[laps["LapTime"].notna()].copy()

# Convert lap time to seconds
laps["LapTime_s"] = laps["LapTime"].dt.total_seconds()


# Exclude wet / intermediate tyres
dry_compounds = ["SOFT", "MEDIUM", "HARD"]
laps = laps[laps["Compound"].isin(dry_compounds)].copy()

# Exclude first two laps 
laps = laps[laps["LapNumber"] > 2].copy()

# exclude pit in/out laps 
if "PitInTime" in laps.columns:
    laps = laps[laps["PitInTime"].isna()].copy()
if "PitOutTime" in laps.columns:
    laps = laps[laps["PitOutTime"].isna()].copy()

def compute_lap_features(lap):
    tel = lap.get_car_data().add_distance()

    if tel.empty:
        return None

    # basic channels
    speed = tel["Speed"].to_numpy()
    throttle = tel["Throttle"].to_numpy()
    brake = tel["Brake"].to_numpy()

    if brake.dtype == bool:
        brake = brake.astype(float)
    else:
        brake = pd.to_numeric(brake, errors="coerce")
        brake = np.nan_to_num(brake)

    throttle = pd.to_numeric(throttle, errors="coerce")
    throttle = np.nan_to_num(throttle)

    speed = pd.to_numeric(speed, errors="coerce")
    speed = np.nan_to_num(speed)

    features = {
        "LapNumber": lap["LapNumber"],
        "Compound": lap["Compound"],
        "LapTime_s": lap["LapTime_s"],
        "MeanSpeed": np.mean(speed),
        "BrakeIntensity": np.mean(brake),
        "AccelIntensity": np.mean(throttle),
        "CorneringSeverity": np.std(speed),
    }
    return features

rows = []

for _, lap in laps.iterrows():
    feat = compute_lap_features(lap)
    if feat is not None:
        rows.append(feat)

df_features = pd.DataFrame(rows)
df_features = df_features.sort_values("LapNumber").reset_index(drop=True)

df_features["Stint"] = (df_features["Compound"] != df_features["Compound"].shift()).cumsum()

# print(df_features.head(20))
# print("\nNumber of usable laps:", len(df_features))

# Save to CSV
# df_features.to_csv(f"{YEAR}_{GP}_{DRIVER}_lap_features.csv", index=False)
# print(f"\nSaved to {YEAR}_{GP}_{DRIVER}_lap_features.csv")

features = ["BrakeIntensity", "AccelIntensity", "CorneringSeverity"]

scaler = MinMaxScaler()
df_features[features] = scaler.fit_transform(df_features[features])

df_features["SlidingProxy"] = (
    df_features["BrakeIntensity"]
    + df_features["AccelIntensity"]
    + df_features["CorneringSeverity"]
)

g = 9.81
m = 800.0           # approximate F1 car mass
k_downforce = 0.002  # placeholder aero coefficient need to find a better way to get this

df_features["Fz"] = m * g + k_downforce * (df_features["MeanSpeed"] ** 2)


beta = 1e-9   # placeholder coefficient need  will to find this 

q_values = []

for stint_id, stint_data in df_features.groupby("Stint"):

    q = 1.0  # reset tyre health for new tyres

    for idx, row in stint_data.iterrows():

        q_values.append(q)

        q = q - beta * row["LapTime_s"] * row["Fz"] * row["SlidingProxy"]
        q = max(q, 0.0)

df_features["TyreHealth"] = q_values

plt.figure()
plt.plot(df_features["LapNumber"], df_features["TyreHealth"], marker="o")
plt.xlabel("Lap")
plt.ylabel("Tyre Health")
plt.title("Estimated Tyre Degradation")
plt.grid(True)
plt.show()

print(df_features[["LapNumber", "Compound", "MeanSpeed", "SlidingProxy", "Fz", "TyreHealth"]].head(20))

plt.figure()

plt.scatter(df_features["TyreHealth"], df_features["LapTime_s"])

plt.xlabel("Tyre Health")
plt.ylabel("Lap Time (s)")
plt.title("Lap Time vs Tyre Health")

plt.show()

plt.figure()

for stint_id, stint_data in df_features.groupby("Stint"):
    plt.plot(
        stint_data["LapNumber"],
        stint_data["TyreHealth"],
        marker="o",
        label=f"Stint {stint_id} ({stint_data['Compound'].iloc[0]})"
    )

plt.xlabel("Lap")
plt.ylabel("Tyre Health")
plt.title("Tyre Degradation by Stint")
plt.legend()
plt.grid(True)

plt.show()

plt.figure()

colors = {
    "SOFT": "red",
    "MEDIUM": "orange",
    "HARD": "gray"
}

for compound, stint_data in df_features.groupby("Compound"):
    plt.scatter(
        stint_data["TyreHealth"],
        stint_data["LapTime_s"],
        label=compound,
        color=colors.get(compound, None)
    )

plt.xlabel("Tyre Health")
plt.ylabel("Lap Time (s)")
plt.title("Lap Time vs Tyre Health by Compound")
plt.legend()
plt.grid(True)
plt.show()