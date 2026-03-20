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

# Exclude pit in/out laps
if "PitInTime" in laps.columns:
    laps = laps[laps["PitInTime"].isna()].copy()
if "PitOutTime" in laps.columns:
    laps = laps[laps["PitOutTime"].isna()].copy()


def compute_lap_features(lap):
    tel = lap.get_car_data().add_distance()

    if tel.empty:
        return None

    # Raw telemetry
    speed_kmh = tel["Speed"].to_numpy()
    throttle = tel["Throttle"].to_numpy()
    brake = tel["Brake"].to_numpy()

    # Clean brake
    if brake.dtype == bool:
        brake = brake.astype(float)
    else:
        brake = pd.to_numeric(brake, errors="coerce")
        brake = np.nan_to_num(brake)

    # Clean throttle
    throttle = pd.to_numeric(throttle, errors="coerce")
    throttle = np.nan_to_num(throttle)

    # Clean speed and convert to m/s
    speed_kmh = pd.to_numeric(speed_kmh, errors="coerce")
    speed_kmh = np.nan_to_num(speed_kmh)
    speed_ms = speed_kmh / 3.6

    features = {
        "LapNumber": lap["LapNumber"],
        "Compound": lap["Compound"],
        "LapTime_s": lap["LapTime_s"],
        "MeanSpeed_ms": np.mean(speed_ms),
        "BrakeIntensity_raw": np.mean(brake),
        "AccelIntensity_raw": np.mean(throttle),
        "CorneringSeverity_raw": np.std(speed_ms),
    }
    return features


rows = []
for _, lap in laps.iterrows():
    feat = compute_lap_features(lap)
    if feat is not None:
        rows.append(feat)

df_features = pd.DataFrame(rows)
df_features = df_features.sort_values("LapNumber").reset_index(drop=True)

# Identify stints by compound changes
df_features["Stint"] = (df_features["Compound"] != df_features["Compound"].shift()).cumsum()

raw_features = ["BrakeIntensity_raw", "AccelIntensity_raw", "CorneringSeverity_raw"]

scaler = MinMaxScaler()
scaled = scaler.fit_transform(df_features[raw_features])

df_features["BrakeIntensity"] = scaled[:, 0]
df_features["AccelIntensity"] = scaled[:, 1]
df_features["CorneringSeverity"] = scaled[:, 2]

# Sliding proxy (unweighted baseline)
df_features["SlidingProxy"] = (
    df_features["BrakeIntensity"]
    + df_features["AccelIntensity"]
    + df_features["CorneringSeverity"]
)

g = 9.81
m = 800.0

# Assumption: aero downforce equals vehicle weight at 150 km/h
v_ref_ms = 150.0 / 3.6
k_aero = (m * g) / (v_ref_ms ** 2)

df_features["Fz_N"] = m * g + k_aero * (df_features["MeanSpeed_ms"] ** 2)


# Fixed thermal parameters
T_air = 25.0      # deg C
T_opt = 90.0      # deg C
p = 2             # fixed curvature

# Effective parameters to tune
a_temp = 2e-5     # heating coefficient = k_h/(m c)
b_cool = 0.015    # cooling coefficient = k_c/(m c)
lambda_temp = 5e-4
beta = 1e-9       # baseline wear coefficient

# Optional compound dependence for wear
beta_map = {
    "SOFT": 1.2e-9,
    "MEDIUM": 1.0e-9,
    "HARD": 0.8e-9
}

T_values = []
psi_values = []
q_values = []

for stint_id, stint_data in df_features.groupby("Stint"):

    # Reset states at each stint
    T_state = 40   # initial latent tyre temperature at stint start (deg C)
    q_state = 1.0    # fresh tyre

    for idx, row in stint_data.iterrows():

        # Store current states at start of lap n
        T_values.append(T_state)
        q_values.append(q_state)

        # Thermal amplification
        psi_T = 1.0 + lambda_temp * max(0.0, T_state - T_opt) ** p
        psi_values.append(psi_T)

        # Compound-specific wear coefficient
        beta_c = beta_map.get(row["Compound"], beta)

        # Lap duration
        dt = row["LapTime_s"]

        # Inputs
        Fz = row["Fz_N"]
        vs = row["SlidingProxy"]

        # Temperature update
        dTdt = a_temp * Fz * vs - b_cool * (T_state - T_air)
        T_next = T_state + dt * dTdt

        # Tyre health update
        dq = beta_c * dt * Fz * vs * psi_T
        q_next = q_state - dq
        q_next = max(q_next, 0.0)

        # Advance states
        T_state = T_next
        q_state = q_next

df_features["TyreTemp_C"] = T_values
df_features["Psi_T"] = psi_values
df_features["TyreHealth"] = q_values

csv_name = f"{YEAR}_{GP}_{DRIVER}_temperature_informed_features.csv"
df_features.to_csv(csv_name, index=False)
print(f"Saved feature table to: {csv_name}")

print(
    df_features[
        [
            "LapNumber", "Compound", "LapTime_s", "MeanSpeed_ms",
            "BrakeIntensity", "AccelIntensity", "CorneringSeverity",
            "SlidingProxy", "Fz_N", "TyreTemp_C", "Psi_T", "TyreHealth"
        ]
    ].head(20)
)

# 1. Latent tyre temperature across race
plt.figure()
plt.plot(df_features["LapNumber"], df_features["TyreTemp_C"], marker="o")
plt.xlabel("Lap")
plt.ylabel("Latent Tyre Temperature (°C)")
plt.title("Estimated Latent Tyre Temperature")
plt.grid(True)
plt.show()

# 2. Thermal amplification factor
plt.figure()
plt.plot(df_features["LapNumber"], df_features["Psi_T"], marker="o")
plt.xlabel("Lap")
plt.ylabel("Thermal Amplification $\\psi(T)$")
plt.title("Temperature-Based Wear Amplification")
plt.grid(True)
plt.show()

# 3. Tyre health across race
plt.figure()
plt.plot(df_features["LapNumber"], df_features["TyreHealth"], marker="o")
plt.xlabel("Lap")
plt.ylabel("Tyre Health")
plt.title("Temperature-Informed Tyre Health")
plt.grid(True)
plt.show()

# 4. Tyre health by stint
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
plt.title("Temperature-Informed Tyre Health by Stint")
plt.legend()
plt.grid(True)
plt.show()

# 5. Lap time vs tyre health by compound
plt.figure()
colors = {"SOFT": "red", "MEDIUM": "orange", "HARD": "gray"}

for compound, comp_data in df_features.groupby("Compound"):
    plt.scatter(
        comp_data["TyreHealth"],
        comp_data["LapTime_s"],
        label=compound,
        color=colors.get(compound, None)
    )

plt.xlabel("Tyre Health")
plt.ylabel("Lap Time (s)")
plt.title("Lap Time vs Temperature-Informed Tyre Health by Compound")
plt.legend()
plt.grid(True)
plt.show()

# 6. Temperature by stint
plt.figure()
for stint_id, stint_data in df_features.groupby("Stint"):
    plt.plot(
        stint_data["LapNumber"],
        stint_data["TyreTemp_C"],
        marker="o",
        label=f"Stint {stint_id} ({stint_data['Compound'].iloc[0]})"
    )

plt.xlabel("Lap")
plt.ylabel("Latent Tyre Temperature (°C)")
plt.title("Latent Tyre Temperature by Stint")
plt.legend()
plt.grid(True)
plt.show()