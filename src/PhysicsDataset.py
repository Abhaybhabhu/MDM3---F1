"""

Extract raw telemetry-derived lap features from FastF1
Run the calibrated thermal tyre model on top

Output:
    physics_raw_2022_2025.csv          (Stage 1: raw telemetry features)
    physics_modelled_2022_2025.csv     (Stage 2: + thermal model states)
"""

import os
import time
import pathlib
import warnings
from typing import Optional, Dict, List

import numpy as np
import pandas as pd
import fastf1

warnings.filterwarnings("ignore", module="fastf1")
warnings.filterwarnings("ignore", category=FutureWarning)

SEASONS = [2022, 2023, 2024, 2025]
test_seasons = [2022]
SESSION_TYPE = "R"
CACHE_DIR = "fastf1_cache"
OUTPUT_RAW_CSV = "physics_raw_2022_2025.csv"
OUTPUT_MODELLED_CSV = "physics_modelled_2022_2025.csv"
REQUEST_PAUSE_SEC = 1.0

DRY_COMPOUNDS = {"SOFT", "MEDIUM", "HARD"}
WET_COMPOUNDS = {"WET", "INTERMEDIATE"}
MIN_LAP_NUMBER = 2  # skip lap 1 only

# THERMAL MODEL PARAMETERS (CALIBRATED)
VEHICLE_MASS_KG = 800.0
G = 9.81
V_REF_MS = 150.0 / 3.6
K_AERO = (VEHICLE_MASS_KG * G) / (V_REF_MS ** 2)

A_TEMP = 1e-5       # calibrated heating coefficient
B_COOL = 0.005      # calibrated cooling coefficient
T_OPT_C = 90.0     # optimal tyre temperature NOTE : another change during debugging.
P_CURV = 2           # thermal penalty curvature
LAMBDA_TEMP = 5e-4   # thermal penalty strength
T_START_OFFSET = 15.0  # initial tyre temp = T_env + offset
DEFAULT_AMBIENT_C = 25.0
W_AIR = 0.4          # weight for air temp in T_env
W_TRACK = 0.6        # weight for track temp in T_env

V_SLIP_SCALE = 20
V_S_MAX = 2.25  # m/s equivalent

# DRS-corrected beta ratios (from EstimatingBetaswithDRS.py)
BETA_SOFT = 1.2e-8
BETA_MAP = {
    "SOFT": BETA_SOFT * 1.0,      # baseline
    "MEDIUM": BETA_SOFT * 0.3628,  # data-driven
    "HARD": BETA_SOFT * 0.15,      # floored for plausibility
}

# Sliding proxy weights
LAMBDA_BRAKE = 0.35
LAMBDA_ACCEL = 0.20
LAMBDA_CORNER = 0.45

# Sprint / testing detection
SPRINT_KEYWORDS = {"sprint", "sprint_qualifying", "sprint_shootout"}

# Known wet/problematic races to force-skip
FORCE_EXCLUDE = {
    (2022, "Emilia Romagna Grand Prix"),
    (2022, "Monaco Grand Prix"),
    (2022, "Singapore Grand Prix"),
    (2022, "Japanese Grand Prix"),
    (2023, "Monaco Grand Prix"),
    (2023, "Dutch Grand Prix"),
    (2024, "Canadian Grand Prix"),
    (2024, "British Grand Prix"),
    (2025, "Australian Grand Prix"),
}

os.makedirs(CACHE_DIR, exist_ok=True)
fastf1.Cache.enable_cache(CACHE_DIR)


def is_testing_or_sprint(event_row: pd.Series) -> bool:
    rnd = event_row.get("RoundNumber", np.nan)
    if pd.notna(rnd) and int(rnd) == 0:
        return True
    
    event_format = str(event_row.get("EventFormat", "")).strip().lower()
    if "testing" in event_format or "test" in event_format:
        return True
    if event_format in SPRINT_KEYWORDS:
        return True
    
    for col in ["Session1", "Session2", "Session3", "Session4", "Session5"]:
        val = str(event_row.get(col, "")).strip().lower()
        if "sprint" in val:
            return True
    
    name = str(event_row.get("EventName", "")).lower()
    if "test" in name:
        return True
    
    return False


def is_wet_race(session, year: int, event_name: str) -> bool:
    for ey, en in FORCE_EXCLUDE:
        if ey == year and en.lower() in event_name.lower():
            return True
    
    try:
        laps = session.laps
        if laps is not None and not laps.empty and "Compound" in laps.columns:
            compounds = set(laps["Compound"].dropna().astype(str).str.upper().unique())
            if compounds & WET_COMPOUNDS:
                return True
    except Exception:
        pass
    
    try:
        weather = session.weather_data
        if weather is not None and not weather.empty and "Rainfall" in weather.columns:
            if weather["Rainfall"].astype(bool).mean() > 0.10:
                return True
    except Exception:
        pass
    
    return False


def get_location(event_row: pd.Series) -> str:
    for key in ["Location", "Country", "EventName"]:
        val = event_row.get(key, "")
        if pd.notna(val) and str(val).strip():
            return str(val).strip()
    return "Unknown"


def get_circuit_length_km(location: str) -> float:
    lookup = {
        "Bahrain": 5.412, "Sakhir": 5.412, "Jeddah": 6.174,
        "Melbourne": 5.278, "Suzuka": 5.807, "Shanghai": 5.451,
        "Miami": 5.412, "Monaco": 3.337, "Barcelona": 4.657,
        "Montréal": 4.361, "Montreal": 4.361, "Spielberg": 4.318,
        "Silverstone": 5.891, "Budapest": 4.381, "Spa": 7.004,
        "Zandvoort": 4.259, "Monza": 5.793, "Singapore": 4.940,
        "Baku": 6.003, "Austin": 5.513, "Mexico": 4.304,
        "São Paulo": 4.309, "Las Vegas": 6.201, "Lusail": 5.380,
        "Abu Dhabi": 5.281, "Yas Island": 5.281, "Imola": 4.909,
    }
    loc_lower = str(location).lower()
    for key, val in lookup.items():
        if key.lower() in loc_lower or loc_lower in key.lower():
            return val
    return 5.0


# TELEMETRY EXTRACTION (per lap, fully defensive)
def extract_lap_telemetry(lap) -> Optional[Dict]:
    try:
        tel = lap.get_car_data()
    except Exception:
        return None
    
    if tel is None or tel.empty:
        return None
    
    if not all(c in tel.columns for c in ["Speed", "Throttle", "Brake"]):
        return None
    
    try:
        speed_kmh = pd.to_numeric(tel["Speed"], errors="coerce").fillna(0).to_numpy()
        throttle = pd.to_numeric(tel["Throttle"], errors="coerce").fillna(0).to_numpy()
        
        brake_raw = tel["Brake"]
        if brake_raw.dtype == bool:
            brake = brake_raw.astype(float).to_numpy()
        else:
            brake = pd.to_numeric(brake_raw, errors="coerce").fillna(0).to_numpy()
        
        speed_ms = speed_kmh / 3.6
        
        if len(speed_ms) < 10:
            return None
        
        # DRS 
        drs_active_frac = np.nan
        drs_detected = False
        if "DRS" in tel.columns:
            drs = pd.to_numeric(tel["DRS"], errors="coerce").fillna(0).to_numpy()
            drs_active_frac = float(np.mean(drs >= 10))
            drs_detected = drs_active_frac > 0.01
        
        return {
            "MeanSpeed_ms": float(np.mean(speed_ms)),
            "MaxSpeed_ms": float(np.max(speed_ms)),
            "MinSpeed_ms": float(np.min(speed_ms)),
            "SpeedStd_ms": float(np.std(speed_ms)),
            "BrakeIntensity_raw": float(np.mean(brake)),
            "BrakeFraction": float(np.mean(brake > 0.5)),
            "AccelIntensity_raw": float(np.mean(throttle)),
            "FullThrottleFrac": float(np.mean(throttle > 95.0)),
            "CorneringSeverity_raw": float(np.std(speed_ms)),
            "ThrottleMean_pct": float(np.mean(throttle)),
            "TelemetrySamples": int(len(speed_ms)),
            "DRSActiveFraction": drs_active_frac,
            "DRSDetected": int(drs_detected),
        }
    except Exception:
        return None


# WEATHER MERGE 
def get_weather_for_laps(laps_df: pd.DataFrame, session) -> pd.DataFrame:
    out = laps_df.copy()
    weather_cols = ["AirTemp", "TrackTemp", "Humidity", "WindSpeed"]
    
    for col in weather_cols:
        if col not in out.columns:
            out[col] = np.nan
    
    try:
        weather = session.weather_data
        if weather is None or weather.empty:
            return out
        
        use_cols = [c for c in weather_cols if c in weather.columns]
        if not use_cols or "Time" not in weather.columns:
            return out
        
        if "LapStartDate" not in out.columns or out["LapStartDate"].isna().all():
            return out
        
        wx = weather.copy()
        if pd.api.types.is_timedelta64_dtype(wx["Time"]):
            wx["_t"] = session.date + wx["Time"]
        else:
            wx["_t"] = pd.to_datetime(wx["Time"])
        
        out_sorted = out.sort_values("LapStartDate").copy()
        wx_sorted = wx.sort_values("_t")
        
        merged = pd.merge_asof(
            out_sorted,
            wx_sorted[["_t"] + use_cols].drop_duplicates(subset=["_t"]),
            left_on="LapStartDate",
            right_on="_t",
            direction="nearest",
            suffixes=("", "_wx"),
        )
        merged = merged.drop(columns=["_t"], errors="ignore")
        
        for col in weather_cols:
            wx_col = f"{col}_wx"
            if wx_col in merged.columns:
                merged[col] = merged[wx_col].combine_first(merged.get(col, pd.Series(dtype=float)))
                merged = merged.drop(columns=[wx_col])
        
        return merged
    except Exception:
        return out


# STAGE 1: PROCESS ONE RACE
def process_race(year: int, event_row: pd.Series) -> Optional[pd.DataFrame]:
    round_number = int(event_row["RoundNumber"])
    event_name = str(event_row["EventName"])
    location = get_location(event_row)
    circuit_km = get_circuit_length_km(location)
    race_id = f"{year}_{event_name}"
    
    print(f"  Loading {year} R{round_number:02d} - {event_name}")
    
    try:
        time.sleep(REQUEST_PAUSE_SEC)
        session = fastf1.get_session(year, round_number, SESSION_TYPE)
        session.load(laps=True, telemetry=True, weather=True, messages=False)
    except Exception as e:
        print(f"    Session load failed: {e}")
        return None
    
    if is_wet_race(session, year, event_name):
        print("    Skipping wet/mixed race")
        return None
    
    try:
        all_laps = session.laps
        if all_laps is None or all_laps.empty:
            print("    No lap data available")
            return None
    except Exception as e:
        print(f"    Cannot access laps: {e}")
        return None
    
    try:
        all_laps = get_weather_for_laps(all_laps, session)
    except Exception:
        pass
    
    rows = []
    ok_count = 0
    fail_count = 0
    skip_count = 0
    total = len(all_laps)
    
    for i in range(total):
        try:
            lap = all_laps.iloc[i]
        except Exception:
            fail_count += 1
            continue
        
        try:
            lap_number = int(lap["LapNumber"]) if pd.notna(lap.get("LapNumber")) else -1
        except Exception:
            fail_count += 1
            continue
        
        if lap_number < MIN_LAP_NUMBER:
            skip_count += 1
            continue
        
        lap_time = lap.get("LapTime", pd.NaT)
        if pd.isna(lap_time):
            skip_count += 1
            continue
        
        try:
            lap_time_s = float(lap_time.total_seconds())
        except Exception:
            skip_count += 1
            continue
        
        if lap_time_s <= 0 or lap_time_s > 300:
            skip_count += 1
            continue
        
        pit_in = lap.get("PitInTime", pd.NaT)
        pit_out = lap.get("PitOutTime", pd.NaT)
        if pd.notna(pit_in) or pd.notna(pit_out):
            skip_count += 1
            continue
        
        compound = str(lap.get("Compound", "")).upper().strip()
        if compound not in DRY_COMPOUNDS:
            skip_count += 1
            continue
        
        driver = str(lap.get("Driver", "")).strip()
        if not driver:
            skip_count += 1
            continue
        
        tel_features = extract_lap_telemetry(lap)
        if tel_features is None:
            fail_count += 1
            continue
        
        tire_age = 0
        for key in ["TyreLife", "TireAge"]:
            val = lap.get(key, np.nan)
            if pd.notna(val):
                try:
                    tire_age = int(val)
                    break
                except Exception:
                    pass
        
        stint = np.nan
        val = lap.get("Stint", np.nan)
        if pd.notna(val):
            try:
                stint = int(val)
            except Exception:
                pass
        
        row = {
            "Season": year,
            "RaceID": race_id,
            "EventName": event_name,
            "Location": location,
            "CircuitLength_km": circuit_km,
            "Driver": driver,
            "Team": str(lap.get("Team", "")).strip(),
            "LapNumber": lap_number,
            "Stint": stint,
            "Compound": compound,
            "TireAge": tire_age,
            "LapTime_s": lap_time_s,
            "AirTemp": pd.to_numeric(lap.get("AirTemp", np.nan), errors="coerce"),
            "TrackTemp": pd.to_numeric(lap.get("TrackTemp", np.nan), errors="coerce"),
            "Humidity": pd.to_numeric(lap.get("Humidity", np.nan), errors="coerce"),
            "WindSpeed": pd.to_numeric(lap.get("WindSpeed", np.nan), errors="coerce"),
        }
        row.update(tel_features)
        rows.append(row)
        ok_count += 1
        
        if (i + 1) % 100 == 0 or (i + 1) == total:
            print(f"    Progress: {i+1}/{total}  (ok={ok_count} fail={fail_count} skip={skip_count})", end="\r")
    
    print(f"    Done: {ok_count} ok, {fail_count} fail, {skip_count} skip out of {total} laps")
    
    if not rows:
        print("    No usable rows")
        return None
    
    race_df = pd.DataFrame(rows)
    race_df = race_df.sort_values(["Driver", "LapNumber"]).reset_index(drop=True)
    print(f"    Kept {len(race_df)} rows")
    return race_df


# STAGE 2: THERMAL MODEL
def add_physics_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add scaled features, sliding proxy, Fz, and run the thermal model.
    Operates on the raw physics CSV — no FastF1 dependency.
    """
    df = df.copy()
    df = df.sort_values(["RaceID", "Driver", "LapNumber"]).reset_index(drop=True)
    
    # --- Reconstruct stints if missing ---
    if df["Stint"].isna().mean() > 0.5:
        print("  Reconstructing stints from TireAge + Compound...")
        df["Stint"] = 0
        stint_counter = 0
        for (race_id, driver), group in df.groupby(["RaceID", "Driver"], sort=False):
            prev_age = None
            prev_comp = None
            for idx in group.index:
                age = df.at[idx, "TireAge"]
                comp = df.at[idx, "Compound"]
                new_stint = (prev_age is None) or (age <= prev_age) or (comp != prev_comp)
                if new_stint:
                    stint_counter += 1
                df.at[idx, "Stint"] = stint_counter
                prev_age = age
                prev_comp = comp
    
    # MinMax scale within each RaceID + Driver
    raw_cols = ["BrakeIntensity_raw", "AccelIntensity_raw", "CorneringSeverity_raw"]
    scaled_names = ["BrakeIntensity", "AccelIntensity", "CorneringSeverity"]
    
    for sc in scaled_names:
        df[sc] = 0.0
    
    for (race_id, driver), group in df.groupby(["RaceID", "Driver"]):
        idx = group.index
        for raw, scaled in zip(raw_cols, scaled_names):
            vals = df.loc[idx, raw]
            vmin, vmax = vals.min(), vals.max()
            if vmax > vmin:
                df.loc[idx, scaled] = (vals - vmin) / (vmax - vmin)
            else:
                df.loc[idx, scaled] = 0.0
    
    # weighted sliding proxy
    A = LAMBDA_BRAKE * df["BrakeIntensity"]
    B = LAMBDA_ACCEL * df["AccelIntensity"] # is this throttle?
    C = LAMBDA_CORNER * df["CorneringSeverity"]
    # NOTE: mutpliying this by ten to get it in a more reasonable range for the thermal model, since cornering severity is often quite low..
    df["SlidingProxy"] = V_SLIP_SCALE*C * (A+B) 
    # Justification is that our proxy for slip is rough and this prevents it from being too unrealistic at high values..
    
    # vertical load
    df["Fz_N"] = VEHICLE_MASS_KG * G + K_AERO * (df["MeanSpeed_ms"] ** 2)
    
    # --- Environmental temperature ---
    air = pd.to_numeric(df["AirTemp"], errors="coerce")
    track = pd.to_numeric(df["TrackTemp"], errors="coerce")
    
    df["T_env_C"] = DEFAULT_AMBIENT_C
    both = air.notna() & track.notna()
    air_only = air.notna() & track.isna()
    track_only = air.isna() & track.notna()
    df.loc[both, "T_env_C"] = W_AIR * air[both] + W_TRACK * track[both]
    df.loc[air_only, "T_env_C"] = air[air_only]
    df.loc[track_only, "T_env_C"] = track[track_only]
    
    # DRS availability
    if "DRSDetected" in df.columns:
        df["DRS_Available"] = df["DRSDetected"].fillna(0).astype(int)
    else:
        df["DRS_Available"] = 0

    # Thermal model (EXPONENTIAL decay)
    df = df.sort_values(["RaceID", "Driver", "Stint", "LapNumber"]).reset_index(drop=True)
    
    T_vals = np.full(len(df), np.nan)
    psi_vals = np.full(len(df), np.nan)
    q_vals = np.full(len(df), np.nan)
    
    for (race_id, driver, stint), group in df.groupby(["RaceID", "Driver", "Stint"], sort=False):
        idx = group.index.tolist()
        
        T_env_start = float(df.at[idx[0], "T_env_C"])
        T_state = T_env_start + T_START_OFFSET
        q_state = 1.0
        
        for row_idx in idx:
            row = df.loc[row_idx]
            dt = float(row["LapTime_s"])
            Fz = float(row["Fz_N"])
            vs = float(row["SlidingProxy"])
            T_env = float(row["T_env_C"])
            beta_c = BETA_MAP.get(row["Compound"], BETA_SOFT)
            
            T_vals[row_idx] = T_state
            q_vals[row_idx] = q_state
            
            psi_T = 1.0 + LAMBDA_TEMP * max(0.0, T_state - T_OPT_C) ** P_CURV
            psi_vals[row_idx] = psi_T

            vs_eff = V_S_MAX * np.tanh(vs / V_S_MAX)
            
            # Temperature update
            dTdt = A_TEMP * Fz * vs_eff - B_COOL * (T_state - T_env)
            T_state = T_state + dt * dTdt
            
            # Exponential tyre health decay
            damage = beta_c * dt * Fz * vs * psi_T
            q_state = q_state * np.exp(-damage)
    
    df["TyreTemp_C"] = T_vals
    df["Psi_T"] = psi_vals
    df["TyreHealth"] = q_vals
    df["DamageState"] = 1.0 - df["TyreHealth"]
    
    return df


# MAIN
def main():
    all_frames: List[pd.DataFrame] = []
    saved = []
    failed = []
    # NOTE : changed this for testing.
    for year in SEASONS:
        print(f"\n{'='*60}")
        print(f"  SEASON {year}")
        print(f"{'='*60}")
        try:
            schedule = fastf1.get_event_schedule(year)
        except Exception as e:
            print(f"Could not load schedule: {e}")
            continue

        # NOTE: only doing the first few races for testing.

        for _, event_row in schedule.iterrows():
            event_name = str(event_row.get("EventName", "Unknown"))
            rnd = event_row.get("RoundNumber", np.nan)
            if pd.isna(rnd):
                continue
            
            if is_testing_or_sprint(event_row):
                print(f"  Skipping: {year} R{int(rnd):02d} - {event_name}")
                continue
            
            try:
                race_df = process_race(year, event_row)
            except Exception as e:
                print(f"  Crashed on {event_name}: {e}")
                race_df = None
            
            if race_df is not None and not race_df.empty:
                all_frames.append(race_df)
                saved.append(f"{year} R{int(rnd):02d} - {event_name}")
            else:
                failed.append(f"{year} R{int(rnd):02d} - {event_name}")
    
    if not all_frames:
        print("\nNo data extracted.")
        return
    
    # ---- STAGE 1: Save raw physics CSV ----
    raw_df = pd.concat(all_frames, ignore_index=True)
    raw_df = raw_df.sort_values(["Season", "RaceID", "Driver", "LapNumber"]).reset_index(drop=True)
    
    raw_path = pathlib.Path(OUTPUT_RAW_CSV)
    raw_df.to_csv(raw_path, index=False)
    
    print(f"\n{'='*60}")
    print(f"  STAGE 1: Raw Physics Dataset")
    print(f"{'='*60}")
    print(f"Saved: {raw_path}")
    print(f"Rows:         {len(raw_df):,}")
    print(f"Races:        {raw_df['RaceID'].nunique()}")
    print(f"Drivers:      {raw_df['Driver'].nunique()}")
    print(f"Seasons:      {sorted(raw_df['Season'].unique())}")
    print(f"Mean speed:   {raw_df['MeanSpeed_ms'].mean():.2f} m/s")
    
    # ---- STAGE 2: Add physics model features ----
    print(f"\n{'='*60}")
    print(f"  STAGE 2: Adding Physics Model")
    print(f"{'='*60}")
    modelled_df = add_physics_features(raw_df)
    
    mod_path = pathlib.Path(OUTPUT_MODELLED_CSV)
    modelled_df.to_csv(mod_path, index=False)
    
    print(f"Saved: {mod_path}")
    print(f"Rows:         {len(modelled_df):,}")
    print(f"Mean temp:    {modelled_df['TyreTemp_C'].mean():.2f} C")
    print(f"Min health:   {modelled_df['TyreHealth'].min():.4f}")
    print(f"Mean health:  {modelled_df['TyreHealth'].mean():.4f}")
    print(f"Max temp:     {modelled_df['TyreTemp_C'].max():.2f} C")
    
    print("\nHealth by compound:")
    for comp in ["SOFT", "MEDIUM", "HARD"]:
        sub = modelled_df[modelled_df["Compound"] == comp]
        if len(sub) > 0:
            print(f"  {comp}: mean={sub['TyreHealth'].mean():.4f}  min={sub['TyreHealth'].min():.4f}  n={len(sub):,}")
    
    print("\nRaces per season:")
    for year in SEASONS:
        sub = modelled_df[modelled_df["Season"] == year]
        if len(sub) > 0:
            print(f"  {year}: {sub['RaceID'].nunique()} races, {len(sub):,} rows")
    
    print(f"\n{'='*60}")
    print(f"  SUMMARY")
    print(f"{'='*60}")
    print(f"Saved races ({len(saved)}):")
    for s in saved:
        print(f"  + {s}")
    
    if failed:
        print(f"\nFailed/skipped ({len(failed)}):")
        for f in failed:
            print(f"  - {f}")


if __name__ == "__main__":
    main()
