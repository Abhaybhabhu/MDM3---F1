"""
data_pipeline.py
================
Complete data extraction and filtering pipeline for the
Virtual Race Strategist project.

Covers the 2022–2025 F1 seasons. Filters out:
  - Wet / intermediate weather races
  - Sprint weekend races
  - Safety car laps
  - Virtual safety car laps
  - Yellow flag laps
  - Red flag laps
  - Pit in-laps and out-laps
  - Standing-start lap 1
  - Statistical outlier laps

Extracts all features required by the CatBoost pace model.
FuelLoad is NOT included here — it will be calculated and
added in a separate step using a physics-based fuel model.

Usage:
    python data_pipeline.py

Output:
    training_data.parquet — clean, feature-rich dataset
"""

import warnings
import pathlib
import sys
import os
import re
import time
from typing import Optional, Tuple, List

import numpy as np
import pandas as pd
import fastf1

warnings.filterwarnings("ignore", module="fastf1")
warnings.filterwarnings("ignore", category=FutureWarning)

# ═══════════════════════════════════════════════════════════════
# 1. CONFIGURATION
# ═══════════════════════════════════════════════════════════════

from pathlib import Path

# Get the directory of this script
BASE_DIR = Path(__file__).resolve().parent

# Go up two levels to /root
ROOT_DIR = BASE_DIR.parents[1]

# Construct path to processed folder
OUTPUT_PATH = ROOT_DIR / "data" / "processed" / "training_data" 

print(f"Output path: {OUTPUT_PATH}")

SEASONS = [2022, 2023, 2024, 2025]

# FastF1 cache directory — avoids re-downloading data
CACHE_DIR = pathlib.Path("fastf1_cache")
CACHE_DIR.mkdir(exist_ok=True)
fastf1.Cache.enable_cache(str(CACHE_DIR))

# Processed race-level cache (prevents re-querying API on reruns)
PROCESSED_RACE_DIR = CACHE_DIR / "processed_races"
PROCESSED_RACE_DIR.mkdir(exist_ok=True)

# Skip marker directory — records races that were intentionally
# skipped (wet, no data, etc.) so we don't re-query the API
# for them on subsequent runs.
SKIP_MARKER_DIR = CACHE_DIR / "skipped_races"
SKIP_MARKER_DIR.mkdir(exist_ok=True)

# Optional request pacing + retry. Can be overridden with env vars.
REQUEST_PAUSE_SEC = float(os.getenv("FASTF1_REQUEST_PAUSE_SEC", "1.5"))
MAX_LOAD_RETRIES = int(os.getenv("FASTF1_MAX_RETRIES", "4"))
_LAST_FASTF1_CALL_TS = 0.0

# ── Known Wet Races (manual override) ─────────────────────────
# Some races start dry and end wet (or vice versa). These are
# excluded because mixed conditions break our dry-tire model.
# Format: (year, EventName substring)
FORCE_EXCLUDE_EVENTS = {
    (2022, "Emilia Romagna Grand Prix"),
    (2022, "Monaco Grand Prix"),
    (2022, "Singapore Grand Prix"),
    (2022, "Japanese Grand Prix"),
    (2023, "Monaco Grand Prix"),
    (2023, "Dutch Grand Prix"),
}

# ── Sprint Detection ─────────────────────────────────────────
# Sprint weekends are excluded entirely because:
#   - The sprint race on Saturday uses tire sets from the
#     allocation, affecting what is available for Sunday
#   - Strategy dynamics are different (parc fermé rules, etc.)
# We only keep "conventional" weekend formats.
SPRINT_EVENT_FORMATS = {
    "sprint",
    "sprint_qualifying",
    "sprint_shootout",
}

SPRINT_SESSION_NAMES = {
    "Sprint",
    "Sprint Race",
    "Sprint Qualifying",
    "Sprint Shootout",
}

# ── Circuit Physical Properties ───────────────────────────────
# (length_km, number_of_corners)
CIRCUIT_META = {
    "Sakhir":               (5.412, 15),
    "Jeddah":               (6.174, 27),
    "Melbourne":            (5.278, 14),
    "Suzuka":               (5.807, 18),
    "Shanghai":             (5.451, 16),
    "Miami":                (5.412, 19),
    "Monaco":               (3.337, 19),
    "Montréal":             (4.361, 14),
    "Montreal":             (4.361, 14),
    "Barcelona":            (4.657, 16),
    "Spielberg":            (4.318, 10),
    "Silverstone":          (5.891, 18),
    "Budapest":             (4.381, 14),
    "Spa-Francorchamps":    (7.004, 19),
    "Zandvoort":            (4.259, 14),
    "Monza":                (5.793, 11),
    "Singapore":            (4.940, 23),
    "Baku":                 (6.003, 20),
    "Austin":               (5.513, 20),
    "México":               (4.304, 17),
    "Mexico City":          (4.304, 17),
    "São Paulo":            (4.309, 15),
    "Sao Paulo":            (4.309, 15),
    "Las Vegas":            (6.201, 17),
    "Lusail":               (5.380, 16),
    "Yas Island":           (5.281, 16),
    "Abu Dhabi":            (5.281, 16),
    "Imola":                (4.909, 19),
}

# ── Pit Stop Time Loss (seconds) by Circuit ──────────────────
PIT_LOSS = {
    "Sakhir":               22.5,
    "Jeddah":               23.0,
    "Melbourne":            22.0,
    "Suzuka":               22.5,
    "Shanghai":             23.0,
    "Miami":                24.0,
    "Monaco":               20.0,
    "Montréal":             18.5,
    "Montreal":             18.5,
    "Barcelona":            22.0,
    "Spielberg":            19.5,
    "Silverstone":          20.5,
    "Budapest":             21.0,
    "Spa-Francorchamps":    21.0,
    "Zandvoort":            20.0,
    "Monza":                22.0,
    "Singapore":            28.0,
    "Baku":                 24.5,
    "Austin":               22.0,
    "México":               22.0,
    "Mexico City":          22.0,
    "São Paulo":            22.5,
    "Sao Paulo":            22.5,
    "Las Vegas":            24.0,
    "Lusail":               22.0,
    "Yas Island":           22.5,
    "Abu Dhabi":            22.5,
    "Imola":                22.5,
}
DEFAULT_PIT_LOSS = 23.0

# Track status codes from F1 timing
# We ONLY keep laps with status '1' (all clear)
TRACK_STATUS_GREEN = {"1"}

# Wet tire compound names
WET_COMPOUNDS = {"WET", "INTERMEDIATE", "UNKNOWN", "TEST_UNKNOWN"}

# Minimum fraction of a race that must have zero rainfall to
# classify as a dry race (when not in manual exclusion list)
DRY_THRESHOLD = 0.90

# Outlier bounds relative to session median lap time
OUTLIER_LOWER_FACTOR = 0.80
OUTLIER_UPPER_FACTOR = 1.50


# ═══════════════════════════════════════════════════════════════
# 2. HELPER FUNCTIONS
# ═══════════════════════════════════════════════════════════════

def _match_location(location: str, lookup: dict, default):
    """
    Fuzzy-match a FastF1 location string to our lookup tables.
    Tries exact match, then substring matching in both directions.
    """
    if not location:
        return default

    # Exact match
    if location in lookup:
        return lookup[location]

    # Substring matching
    loc_lower = location.lower()
    for key, value in lookup.items():
        if key.lower() in loc_lower or loc_lower in key.lower():
            return value

    return default


def _slugify(value: str) -> str:
    """Convert a string into a safe filename component."""
    value = re.sub(r"[^A-Za-z0-9]+", "_", str(value))
    return value.strip("_") or "unknown"


def _race_cache_path(
    year: int, round_number: int, event_name: str
) -> pathlib.Path:
    """Path for the processed race parquet cache."""
    slug = _slugify(event_name)
    return (
        PROCESSED_RACE_DIR
        / f"{year}_R{int(round_number):02d}_{slug}.parquet"
    )


def _skip_marker_path(
    year: int, round_number: int, event_name: str
) -> pathlib.Path:
    """
    Path for a skip marker file.
    When a race is intentionally skipped (wet, no data, sprint),
    we write a small text file so that subsequent runs don't
    re-download and re-check it.
    """
    slug = _slugify(event_name)
    return (
        SKIP_MARKER_DIR
        / f"{year}_R{int(round_number):02d}_{slug}.skip"
    )


def _write_skip_marker(
    year: int,
    round_number: int,
    event_name: str,
    reason: str,
) -> None:
    """Write a skip marker file recording why a race was skipped."""
    path = _skip_marker_path(year, round_number, event_name)
    path.write_text(reason, encoding="utf-8")


def _read_skip_marker(
    year: int, round_number: int, event_name: str
) -> Optional[str]:
    """
    Read a skip marker file. Returns the reason string if the
    marker exists, or None if it doesn't.
    """
    path = _skip_marker_path(year, round_number, event_name)
    if path.exists():
        return path.read_text(encoding="utf-8").strip()
    return None


def _is_rate_limited(error_msg: str) -> bool:
    """
    Check if an error message indicates an API rate limit.
    Covers known FastF1 / F1 API error patterns.
    """
    msg = error_msg.lower()
    return (
        "429" in msg
        or "rate" in msg
        or "too many request" in msg
        or "500 calls" in msg
        or "calls/h" in msg
        or "limit" in msg and "api" in msg
    )


def _throttle_fastf1_requests() -> None:
    """
    Enforce a minimum delay between FastF1 API calls.
    This only gates inter-session calls; FastF1's internal
    sub-calls within session.load() cannot be throttled
    externally. The retry logic with backoff is the real
    protection against rate limits.
    """
    global _LAST_FASTF1_CALL_TS
    if REQUEST_PAUSE_SEC <= 0:
        return

    now = time.monotonic()
    elapsed = now - _LAST_FASTF1_CALL_TS
    if elapsed < REQUEST_PAUSE_SEC:
        time.sleep(REQUEST_PAUSE_SEC - elapsed)

    _LAST_FASTF1_CALL_TS = time.monotonic()


def get_circuit_meta(location: str) -> Tuple[float, int]:
    return _match_location(location, CIRCUIT_META, (5.0, 15))


def get_pit_loss(location: str) -> float:
    return _match_location(location, PIT_LOSS, DEFAULT_PIT_LOSS)


# ═══════════════════════════════════════════════════════════════
# 3. SPRINT DETECTION
# ═══════════════════════════════════════════════════════════════

def _get_session_names(event_row) -> list:
    """Extract all session names from a schedule row."""
    sessions = []
    for i in range(1, 6):
        col = f"Session{i}"
        try:
            val = event_row.get(col, "")
            if val and str(val).strip() and str(val) != "nan":
                sessions.append(str(val).strip())
        except Exception:
            pass
    return sessions


def is_sprint_weekend(event_row) -> bool:
    """
    Determine if an event is a sprint weekend using two
    detection methods for robustness across FastF1 versions.

    Method 1: Check EventFormat column
    Method 2: Check session names for Sprint sessions
    """
    # Method 1: EventFormat column
    event_format = ""
    try:
        event_format = str(
            event_row.get("EventFormat", "")
        ).strip().lower()
    except Exception:
        pass

    if event_format in SPRINT_EVENT_FORMATS:
        return True

    # Method 2: Session names contain "Sprint"
    session_names = _get_session_names(event_row)
    for name in session_names:
        if name in SPRINT_SESSION_NAMES:
            return True
        if "sprint" in name.lower():
            return True

    return False


# ═══════════════════════════════════════════════════════════════
# 4. WET WEATHER DETECTION
# ═══════════════════════════════════════════════════════════════

def is_wet_race(session, year: int, event_name: str) -> bool:
    """
    Determine whether a race had wet conditions.

    Uses three signals (any one is sufficient to classify as wet):
      1. Manual exclusion list
      2. Any driver used WET or INTERMEDIATE compounds
      3. Rainfall detected in >10% of weather readings

    Parameters
    ----------
    session : fastf1.core.Session
        Loaded race session.
    year : int
        Season year.
    event_name : str
        Event name string from the schedule.

    Returns
    -------
    bool
        True if the race should be excluded due to wet conditions.
    """
    # ── Signal 1: Manual exclusion list ───────────────────────
    for exc_year, exc_name in FORCE_EXCLUDE_EVENTS:
        if exc_year == year and exc_name.lower() in event_name.lower():
            return True
        if exc_year == year and event_name.lower() in exc_name.lower():
            return True

    # ── Signal 2: Wet tire compounds used ─────────────────────
    try:
        laps = session.laps
        if laps is not None and not laps.empty:
            compounds_used = set(
                laps["Compound"].dropna().str.upper().unique()
            )
            if compounds_used & WET_COMPOUNDS:
                return True
    except Exception:
        pass

    # ── Signal 3: Rainfall in weather data ────────────────────
    try:
        weather = session.weather_data
        if weather is not None and not weather.empty:
            if "Rainfall" in weather.columns:
                rain_fraction = (
                    weather["Rainfall"].astype(bool).mean()
                )
                if rain_fraction > (1.0 - DRY_THRESHOLD):
                    return True
    except Exception:
        pass

    return False


# ═══════════════════════════════════════════════════════════════
# 5. LAP FILTERING
# ═══════════════════════════════════════════════════════════════

def filter_laps(laps: pd.DataFrame) -> pd.DataFrame:
    """
    Apply all filtering rules to remove laps that would
    confuse the pace model.

    Filters applied (in order):
      1. Remove laps with missing LapTime
      2. Remove pit in-laps (PitInTime is not NaN)
      3. Remove pit out-laps (PitOutTime is not NaN)
      4. Remove safety car laps (TrackStatus != '1')
      5. Remove VSC laps (same filter)
      6. Remove yellow flag laps (same filter)
      7. Remove red flag laps (same filter)
      8. Remove lap 1 (standing start)
      9. Remove laps with missing/invalid compound data
      10. Remove statistical outliers

    Parameters
    ----------
    laps : pd.DataFrame
        Raw laps DataFrame from FastF1.

    Returns
    -------
    pd.DataFrame
        Filtered laps with a new LapTimeSec column.
    """
    initial_count = len(laps)
    removal_log = {}

    # ── 1. Missing LapTime ────────────────────────────────────
    mask = laps["LapTime"].notna()
    removal_log["Missing LapTime"] = (~mask).sum()
    laps = laps[mask].copy()

    # Convert LapTime (timedelta64) to float seconds
    laps["LapTimeSec"] = laps["LapTime"].dt.total_seconds()

    # Remove any laps where conversion produced NaN or <= 0
    mask = laps["LapTimeSec"] > 0
    removal_log["Invalid LapTimeSec"] = (~mask).sum()
    laps = laps[mask]

    # ── 2. Pit In-Laps ───────────────────────────────────────
    mask = laps["PitInTime"].isna()
    removal_log["Pit In-Laps"] = (~mask).sum()
    laps = laps[mask]

    # ── 3. Pit Out-Laps ──────────────────────────────────────
    mask = laps["PitOutTime"].isna()
    removal_log["Pit Out-Laps"] = (~mask).sum()
    laps = laps[mask]

    # ── 4–7. Track Status (SC, VSC, Yellow, Red) ─────────────
    if "TrackStatus" in laps.columns:
        laps["_status_clean"] = (
            laps["TrackStatus"]
            .astype(str)
            .str.strip()
        )

        mask = laps["_status_clean"].apply(
            lambda s: len(s) > 0 and all(
                c in TRACK_STATUS_GREEN for c in s
            )
        )

        non_green_laps = (~mask).sum()
        removal_log[
            "Non-green laps (SC/VSC/Yellow/Red)"
        ] = non_green_laps
        laps = laps[mask]
        laps = laps.drop(columns=["_status_clean"])
    else:
        removal_log[
            "Non-green laps"
        ] = "N/A (no TrackStatus column)"

    # ── 8. Lap 1 (Standing Start) ────────────────────────────
    mask = laps["LapNumber"] > 1
    removal_log["Lap 1 (standing start)"] = (~mask).sum()
    laps = laps[mask]

    # ── 9. Invalid Compound Data ──────────────────────────────
    if "Compound" in laps.columns:
        laps["Compound"] = (
            laps["Compound"].astype(str).str.upper()
        )
        mask = (
            laps["Compound"].notna()
            & ~laps["Compound"].isin(WET_COMPOUNDS)
            & ~laps["Compound"].isin({"NAN", "NONE", ""})
        )
        removal_log["Invalid/wet compound"] = (~mask).sum()
        laps = laps[mask]

    # ── 10. Statistical Outliers ──────────────────────────────
    if len(laps) > 10:
        median_time = laps["LapTimeSec"].median()
        lower = median_time * OUTLIER_LOWER_FACTOR
        upper = median_time * OUTLIER_UPPER_FACTOR
        mask = laps["LapTimeSec"].between(lower, upper)
        removal_log["Statistical outliers"] = (~mask).sum()
        laps = laps[mask]

    # ── Summary ───────────────────────────────────────────────
    final_count = len(laps)
    total_removed = initial_count - final_count

    print(
        f"     Filtering: {initial_count} → {final_count} laps "
        f"({total_removed} removed)"
    )
    for reason, count in removal_log.items():
        if isinstance(count, int) and count > 0:
            print(f"       - {reason}: {count}")

    return laps


# ═══════════════════════════════════════════════════════════════
# 6. DIRTY AIR / GAP FEATURE
# ═══════════════════════════════════════════════════════════════

def compute_gap_to_car_ahead(
    laps: pd.DataFrame,
    circuit_length_m: float,
    verbose: bool = False
) -> pd.DataFrame:
    """
    Compute the time gap to the car immediately ahead on track.

    Uses two methods with automatic fallback:
      1. FastF1's DistanceToDriverAhead (direct F1 measurement)
      2. LapStartDate timestamp differencing (derived)

    Parameters
    ----------
    laps : pd.DataFrame
        Filtered laps with LapTimeSec, LapNumber, LapStartDate.
    circuit_length_m : float
        Circuit length in meters for distance → time conversion.

    Returns
    -------
    pd.DataFrame
        Input DataFrame with GapToCarAhead column added (seconds).
    """
    laps = laps.copy()

    # ── Method 1: Direct F1 timing data ───────────────────────
    has_distance = (
        "DistanceToDriverAhead" in laps.columns
        and laps["DistanceToDriverAhead"].notna().mean() > 0.5
    )

    if has_distance:
        avg_speed_ms = circuit_length_m / laps["LapTimeSec"]
        laps["GapToCarAhead"] = (
            pd.to_numeric(
                laps["DistanceToDriverAhead"], errors="coerce"
            )
            / avg_speed_ms
        )
        method = "DistanceToDriverAhead"
    else:
        # ── Method 2: Timestamp fallback ──────────────────────
        laps["GapToCarAhead"] = np.nan

        for lap_num, group in laps.groupby("LapNumber"):
            if len(group) < 2:
                continue

            if "LapStartDate" not in group.columns:
                continue

            sorted_grp = group.sort_values("LapStartDate")
            deltas = (
                sorted_grp["LapStartDate"]
                .diff()
                .dt.total_seconds()
            )
            laps.loc[sorted_grp.index, "GapToCarAhead"] = (
                deltas.values
            )

        method = "LapStartDate"

    # print out some stats about the gap feature to verify it looks reasonable
    if verbose:
        gap_stats = laps["GapToCarAhead"].describe()
        print(f"      pre-na filling GapToCarAhead stats:\n{gap_stats}")

    # ── Post-processing ───────────────────────────────────────
    laps["GapToCarAhead"] = laps["GapToCarAhead"].fillna(10.0)
    laps.loc[
        laps["GapToCarAhead"] < 0, "GapToCarAhead"
    ] = 0.5
    laps["GapToCarAhead"] = laps["GapToCarAhead"].clip(
        0.0, 10.0
    )
    if verbose:
        # print distribution stats for the gap feature to verify it looks reasonable
        gap_stats = laps["GapToCarAhead"].describe()
        print(f"       GapToCarAhead stats:\n{gap_stats}")

    print(f"       Gap method: {method}")
    return laps


# ═══════════════════════════════════════════════════════════════
# 7. FEATURE ENGINEERING
# ═══════════════════════════════════════════════════════════════

def engineer_features(
    laps: pd.DataFrame,
    session,
    location: str,
    total_laps: int,
    year: int,
    event_name: str,
    verbose: bool = False,
) -> pd.DataFrame:
    """
    Engineer all features required by the CatBoost pace model.

    Features created:
      A) Tire:       TireAge, TireAgeSq, Compound
      B) Car/Driver: Driver, Team
      C) Race State: RaceLapNumber, GapToCarAhead, DRS_Available
      D) Weather:    TrackTemp, AirTemp, Humidity, WindSpeed
      E) Track:      TrackName, CircuitLength, NumberOfCorners
      F) Metadata:   RaceID, LapNumber, PitLossSec, TotalRaceLaps

    NOTE: FuelLoad is NOT calculated here. It will be added
    later using a separate physics-based fuel consumption model.

    Parameters
    ----------
    laps : pd.DataFrame
        Filtered laps from filter_laps().
    session : fastf1.core.Session
        Loaded session (for weather data access).
    location : str
        Circuit location string from FastF1.
    total_laps : int
        Total race distance in laps.
    year : int
        Season year.
    event_name : str
        Event name for RaceID construction.

    Returns
    -------
    pd.DataFrame
        Feature-rich DataFrame ready for model training.
    """

    if verbose:
        print(f"{laps[['LapStartDate']].head()=}")  # print first few rows to verify LapTimeSec and GapToCarAhead look correct
        print(f" {laps.columns=}")  # print columns to verify expected columns are present before engineering
    if verbose:
        print(f"     Engineering features for {location}...")

    laps = laps.copy()
    circuit_length_km, num_corners = get_circuit_meta(location)
    circuit_length_m = circuit_length_km * 1000

    # ══════════════════════════════════════════════════════════
    # A) TIRE FEATURES
    # ══════════════════════════════════════════════════════════

    if "TyreLife" in laps.columns:
        laps["TireAge"] = pd.to_numeric(
            laps["TyreLife"], errors="coerce"
        ).fillna(0).astype(int)
    else:
        if "Stint" in laps.columns:
            laps["TireAge"] = laps.groupby(
                ["Driver", "Stint"]
            ).cumcount()
        else:
            laps["TireAge"] = 0

    laps["TireAgeSq"] = laps["TireAge"] ** 2

    # ══════════════════════════════════════════════════════════
    # B) CAR / DRIVER FEATURES
    # ══════════════════════════════════════════════════════════

    laps["Driver"] = laps["Driver"].astype(str).str.strip()
    laps["Team"] = laps["Team"].astype(str).str.strip()

    # ══════════════════════════════════════════════════════════
    # C) RACE STATE FEATURES
    # ══════════════════════════════════════════════════════════

    laps["RaceLapNumber"] = laps["LapNumber"].astype(int)

    laps = compute_gap_to_car_ahead(laps, circuit_length_m, verbose=verbose)

    laps["DRS_Available"] = 0
    drs_mask = (
        (laps["GapToCarAhead"] <= 1.0)
        & (laps["LapNumber"] > 2)
    )
    laps.loc[drs_mask, "DRS_Available"] = 1
    if verbose:
        available_pct = laps["DRS_Available"].mean() * 100
        print(
            f"       DRS Available: {available_pct:.1f}% of laps"
        )
        # print the if the mask is working correctly by showing some examples
        print(
            f"       DRS Available examples:\n"
            f"{laps[drs_mask][['LapNumber', 'GapToCarAhead', 'DRS_Available']].head(10)}"
        )

    # ══════════════════════════════════════════════════════════
    # D) WEATHER FEATURES
    # ══════════════════════════════════════════════════════════

    weather = None
    try:
        if verbose:
            print("       Loading weather data...")
        weather = session.weather_data
    except Exception:
        if verbose:
            print(
                f"       Weather: failed to load weather data, "
                f"using NaN for weather features"
            )
        pass

    if (
        weather is not None
        and not weather.empty
        and "LapStartDate" in laps.columns
    ):
        weather = weather.copy()

        weather_cols = [
            "AirTemp", "TrackTemp", "Humidity", "WindSpeed"
        ]
        available_cols = [
            c for c in weather_cols if c in weather.columns
        ]

        if available_cols and "Time" in weather.columns:
            try:
                if pd.api.types.is_timedelta64_dtype(
                    weather["Time"]
                ):
                    session_start = session.date
                    weather["_merge_time"] = (
                        session_start + weather["Time"]
                    )
                else:
                    weather["_merge_time"] = pd.to_datetime(
                        weather["Time"]
                    )

                laps = laps.sort_values("LapStartDate")
                weather = weather.sort_values("_merge_time")

                laps = pd.merge_asof(
                    laps,
                    weather[
                        ["_merge_time"] + available_cols
                    ].drop_duplicates(subset=["_merge_time"]),
                    left_on="LapStartDate",
                    right_on="_merge_time",
                    direction="nearest",
                )

                if "_merge_time" in laps.columns:
                    laps = laps.drop(columns=["_merge_time"])

                print(
                    f"       Weather: merged "
                    f"{len(available_cols)} columns"
                )

            except Exception as e:
                print(
                    f"       Weather: merge failed ({e}), "
                    f"using NaN"
                )
                for col in weather_cols:
                    if col not in laps.columns:
                        laps[col] = np.nan
    else:
        print(
            f"       Weather: no data available, using NaN"
        )

    for col in ["AirTemp", "TrackTemp", "Humidity", "WindSpeed"]:
        if col not in laps.columns:
            laps[col] = np.nan
    
    if verbose:
        print(
            f"       Weather feature stats:\n"
            f"{laps[['AirTemp', 'TrackTemp', 'Humidity', 'WindSpeed']].describe()}"
        )
        # After building weather["_merge_time"]:
        print(f"       Weather time dtype: {weather['_merge_time'].dtype}")
        print(f"       Weather time range: {weather['_merge_time'].min()} → {weather['_merge_time'].max()}")
        print(f"       LapStartDate dtype: {laps['LapStartDate'].dtype}")
        print(f"       LapStartDate range: {laps['LapStartDate'].min()} → {laps['LapStartDate'].max()}")
        print(f"       session.date: {session.date}")
        print(f"       Weather sample:\n{weather[['_merge_time'] + available_cols].head()}")
        print(f"       Lap data sample:\n {laps[['Time', 'LapTime', 'LapStartDate']].head(10)}")

    # ══════════════════════════════════════════════════════════
    # E) TRACK FEATURES
    # ════════════════════════════════════════════════
    laps["TrackName"] = location
    laps["CircuitLength"] = circuit_length_km
    laps["NumberOfCorners"] = num_corners

    # ══════════════════════════════════════════════════════════
    # F) METADATA (not model features, used for grouping)
    # ══════════════════════════════════════════════════════════

    laps["RaceID"] = f"{year}_{event_name}"
    laps["PitLossSec"] = get_pit_loss(location)
    laps["TotalRaceLaps"] = total_laps

    return laps


# ═══════════════════════════════════════════════════════════════
# 8. SESSION PROCESSING
# ═══════════════════════════════════════════════════════════════

def process_session(
    year: int,
    event_name: str,
    round_number: int,
    verbose: bool = False

) -> Optional[pd.DataFrame]:
    """
    Load, filter, and feature-engineer a single race session.

    Returns only the OUTPUT_COLUMNS so that the race-level
    cache doesn't contain unnecessary raw FastF1 columns.

    If the session is skipped (wet, no data), writes a skip
    marker so subsequent runs don't re-query the API.

    Parameters
    ----------
    year : int
        Season year.
    event_name : str
        Event name from the schedule.
    round_number : int
        Round number for session loading.

    Returns
    -------
    pd.DataFrame or None
        Processed laps with only OUTPUT_COLUMNS,
        or None if the session is excluded/empty.
    """
    session = None
    last_error = None

    for attempt in range(1, MAX_LOAD_RETRIES + 1):
        try:
            _throttle_fastf1_requests()
            session = fastf1.get_session(
                year, round_number, "R"
            )
            session.load(
                telemetry=False,
                weather=True,
                messages=False,
            )
            break
        except Exception as e:
            last_error = e
            msg = str(e)

            if attempt == MAX_LOAD_RETRIES:
                print(
                    f"     Failed to load after "
                    f"{attempt} tries: {e}"
                )
                _write_skip_marker(
                    year, round_number, event_name,
                    f"load_failed: {e}",
                )
                return None

            backoff = REQUEST_PAUSE_SEC * (2 ** (attempt - 1))

            if _is_rate_limited(msg):
                print(
                    f"     Rate-limited "
                    f"(attempt {attempt}/{MAX_LOAD_RETRIES}); "
                    f"sleeping {backoff:.1f}s"
                )
            else:
                print(
                    f"     Load failed "
                    f"(attempt {attempt}/{MAX_LOAD_RETRIES}): "
                    f"{e}; retrying in {backoff:.1f}s"
                )
            time.sleep(backoff)

    if session is None:
        print(
            f"     Failed to create session: {last_error}"
        )
        _write_skip_marker(
            year, round_number, event_name,
            f"session_none: {last_error}",
        )
        return None

    # ── Check for wet conditions ──────────────────────────────
    if is_wet_race(session, year, event_name):
        print(f"     Skipping (wet race)")
        _write_skip_marker(
            year, round_number, event_name, "wet_race"
        )
        return None

    # ── Get lap data ──────────────────────────────────────────
    laps = session.laps
    if laps is None or laps.empty:
        print(f"     No lap data")
        _write_skip_marker(
            year, round_number, event_name, "no_lap_data"
        )
        return None
    
    # try to recompute Lapstartdate
    # NOTE this is the fix for the weatherdata issue.
    laps["LapStartDate"] = session.date + laps["Time"] - laps["LapTime"]

    # ── Extract location ──────────────────────────────────────
    location = ""
    try:
        location = session.event["Location"]
    except Exception:
        try:
            location = session.event.Location
        except Exception:
            location = event_name

    # ── Total race laps ───────────────────────────────────────
    total_laps = 0
    try:
        total_laps = session.total_laps
    except Exception:
        pass
    if not total_laps or total_laps == 0:
        total_laps = int(laps["LapNumber"].max())

    # ── Filter laps ───────────────────────────────────────────
    laps_filtered = filter_laps(laps)

    if laps_filtered.empty:
        print(f"     No laps remaining after filtering")
        _write_skip_marker(
            year, round_number, event_name,
            "no_laps_after_filter",
        )
        return None

    # ── Engineer features ─────────────────────────────────────
    result = engineer_features(
        laps_filtered,
        session=session,
        location=location,
        total_laps=total_laps,
        year=year,
        event_name=event_name,
        verbose=verbose,
    )

    # ── Select only output columns ────────────────────────────
    # This ensures the race-level cache doesn't contain the
    # ~31 raw FastF1 columns that are no longer needed.
    cols_to_keep = [
        c for c in OUTPUT_COLUMNS if c in result.columns
    ]
    result = result[cols_to_keep].copy()

    return result


# ═══════════════════════════════════════════════════════════════
# 9. COLUMN SELECTION & FINAL CLEANUP
# ═══════════════════════════════════════════════════════════════

# Columns to keep in the final output.
# NOTE: FuelLoad is deliberately absent — it will be
# calculated and appended in a separate pipeline step.
OUTPUT_COLUMNS = [
    # ── Target ──
    "LapTimeSec",

    # ── Categorical Features (for CatBoost) ──
    "Compound",
    "TrackName",
    "Driver",
    "Team",

    # ── Numerical Features ──
    "TireAge",
    "TireAgeSq",
    "RaceLapNumber",
    "TrackTemp",
    "AirTemp",
    "Humidity",
    "WindSpeed",
    "GapToCarAhead",
    "DRS_Available",
    "CircuitLength",
    "NumberOfCorners",

    # ── Metadata (not model inputs) ──
    "RaceID",
    "LapNumber",
    "PitLossSec",
    "TotalRaceLaps",
]


def select_and_clean(df: pd.DataFrame) -> pd.DataFrame:
    """
    Select only the columns we need and do final cleanup.
    """
    keep = [c for c in OUTPUT_COLUMNS if c in df.columns]
    missing = [c for c in OUTPUT_COLUMNS if c not in df.columns]
    if missing:
        print(
            f"\nWarning: missing columns (will be NaN): {missing}"
        )
        for col in missing:
            df[col] = np.nan
        keep = OUTPUT_COLUMNS

    df = df[keep].copy()

    # ── Fill remaining weather NaN with column medians ────────
    weather_cols = [
        "TrackTemp", "AirTemp", "Humidity", "WindSpeed"
    ]
    for col in weather_cols:
        if col in df.columns:
            median_val = df[col].median()
            if pd.notna(median_val):
                df[col] = df[col].fillna(median_val)
            else:
                defaults = {
                    "TrackTemp": 35.0,
                    "AirTemp": 28.0,
                    "Humidity": 50.0,
                    "WindSpeed": 2.0,
                }
                df[col] = df[col].fillna(
                    defaults.get(col, 0)
                )

    # ── Ensure correct dtypes ─────────────────────────────────
    dtype_map = {
        "LapTimeSec":      "float64",
        "TireAge":         "int32",
        "TireAgeSq":       "int32",
        "RaceLapNumber":   "int32",
        "TrackTemp":       "float32",
        "AirTemp":         "float32",
        "Humidity":        "float32",
        "WindSpeed":       "float32",
        "GapToCarAhead":   "float32",
        "DRS_Available":   "int8",
        "CircuitLength":   "float32",
        "NumberOfCorners": "int16",
        "LapNumber":       "int32",
        "PitLossSec":      "float32",
        "TotalRaceLaps":   "int16",
        "Compound":        "str",
        "TrackName":       "str",
        "Driver":          "str",
        "Team":            "str",
        "RaceID":          "str",
    }
    for col, dtype in dtype_map.items():
        if col in df.columns:
            try:
                if dtype == "str":
                    df[col] = df[col].astype(str)
                else:
                    df[col] = pd.to_numeric(
                        df[col], errors="coerce"
                    ).astype(np.dtype(dtype))
            except (ValueError, TypeError):
                pass

    # ── Drop any rows where the target is still NaN ───────────
    df = df.dropna(subset=["LapTimeSec"])

    return df


# ═══════════════════════════════════════════════════════════════
# 10. MAIN PIPELINE
# ═══════════════════════════════════════════════════════════════

def build_dataset(
    seasons: Optional[List[int]] = None,
    output_path: str = "training_data",
    verbose: bool = False,
) -> pd.DataFrame:
    """
    Run the complete data pipeline across all specified seasons.

    For each race:
      1. Check if it was previously skipped → skip again
      2. Check if it was previously processed → load from cache
      3. Otherwise load from API, process, and cache
      4. Skip sprint weekends entirely

    Parameters
    ----------
    seasons : list of int
        Years to process. Defaults to SEASONS config.
    output_path : str
        Base filename (without extension) for the output files.
        A .parquet file will be created.

    Returns
    -------
    pd.DataFrame
        Complete, clean, feature-rich training dataset.
    """
    if seasons is None:
        seasons = SEASONS

    all_frames = []
    race_summary = []

    for year in seasons:
        print(f"\n{'=' * 60}")
        print(f"  SEASON {year}")
        print(f"{'=' * 60}")

        try:
            _throttle_fastf1_requests()
            schedule = fastf1.get_event_schedule(
                year, include_testing=False
            )
        except Exception as e:
            print(f"  Failed to load schedule: {e}")
            continue

        for _, event_row in schedule.iterrows():
            event_name = event_row.get("EventName", "Unknown")
            round_number = event_row.get("RoundNumber", 0)

            if round_number == 0:
                continue

            # ── Skip sprint weekends ──────────────────────────
            if is_sprint_weekend(event_row):
                event_format = str(
                    event_row.get("EventFormat", "sprint")
                ).strip()
                print(
                    f"\n  Round {round_number}: {event_name}"
                )
                print(
                    f"     Skipping (sprint weekend: "
                    f"{event_format})"
                )
                race_summary.append({
                    "Year": year,
                    "Event": event_name,
                    "Laps": 0,
                    "Drivers": 0,
                    "Status": "Sprint",
                })
                continue

            print(f"\n  Round {round_number}: {event_name}")

            # ── Check skip marker ─────────────────────────────
            skip_reason = _read_skip_marker(
                year, round_number, event_name
            )
            if skip_reason is not None:
                print(
                    f"     Previously skipped: {skip_reason}"
                )
                race_summary.append({
                    "Year": year,
                    "Event": event_name,
                    "Laps": 0,
                    "Drivers": 0,
                    "Status": f"Skipped: {skip_reason}",
                })
                continue

            # ── Check processed race cache ────────────────────
            race_cache_file = _race_cache_path(
                year, round_number, event_name
            )
            result = None

            if race_cache_file.exists():
                try:
                    result = pd.read_parquet(race_cache_file)
                    print(
                        f"     Loaded from cache: "
                        f"{race_cache_file.name}"
                    )
                except Exception as e:
                    print(
                        f"     Cache read failed ({e}); "
                        f"reprocessing"
                    )
                    result = None

            # ── Process from API if not cached ────────────────
            if result is None:
                result = process_session(
                    year, event_name, round_number, verbose=verbose
                )

                # Cache successfully processed races
                if result is not None and not result.empty:
                    try:
                        result.to_parquet(
                            race_cache_file, index=False
                        )
                        print(
                            f"     Cached: "
                            f"{race_cache_file.name}"
                        )
                    except Exception as e:
                        print(
                            f"     Could not cache: {e}"
                        )

            # ── Collect results ────────────────────────────────
            if result is not None and not result.empty:
                all_frames.append(result)
                n_laps = len(result)
                n_drivers = result["Driver"].nunique()
                race_summary.append({
                    "Year": year,
                    "Event": event_name,
                    "Laps": n_laps,
                    "Drivers": n_drivers,
                    "Status": "OK",
                })
                print(
                    f"     Collected {n_laps} laps from "
                    f"{n_drivers} drivers"
                )
            else:
                race_summary.append({
                    "Year": year,
                    "Event": event_name,
                    "Laps": 0,
                    "Drivers": 0,
                    "Status": "Excluded",
                })

    # ── Combine All Races ─────────────────────────────────────
    if not all_frames:
        print(
            "\nNo data collected. "
            "Check FastF1 cache and network."
        )
        sys.exit(1)

    print(f"\n{'=' * 60}")
    print(f"  COMBINING & CLEANING")
    print(f"{'=' * 60}")

    dataset = pd.concat(all_frames, ignore_index=True)
    print(f"  Raw combined: {len(dataset)} laps")

    dataset = select_and_clean(dataset)
    print(f"  After cleanup: {len(dataset)} laps")

    # ── Save Parquet ──────────────────────────────────────────
    parquet_path = f"{output_path}.parquet"
    dataset.to_parquet(parquet_path, index=False)
    print(f"\n  Saved to: {parquet_path}")
    # also going to save to a csv for easier inspection. 
    csv_path = f"{output_path}.csv"
    dataset.to_csv(csv_path, index=False)
    print(f"  Also saved to: {csv_path}")

    # ── Summary Statistics ────────────────────────────────────
    print(f"\n{'=' * 60}")
    print(f"  DATASET SUMMARY")
    print(f"{'=' * 60}")
    print(f"  Total laps:    {len(dataset):,}")
    print(f"  Total races:   {dataset['RaceID'].nunique()}")
    print(
        f"  Seasons:       "
        f"{sorted(dataset['RaceID'].str[:4].unique())}"
    )
    print(f"  Drivers:       {dataset['Driver'].nunique()}")
    print(f"  Teams:         {dataset['Team'].nunique()}")
    print(f"  Tracks:        {dataset['TrackName'].nunique()}")
    print(
        f"  Compounds:     "
        f"{sorted(dataset['Compound'].unique())}"
    )
    print(
        f"  Lap time range: "
        f"{dataset['LapTimeSec'].min():.1f}s – "
        f"{dataset['LapTimeSec'].max():.1f}s"
    )
    print(
        f"  Median lap:    "
        f"{dataset['LapTimeSec'].median():.1f}s"
    )

    # ── Feature Completeness ──────────────────────────────────
    print(f"\n  Feature completeness:")
    for col in OUTPUT_COLUMNS:
        if col in dataset.columns:
            pct = dataset[col].notna().mean() * 100
            indicator = (
                "OK" if pct > 95
                else "WARN" if pct > 50
                else "MISSING"
            )
            print(f"    {indicator} {col:25s} {pct:5.1f}%")

    # ── Per-Race Breakdown ────────────────────────────────────
    print(f"\n  Per-race breakdown:")
    summary_df = pd.DataFrame(race_summary)
    print(summary_df.to_string(index=False))

    return dataset


# ═══════════════════════════════════════════════════════════════
# 11. CLI ENTRY POINT
# ═══════════════════════════════════════════════════════════════

if __name__ == "__main__":
    print("F1 Virtual Race Strategist - Data Pipeline")
    print(f"   Seasons: {SEASONS}")
    print(f"   Output:  training_data.parquet")
    print(f"   Note:    FuelLoad will be added separately")
    print(f"   Note:    Sprint weekends excluded")
    print()

    df = build_dataset(
        seasons=SEASONS,
        output_path=OUTPUT_PATH
    )

    # ── Sanity Checks ─────────────────────────────────────────
    print(f"\n{'=' * 60}")
    print(f"  SANITY CHECKS")
    print(f"{'=' * 60}")


    # No wet compounds
    wet_leaks = df["Compound"].isin(WET_COMPOUNDS).sum()
    print(
        f"  Wet compound laps: {wet_leaks} "
        f"{'OK' if wet_leaks == 0 else 'LEAK'}"
    )

    # No lap 1s
    lap1_leaks = (df["LapNumber"] <= 1).sum()
    print(
        f"  Lap 1 entries:     {lap1_leaks} "
        f"{'OK' if lap1_leaks == 0 else 'LEAK'}"
    )

    # Reasonable lap times (55–140 seconds for F1)
    unreasonable = (
        (df["LapTimeSec"] < 55) | (df["LapTimeSec"] > 140)
    ).sum()
    print(
        f"  Unreasonable times: {unreasonable} "
        f"{'OK' if unreasonable == 0 else 'WARN'}"
    )

    # No negative tire ages
    neg_tire_age = (df["TireAge"] < 0).sum()
    print(
        f"  Negative TireAge:  {neg_tire_age} "
        f"{'OK' if neg_tire_age == 0 else 'FAIL'}"
    )

    # FuelLoad should NOT be present
    has_fuel = "FuelLoad" in df.columns
    print(
        f"  FuelLoad absent:   "
        f"{'OK (will be added later)' if not has_fuel else 'FAIL: should not be here'}"
    )

    print(
        f"\nPipeline complete. Dataset ready for "
        f"fuel model attachment and model training."
    )