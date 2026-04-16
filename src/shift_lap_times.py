'''
Will Buesnel, Apr 2026

This script shifts the dataset, so all the features allign the next lap time (ie the one that they are supposed to predict).
This is necessary because the features are currently alligned to the lap they were observed on, so we would be predicting the lap time knowing everything about it.
This is a bit of a hack, but it is much easier than trying to re-engineer the feature extraction to align to the next lap.

I was thinking about removing the F_z feature, since it is essentially a linear transformation of lap speed, but if its predicting the next lap I actually think it could stay.
for now though, I will take it out and see how the model performs.
'''

import pandas as pd
import numpy as np
import re
from pathlib import Path


# define constants
BASE_DIR = Path(__file__).resolve().parents[1]
INPUT_PARQUET_PATH =  BASE_DIR / "data" / "processed" / "training_data_with_physics.parquet"
OUTPUT_CSV_PATH = BASE_DIR / "data" / "processed" / "training_data_with_physics_shifted.csv"
OUTPUT_PARQUET_PATH = BASE_DIR / "data" / "processed" / "training_data_with_physics_shifted.parquet"


if __name__ == "__main__":
    print("Reading data from:", INPUT_PARQUET_PATH)
    df = pd.read_parquet(INPUT_PARQUET_PATH)
    initial_df = df.copy() # for debugging
    print("Shifting data to align features with next lap time...")
    df = df.sort_values(["RaceID", "Driver", "LapNumber"]).reset_index(drop=True)
    df["LapTimeSec"] = df.groupby(["RaceID", "Driver"])["LapTimeSec"].shift(-1)
    df["TyreHealth"] = df.groupby(["RaceID", "Driver"])["TyreHealth"].shift(-1)
    df["TyreTemp_C"] = df.groupby(["RaceID", "Driver"])["TyreTemp_C"].shift(-1)
    df["Psi_T"] = df.groupby(["RaceID", "Driver"])["Psi_T"].shift(-1)
    df["SlidingProxy"] = df.groupby(["RaceID", "Driver"])["SlidingProxy"].shift(-1)
    df["Fz_N"] = df.groupby(["RaceID", "Driver"])["Fz_N"].shift(-1)
    print("Dropping rows with NaN values (last lap of each stint)...")
    df = df.dropna(subset=["LapTimeSec", "TyreHealth", "TyreTemp_C", "Psi_T", "SlidingProxy", "Fz_N"])
    print("Saving shifted data to:", OUTPUT_PARQUET_PATH)
    df.to_parquet(OUTPUT_PARQUET_PATH, index=False)
    print("Done.")

    # check that the shifting worked correctly for a few random samples
    print("\nChecking shifted data against initial data for random samples...")
    for _ in range(2):
        index = np.random.randint(0, len(df))
        sample = df.iloc[index]
        race_id = sample["RaceID"]
        driver = sample["Driver"]
        lap_number = sample["LapNumber"]
        initial_sample = initial_df[
            (initial_df["RaceID"] == race_id) &
            (initial_df["Driver"] == driver) &
            (initial_df["LapNumber"] == lap_number)
        ]        
        if not initial_sample.empty:
            initial_sample = initial_sample.iloc[0]
            print(f"\nSample index: {index}")
            print("Shifted sample:")
            print(sample[["RaceID", "Driver", "LapNumber", "LapTimeSec", "TyreHealth", "TyreTemp_C", "Psi_T", "SlidingProxy", "Fz_N"]])
            print("Initial sample (should be the previous lap):")
            print(initial_sample[["RaceID", "Driver", "LapNumber", "LapTimeSec", "TyreHealth", "TyreTemp_C", "Psi_T", "SlidingProxy", "Fz_N"]])
        else:
            print(f"\nSample index: {index} - No matching initial sample found (this is expected for the last lap of each stint).") 
        initialsample = initial_df.sample(1).iloc[0]

