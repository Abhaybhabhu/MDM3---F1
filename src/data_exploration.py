# WIP script to produce data exploration plots

from pathlib import Path
import numpy as np
import pandas as pd
import fastf1
import seaborn as sns
import matplotlib.pyplot as plt

BASE_DIR = Path(__file__).resolve().parent.parent

data_path = BASE_DIR / "data" / "processed" / "training_data.parquet"
output_dir = BASE_DIR / "figures" / "data_exploration"

output_dir.mkdir(parents=True, exist_ok=True)

# load data into df
df = pd.read_parquet(data_path)

# plot styling
plt.style.use("dark_background")
COMPOUND_COLORS = {"SOFT": "#E8002D", "MEDIUM": "#FFF200", "HARD": "#C8C8C8"}

# Plot lap time against tire age with regression line
for compound, grp in df.groupby("Compound"):
    sns.regplot(data=grp, x="TireAge", y="LapTimeSec",
                order=2, scatter_kws={"alpha": 0.1, "s": 10},
                label=compound, color=COMPOUND_COLORS.get(compound))
plt.tight_layout()
plt.legend()
plt.show()

# PLot lap time against tire age for different laps (compound seperated)
fig, axes = plt.subplots(2, 2, figsize=(12, 8))
for ax, (track, grp) in zip(axes.flat, df.groupby("TrackName")):
    for compound, cgrp in grp.groupby("Compound"):
        cgrp.groupby("TireAge")["LapTimeSec"].median().plot(
            ax=ax, label=compound, color=COMPOUND_COLORS.get(compound)
        )
    ax.set_title(track)
    ax.set_xlabel("Tyre Age")
    ax.set_ylabel("Median Lap Time (s)")
    ax.legend(fontsize=7)
plt.tight_layout()
plt.show()

# plot correlation heatmap between
corr = df.select_dtypes(include=np.number).corr()
plt.figure(figsize=(8, 10))
sns.heatmap(corr[["LapTimeSec"]].sort_values("LapTimeSec"),
            annot=True, fmt=".2f", vmin=-1, vmax=1, cmap="coolwarm")
plt.xticks(rotation=45, ha="right")
plt.yticks(rotation=0)
plt.tight_layout()
plt.show()

# issues with weather data
print(df.groupby("RaceID")[["TrackTemp", "AirTemp", "Humidity", "WindSpeed"]].mean())



