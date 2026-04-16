# WIP script to produce data exploration plots

from pathlib import Path
import numpy as np
import pandas as pd
import fastf1
import seaborn as sns
import matplotlib.pyplot as plt

BASE_DIR = Path(__file__).resolve().parent.parent

data_path = BASE_DIR / "data" / "processed" / "training_data.csv"
output_dir = BASE_DIR / "figures" / "data_exploration"

output_dir.mkdir(parents=True, exist_ok=True)

# load data into df
df = pd.read_csv(data_path)

print(df.columns)

# plot styling
COLORS = {
    'background': '#1A1A1A',
    'axes':       '#2C2C2C',
    'grid':       '#444444',
    'text':       '#FFFFFF',
    'text_muted': '#AAAAAA',
    'soft':       '#E10600',
    'medium':     '#4A90D9',
    'hard':       '#00B4A6',
    'secondary':  '#AAAAAA',
}

COMPOUND_COLORS = {
    "SOFT":   COLORS['soft'],
    "MEDIUM": COLORS['medium'],
    "HARD":   COLORS['hard'],
}

def apply_style(ax, title=None):
    """Apply the dark colour scheme to a given axes object."""
    ax.set_facecolor(COLORS['axes'])
    ax.figure.patch.set_facecolor(COLORS['background'])
    ax.tick_params(colors=COLORS['text'])
    ax.xaxis.label.set_color(COLORS['text'])
    ax.yaxis.label.set_color(COLORS['text'])
    ax.spines[:].set_edgecolor(COLORS['grid'])
    ax.grid(color=COLORS['grid'], linestyle='--', linewidth=0.5)
    if title:
        ax.set_title(title, color=COLORS['text'])
    legend = ax.get_legend()
    if legend:
        legend.get_frame().set_facecolor(COLORS['axes'])
        for text in legend.get_texts():
            text.set_color(COLORS['text'])


# Plot lap time against tire age with regression line
fig, ax = plt.subplots()
for compound, grp in df.groupby("Compound"):
    sns.regplot(data=grp, x="TireAge", y="LapTimeSec",
                ax=ax, order=2, scatter_kws={"alpha": 0.1, "s": 10},
                label=compound, color=COMPOUND_COLORS.get(compound))
ax.set_xlabel("Tyre Age")
ax.set_ylabel("Lap Time (s)")
ax.legend()
apply_style(ax, title="Lap Time vs Tyre Age")
plt.tight_layout()
plt.show()

# Plot lap time against tire age for different tracks (compound separated)
fig, axes = plt.subplots(2, 2, figsize=(12, 8))
fig.patch.set_facecolor(COLORS['background'])
for ax, (track, grp) in zip(axes.flat, df.groupby("TrackName")):
    for compound, cgrp in grp.groupby("Compound"):
        cgrp.groupby("TireAge")["LapTimeSec"].median().plot(
            ax=ax, label=compound, color=COMPOUND_COLORS.get(compound)
        )
    ax.set_xlabel("Tyre Age")
    ax.set_ylabel("Median Lap Time (s)")
    ax.legend(fontsize=7)
    apply_style(ax, title=track)
plt.tight_layout()
plt.show()

# Plot correlation heatmap
corr = df.select_dtypes(include=np.number).corr()
fig, ax = plt.subplots(figsize=(8, 10))
fig.patch.set_facecolor(COLORS['background'])
sns.heatmap(corr[["LapTimeSec"]].sort_values("LapTimeSec"),
            annot=True, fmt=".2f", vmin=-1, vmax=1, cmap="coolwarm",
            ax=ax,
            annot_kws={"color": COLORS['text']},
            linecolor=COLORS['grid'], linewidths=0.5)
ax.set_facecolor(COLORS['axes'])
ax.tick_params(colors=COLORS['text'])
ax.xaxis.label.set_color(COLORS['text'])
ax.yaxis.label.set_color(COLORS['text'])
ax.set_title("Feature Correlations with Lap Time", color=COLORS['text'])
plt.xticks(rotation=45, ha="right", color=COLORS['text'])
plt.yticks(rotation=0, color=COLORS['text'])
# style the colourbar
cbar = ax.collections[0].colorbar
cbar.ax.tick_params(colors=COLORS['text'])
cbar.ax.yaxis.label.set_color(COLORS['text'])
plt.tight_layout()
plt.show()

'''
# plot gaptocar against laptime
data = df[["LapTimeSec", "GapToCarAhead"]].dropna()
data = data[data["GapToCarAhead"] != 10]
plt.figure()
plt.scatter(x=data["LapTimeSec"], y=data["GapToCarAhead"])
plt.tight_layout()
plt.show()
'''

# Boxplot of laptime by compund
fig, axes = plt.subplots(2, 2, figsize=(12, 8))
fig.patch.set_facecolor(COLORS['background'])
for ax, (track, grp) in zip(axes.flat, df.groupby("TrackName")):
    sns.boxplot(data=grp, x="Compound", y="LapTimeSec",
                palette=COMPOUND_COLORS, ax=ax,
                order=["SOFT", "MEDIUM", "HARD"],
                flierprops={"markerfacecolor": COLORS['text_muted'], "markersize": 2})
    apply_style(ax, title=track)
    ax.set_xlabel("Compound")
    ax.set_ylabel("Lap Time (s)")
plt.tight_layout()
plt.show()

# track temp vs laptime
fig, ax = plt.subplots(figsize=(8, 5))
for compound, grp in df.groupby("Compound"):
    ax.scatter(grp["TrackTemp"], grp["LapTimeSec"],
               alpha=0.1, s=8, label=compound,
               color=COMPOUND_COLORS.get(compound))
# single regression line across all compounds
sns.regplot(data=df, x="TrackTemp", y="LapTimeSec",
            ax=ax, scatter=False, order=2,
            line_kws={"color": COLORS['text_muted'], "linewidth": 1.5, "linestyle": "--"})
ax.set_xlabel("Track Temperature (°C)")
ax.set_ylabel("Lap Time (s)")
ax.legend()
apply_style(ax, title="Track Temp vs Lap Time")
plt.tight_layout()
plt.show()

# DRS vs gap
DRS_COLORS = {0: COLORS['secondary'], 1: COLORS['medium']}
DRS_LABELS = {0: "No DRS", 1: "DRS Available"}

drs_data = df[["LapTimeSec", "GapToCarAhead", "DRS_Available"]].dropna()
drs_data = drs_data[drs_data["GapToCarAhead"] < 5]   # trim outliers

fig, ax = plt.subplots(figsize=(8, 5))
for drs_val, grp in drs_data.groupby("DRS_Available"):
    ax.scatter(grp["GapToCarAhead"], grp["LapTimeSec"],
               alpha=0.1, s=8,
               color=DRS_COLORS[drs_val],
               label=DRS_LABELS[drs_val])
    sns.regplot(data=grp, x="GapToCarAhead", y="LapTimeSec",
                ax=ax, scatter=False, order=2,
                line_kws={"color": DRS_COLORS[drs_val], "linewidth": 1.5})
ax.axvline(x=1.0, color=COLORS['soft'], linewidth=1,
           linestyle="--", label="DRS detection threshold (1s)")
ax.set_xlabel("Gap to Car Ahead (s)")
ax.set_ylabel("Lap Time (s)")
ax.legend()
apply_style(ax, title="Gap to Car Ahead vs Lap Time by DRS Availability")
plt.tight_layout()
plt.show()