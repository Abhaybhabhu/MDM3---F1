# animations
# animates the tyre health bar. call animate_tyre_health(df, driver='GAS', race_id='2023_Japanese Grand Prix', save_path=None)
# 'health' thresholds colours can be changed if required - currently set to >0.90:green, >0.80:amber, else:red 

import matplotlib.pyplot as plt
import matplotlib.animation as animation
import matplotlib.patches as patches
import pandas as pd

def animate_tyre_health(df, driver, race_id, save_path=None, interval=500):
    """
    create health bar animation for a given Driver and RaceID
    """
    valid_drivers = set(df['Driver'].unique())
    if driver not in valid_drivers:
        raise ValueError(
            f"Invalid driver: '{driver}'.\n"
            f"Valid options in dataset: {sorted(valid_drivers)}"
        )
    valid_race_ids = set(df['RaceID'].unique())
    if race_id not in valid_race_ids:
        raise ValueError(
            f"Invalid RaceID: '{race_id}'.\n"
            f"Valid options in dataset: {sorted(valid_race_ids)}"
        )

    driver_race_df = df[(df['Driver'] == driver) & (df['RaceID'] == race_id)].copy()
    driver_race_df = driver_race_df.sort_values('RaceLapNumber').reset_index(drop=True)

    tyre_health = driver_race_df['TyreHealth'].values
    laps = driver_race_df['RaceLapNumber'].values
    compounds = driver_race_df['Compound'].values

    tyre_hist = []

    fig, ax = plt.subplots(figsize=(9, 3.5))
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis('off')
    fig.patch.set_facecolor('#1F1F1F')

    bg_bar = patches.Rectangle((0.05, 0.4), 0.9, 0.25, facecolor='#333333', edgecolor='#555555', linewidth=2, clip_on=False)
    ax.add_patch(bg_bar)

    fg_bar = patches.Rectangle((0.05, 0.4), 0, 0.25, facecolor='#00FF55', edgecolor='#FFFFFF', linewidth=1.5)
    ax.add_patch(fg_bar)

    title_text = ax.text(0.5, 0.85, f"{driver.upper()} | {race_id}", ha='center', va='center',
                         fontsize=14, fontweight='bold', color='#FFFFFF')
    status_text = ax.text(0.5, 0.15, "", ha='center', va='center', fontsize=13, color='#AAAAAA', fontfamily='monospace')
    history_text = ax.text(0.5, 0.05, "", ha='center', va='center', fontsize=11, color='#AAAAAA', fontfamily='monospace')

    def update(frame):
        val = tyre_health[frame]
        compound = compounds[frame]

        if frame > 0 and compound != compounds[frame - 1]:
            prev_compound = compounds[frame - 1]
            prev_final_health = tyre_health[frame - 1]
            prev_lap = int(laps[frame - 1])
            tyre_hist.append({
                'lap': prev_lap,
                'compound': prev_compound,
                'health': prev_final_health
            })

        if frame == len(tyre_health) - 1:
            curr_compound = compound
            curr_final_health = val
            curr_lap = int(laps[frame])
            tyre_hist.append({
                'lap': curr_lap,
                'compound': curr_compound,
                'health': curr_final_health
            })

        fg_bar.set_width(val * 0.9)

        if val > 0.90:
            fg_bar.set_facecolor('#00FF55')
        elif val > 0.80:
            fg_bar.set_facecolor('#FFAA00')
        else:
            fg_bar.set_facecolor('#FF3333')

        status_text.set_text(f"LAP {int(laps[frame]):02d}  |  {compound}  |  HEALTH: {val:.2%}")

        history_text.set_text("")
        for t in tyre_hist:
            history_text.set_text(history_text.get_text() + f"Lap {t['lap']:02d} {t['compound']} {t['health']:.2%}\n")

        return fg_bar, status_text, title_text, history_text

    def init():
        fg_bar.set_width(0)
        status_text.set_text("")
        history_text.set_text("")
        return fg_bar, status_text, title_text, history_text

    ani = animation.FuncAnimation(fig, update, frames=len(tyre_health),
                                  init_func=init, interval=interval, blit=True, repeat=False)

    if save_path:
        ani.save(save_path, writer='ffmpeg', fps=30)
        plt.close(fig)
        print(f"Animation saved to: {save_path}")
    else:
        plt.tight_layout()
        plt.show()

if __name__== "__main__":
    df = pd.read_csv("data/processed/training_data_with_physics.csv")
    animate_tyre_health(df, driver='RIC', race_id='2024_Spanish Grand Prix', save_path=None)
