# animations
# animates a ghost car moving at a **constant** speed around the track to match the pace predicted. 
# it does not follow a telemetry trace so will not slow/ speed up, but completes the track in the predicted time.

# ================================================================
# this assumes that the model outputs a predicted time as a string (or anything subscriptable) in the format 'X:YY.ZZZ'
# adjust parse_time(time) inside animate_ghost_comparison() if the format is different
# ================================================================

# call/ import animate_ghost_comparison()
# example: animate_ghost_comparison(session, driver='PER', lap_number=20, pred_time='1:22.029', show_data=True, save_path=None, interval=20)

# set cache location - folder must exist before running
cache_location = "fastf1_cache/"

# this file needs to exist in the same directory as lap_animations
from lap_animations import get_data

import matplotlib.pyplot as plt
import matplotlib.animation as animation
import numpy as np
import fastf1

def animate_ghost_comparison(session, driver, lap_number, pred_time, 
                             show_data=True, save_path=None, interval=20):
    """
    animate ghost car moving at average speed to match predicted pace
    """

    # helper function to turn time string into seconds
    def parse_time(time_str):
        mins = int(time_str[0])
        secs = float(time_str[2:])
        return mins*60 + secs

    pred_secs = parse_time(pred_time)
    print(f"Predicted lap time in seconds: {pred_secs}")
    actual_data = get_data(session, driver, lap_number)
    actual_time = actual_data['time'][-1]

    time_diff = pred_secs - actual_time
    ghost_fin_first = pred_secs < actual_time

    plt.style.use('dark_background')
    fig, ax = plt.subplots(figsize=(12, 8))

    ax.plot(actual_data['track_x'], actual_data['track_y'], color="#444444", linewidth=8)
    ax.plot(actual_data['x'], actual_data['y'], color=actual_data['colour'], linewidth=1, alpha=0.2)

    actual_car = ax.scatter([], [], s=150, c=actual_data['colour'], 
                            edgecolors='white', zorder=5, label='Actual')
    actual_trail, = ax.plot([], [], color=actual_data['colour'], linewidth=2)

    ghost_car = ax.scatter([], [], s=150, c='#8888FF', 
                           edgecolors='white', zorder=4, label=f'Predicted ({pred_secs:.2f}s)', alpha=0.7)
    ghost_trail, = ax.plot([], [], color='#8888FF', linewidth=2, alpha=0.5)

    info_text = ax.text(0.7, 0.3, '', transform=ax.transAxes, fontsize=10, 
                        verticalalignment='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.5))
    info_text.set_visible(show_data)

    ax.legend(loc='upper right', fontsize=9, framealpha=0.5)
    ax.set_aspect('equal')
    ax.axis('off')
    ax.set_title(f"{actual_data['track']} | {driver} | Lap {lap_number}", color='white', pad=20)

    fps = 30
    total_frames = int(max(actual_time, pred_secs) * fps)

    actual_fin = False
    ghost_fin = False
    first_fin_time = None

    def init():
        actual_car.set_offsets(np.empty((0, 2)))
        actual_trail.set_data([], [])
        ghost_car.set_offsets(np.empty((0, 2)))
        ghost_trail.set_data([], [])
        if show_data:
            info_text.set_text('')
        return actual_car, actual_trail, ghost_car, ghost_trail, info_text

    def update(frame):
        nonlocal actual_fin, ghost_fin, first_fin_time
        current_time = frame / fps

        if current_time <= actual_time:
            idx = min(int(current_time * 30), len(actual_data['time']) - 1)
            actual_car.set_offsets([[actual_data['x'][idx], actual_data['y'][idx]]])
            actual_trail.set_data(actual_data['x'][:idx+1], actual_data['y'][:idx+1])
        else:
            actual_car.set_offsets([[actual_data['x'][-1], actual_data['y'][-1]]])
            if not actual_fin:
                actual_fin = True
                if first_fin_time is None:
                    first_fin_time = actual_time

        if current_time <= pred_secs:
            progress = current_time / pred_secs
            ghost_idx = min(int(progress * (len(actual_data['x']) - 1)), len(actual_data['x']) - 1)
            ghost_car.set_offsets([[actual_data['x'][ghost_idx], actual_data['y'][ghost_idx]]])
            ghost_trail.set_data(actual_data['x'][:ghost_idx+1], actual_data['y'][:ghost_idx+1])
        else:
            ghost_car.set_offsets([[actual_data['x'][-1], actual_data['y'][-1]]])
            if not ghost_fin:
                ghost_fin = True
                if first_fin_time is None:
                    first_fin_time = pred_secs

        if show_data:
            idx = min(int(current_time * 30), len(actual_data['time']) - 1)
            speed = actual_data['speed'][idx]
            gear = actual_data['gear'][idx]
            throttle = actual_data['throttle'][idx]
            brake = actual_data['brake'][idx]
            pedal = "Throttle" if throttle > 10 else ("Braking" if brake > 10 else "Coasting")

            if actual_fin or ghost_fin:
                # one car finished - show delta information
                elapsed_since_first_finish = current_time - first_fin_time

                if ghost_fin_first:
                    if actual_fin:
                        # both done - show final delta
                        delta_str = f"Time Delta: +{abs(time_diff):.3f}s"
                    else:
                        # actual still running - count up + show telemetry
                        delta_str = f"Time Delta: -{elapsed_since_first_finish:.3f}s"
                else:
                    # actual finished first
                    if ghost_fin:
                        # both done - show final delta
                        delta_str = f"Time Delta: +{abs(time_diff):.3f}s"
                    else:
                        # ghost still running - count up
                        delta_str = f"Time Delta: +{elapsed_since_first_finish:.3f}s"

                # show everything
                info_str = (f"{driver}\n"
                            f"Actual: {actual_time:.3f}s\n"
                            f"Predicted: {pred_secs:.3f}s\n"
                            f"{delta_str}\n"
                            f"─────────────\n"
                            f"Speed: {speed:.0f} km/h\n"
                            f"Gear: {gear}\n"
                            f"Status: {pedal}")
            else:
                # during lap - show telemetry only
                info_str = (f"{driver}\n"
                            f"Speed: {speed:.0f} km/h\n"
                            f"Gear: {gear}\n"
                            f"Status: {pedal}")

            info_text.set_text(info_str)

        return actual_car, actual_trail, ghost_car, ghost_trail, info_text
    
    print("Generating ghost comparison animation...")
    print(f"Actual duration: {actual_time:.3f}s | Predicted: {pred_secs:.3f}s")
    if ghost_fin_first:
        print(f"Ghost finishes first by {abs(time_diff):.3f}s")
    else:
        print(f"Actual finishes first by {abs(time_diff):.3f}s")
    
    anim = animation.FuncAnimation(fig, update, frames=total_frames, init_func=init, blit=True, interval=interval, repeat=False)

    if save_path:
        anim.save(save_path, writer='ffmpeg', fps=30)
    else:
        plt.show()

    return anim

if __name__ == "__main__":
    fastf1.Cache.enable_cache(cache_location)

    year = 2022
    event = 20
    session_type = 'R' 

    try:
        session = fastf1.get_session(year, event, session_type)
        session.load()

        animate_ghost_comparison(session, driver='HAM', lap_number=17, pred_time='1:16.029', 
                                 show_data=True, save_path=None, interval=20)

    except Exception as e:
        print(f"An error occurred: {e}")