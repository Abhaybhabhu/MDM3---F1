# animations
# single lap telemetry trace

# set cache location - folder must exist before running
cache_location = "fastf1_cache/"

# usage:
# import this file and call 
# animate_single_driver_lap(session, driver='VER', year=year, lap_number=5, show_data=True, save_path='anims/VER_IMOLA_2025.mp4', interval=10)
# or call animate_multiple_drivers_lap(session, drivers=['VER', 'HAM', 'LEC'], lap_number=14, show_data=False, save_path=None)

import fastf1
import matplotlib.pyplot as plt
plt.style.use("dark_background")
import matplotlib.animation as animation
import numpy as np

def get_colour(team):
    """
    gets a colour based on team
    """
    print(f"got team name: {team}")
    TEAM_COLORS = {
        'Red Bull Racing': '#3671C7',
        'Mercedes': '#27F4D2',
        'Ferrari': '#E80020',
        'McLaren': '#FF8700',
        'Aston Martin': '#229971',
        'Alpine': '#2293D1',
        'Williams': '#64C4FF',
        'Alfa Romeo': '#900000',
        'Haas': '#FFFFFF',
        'AlphaTauri': '#76F4F4',
        'Kick Sauber': '#900000',
        'RB': '#76F4F4',
        'Racing Bulls': '#76F4F4',
        'Sauber': '#900000',
        'Unknown': '#787878'
    }

    if team in TEAM_COLORS:
        return TEAM_COLORS[team]

    # fallback
    return "#00FF44"

def get_data(session, driver, lap_number=7):
    """
    gets telemetry data for a given driver in a given session in a given lap (default lap 7)
    """
    laps = session.laps
    try:
        driver_laps = laps.pick_drivers(driver).pick_fastest()
        if lap_number and len(laps.pick_drivers(driver)) >= lap_number:
             lap = laps.pick_drivers(driver).iloc[lap_number-1]
        else:
             lap = driver_laps

        if lap is None or len(lap) == 0:
            raise ValueError(f"No laps found for driver {driver}")

    except Exception as e:
        raise ValueError(f"Could not find lap data for driver {driver}: {e}")

    tele = lap.get_telemetry()
    tele['TimeSeconds'] = tele['Time'].dt.total_seconds()
    start_time = tele['TimeSeconds'].min()
    tele['TimeSeconds'] -= start_time

    duration = tele['TimeSeconds'].max()
    target_time = np.linspace(0, duration, int(duration * 30)) 

    interp_data = {
        'time': target_time,
        'x': np.interp(target_time, tele['TimeSeconds'], tele['X']),
        'y': np.interp(target_time, tele['TimeSeconds'], tele['Y']),
        'speed': np.interp(target_time, tele['TimeSeconds'], tele['Speed']),
        'gear': np.interp(target_time, tele['TimeSeconds'], tele['nGear']).astype(int),
        'rpm': np.interp(target_time, tele['TimeSeconds'], tele['RPM']),
        'throttle': np.interp(target_time, tele['TimeSeconds'], tele['Throttle']),
        'brake': np.interp(target_time, tele['TimeSeconds'], tele['Brake']),
    }

    try:
        track_pos = session.get_track_position()
        track_x = track_pos['X']
        track_y = track_pos['Y']
    except:
        track_x = tele['X']
        track_y = tele['Y']

    interp_data['track_x'] = track_x
    interp_data['track_y'] = track_y
    interp_data['driver'] = driver
    interp_data['colour'] = get_colour(lap['Team'])
    interp_data['track'] = session.event['EventName']
    interp_data['team'] = lap['Team']

    return interp_data

def get_multiple_drivers_data(session, drivers, lap_number=7):
    """
    gets telemetry data for multiple drivers
    Returns a dictionary with driver codes as keys
    """
    all_data = {}
    for driver in drivers:
        try:
            all_data[driver] = get_data(session, driver, lap_number)
            print(f"Loaded data for {driver}")
        except Exception as e:
            print(f"Warning: Could not load data for {driver}: {e}")

    if len(all_data) == 0:
        raise ValueError("No driver data could be loaded")

    return all_data

def setup_plot(data, lap_number, year, show_data=True):
    """
    initializes the matplotlib figure for single driver
    """
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(data['track_x'], data['track_y'], color="#FFFFFF", linewidth=10)
    ax.plot(data['x'], data['y'], color=data['colour'], linewidth=1, alpha=0.8)

    car_marker = ax.scatter([], [], s=100, c=data['colour'], edgecolors='black', zorder=5)
    trail_line, = ax.plot([], [], color='red', linewidth=2)

    info_text = ax.text(0.5, 0.3, '', transform=ax.transAxes, fontsize=10, 
                        verticalalignment='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.5))
    info_text.set_visible(show_data)

    ax.set_aspect('equal')
    ax.axis('off')

    lap_time = data['time'][-1]
    seconds = lap_time - 60

    ax.set_title(f"{data['track']} {year} | {data['driver']} | Lap {lap_number} (1:{seconds:.3f})")

    return fig, ax, car_marker, trail_line, info_text

def setup_multiple_plot(all_data, lap_number, show_data=True):
    """
    initializes the matplotlib figure for multiple drivers
    """
    fig, ax = plt.subplots(figsize=(12, 8))
    
    # get track data from first driver (all drivers are the same)
    first_driver = list(all_data.keys())[0]
    track_x = all_data[first_driver]['track_x']
    track_y = all_data[first_driver]['track_y']
    track_name = all_data[first_driver]['track']
    
    ax.plot(track_x, track_y, color="#444444", linewidth=8)

    car_markers = {}
    trail_lines = {}
    info_texts = {}
    
    text_y_positions = np.linspace(0.95, 0.5, len(all_data))
    
    for idx, (driver, data) in enumerate(all_data.items()):

        ax.plot(data['x'], data['y'], color=data['colour'], linewidth=1, alpha=0.3)
        car_markers[driver] = ax.scatter([], [], s=150, c=data['colour'], 
                                          edgecolors='white', zorder=5, label=driver)
        trail_lines[driver], = ax.plot([], [], color=data['colour'], linewidth=2)
        info_texts[driver] = ax.text(0.02, text_y_positions[idx], '', 
                                      transform=ax.transAxes, fontsize=9,
                                      verticalalignment='top',
                                      bbox=dict(boxstyle='round', facecolor='#222222', 
                                               alpha=0.8, edgecolor=data['colour']))
        info_texts[driver].set_visible(show_data)

    ax.legend(loc='upper right', fontsize=8, framealpha=0.5)

    drivers_str = ', '.join(all_data.keys())
    ax.set_title(f"{track_name} | Lap {lap_number} | Drivers: {drivers_str}", 
                 color='white', pad=20, fontsize=12)

    ax.set_aspect('equal')
    ax.axis('off')

    return fig, ax, car_markers, trail_lines, info_texts

def animate_single_driver_lap(session, driver, year, lap_number=1, show_data=True, save_path=None, interval=10):
    """
    generate animation for a single driver over a single lap
    """
    print(f"Getting data for {driver}...")
    data = get_data(session, driver, lap_number)

    print("Setting up plot...")
    fig, ax, car_marker, trail_line, info_text = setup_plot(data, lap_number, year, show_data=show_data)

    # nested functions to make FuncAnimation easier to use
    def init():
        car_marker.set_offsets(np.empty((0, 2)))
        trail_line.set_data([], [])
        info_text.set_text('')
        return car_marker, trail_line, info_text

    def update(frame):
        current_x = data['x'][frame]
        current_y = data['y'][frame]
        car_marker.set_offsets([[current_x, current_y]])

        trail_line.set_data(data['x'][:frame+1], data['y'][:frame+1])

        if show_data:
            speed = data['speed'][frame]
            gear = data['gear'][frame]
            rpm = data['rpm'][frame]
            throttle = data['throttle'][frame]
            brake = data['brake'][frame]

            pedal_status = "Coasting"
            if throttle > 10:
                pedal_status = "Throttle"
            if brake > 10:
                pedal_status = "Braking"

            # display data - add features here
            info_str = (f"{driver}\n"
                        f"Speed: {speed:.0f} km/h\n"
                        f"Gear: {gear}\n"
                        f"RPM: {rpm:.0f}\n"
                        f"Status: {pedal_status}")

            info_text.set_text(info_str)

        return car_marker, trail_line, info_text

    print("Generating animation...")

    anim = animation.FuncAnimation(fig, update, frames=len(data['time']), 
                                   init_func=init, blit=True, interval=interval, repeat=False)

    if save_path:
        print(f"Saving to {save_path}...")
        anim.save(save_path, writer='ffmpeg', fps=60)
    else:
        plt.show()

    return anim

def animate_multiple_drivers_lap(session, drivers, lap_number=1, show_data=True, save_path=None, interval=10):
    """
    generate animation for multiple drivers over a single lap simultaneously
    """
    print(f"Getting data for {len(drivers)} drivers: {drivers}...")
    all_data = get_multiple_drivers_data(session, drivers, lap_number)

    print("Setting up plot...")
    fig, ax, car_markers, trail_lines, info_texts = setup_multiple_plot(
        all_data, lap_number, show_data=show_data
    )

    def init():
        """Initialize all drivers"""
        result = []
        for driver in all_data.keys():
            car_markers[driver].set_offsets(np.empty((0, 2)))
            trail_lines[driver].set_data([], [])
            info_texts[driver].set_text('')
            result.extend([car_markers[driver], trail_lines[driver], info_texts[driver]])
        return result

    def update(frame):
        """Update all drivers"""
        result = []
        for driver, data in all_data.items():

            current_x = data['x'][frame]
            current_y = data['y'][frame]
            car_markers[driver].set_offsets([[current_x, current_y]])

            trail_lines[driver].set_data(data['x'][:frame+1], data['y'][:frame+1])

            if show_data:
                speed = data['speed'][frame]
                gear = data['gear'][frame]
                rpm = data['rpm'][frame]
                throttle = data['throttle'][frame]
                brake = data['brake'][frame]

                pedal_status = "Coasting"
                if throttle > 10:
                    pedal_status = "Throttle"
                if brake > 10:
                    pedal_status = "Braking"

                # add display features here
                info_str = (f"{driver}\n"
                            f"Speed: {speed:.0f} km/h\n"
                            f"Gear: {gear}\n"
                            f"RPM: {rpm:.0f}\n"
                            f"Status: {pedal_status}")

                info_texts[driver].set_text(info_str)

            result.extend([car_markers[driver], trail_lines[driver], info_texts[driver]])

        return result

    print("Generating animation...")

    anim = animation.FuncAnimation(fig, update, frames=len(list(all_data.values())[0]['time']), 
                                   init_func=init, blit=True, interval=interval)

    if save_path:
        print(f"Saving to {save_path}...")
        anim.save(save_path, writer='ffmpeg', fps=60)
    else:
        plt.show()

    return anim

if __name__ == "__main__":
    fastf1.Cache.enable_cache(cache_location) 

    year = 2024
    event = 21
    session_type = 'R' 

    try:
        session = fastf1.get_session(year, event, session_type)
        session.load()

        # single driver example
        animate_single_driver_lap(session, driver='VER', year=year, lap_number=48, show_data=True, save_path=None, interval=10)

        # multiple drivers example
        # animate_multiple_drivers_lap(session, drivers=['RUS', 'ALO', 'SAI'], lap_number=9, 
        #                             show_data=False, save_path=None, interval=10)

    except Exception as e:
        print(f"An error occurred: {e}")
