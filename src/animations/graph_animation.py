# animations
# animates data plotting. call animate_data_plot(dataset, "x", "y", "animation", interval=30, save_path=None)

import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

def animate_data_plot(data, x_label, y_label, title, interval=50, save_path=None):
    """
    animates a plot with expanding axes
    """

    # replace this with actual headers
    x_row = data[x_label]
    y_row = data[y_label]

    # ensure they are the same length
    if len(x_row) != len(y_row):
        raise ValueError("x and y rows must have the same length.")

    total_frames = len(x_row)

    fig, ax = plt.subplots()

    # replace these to match
    ax.set_title(title)
    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)
    ax.grid(True)

    # init an empty line object
    line, = ax.plot([], [], lw=2, color='blue', marker='o')

    # nested functions to make FuncAnimation easier to use
    def init():
        line.set_data([], [])

        ax.set_xlim(0, x_row[0] if len(x_row) > 0 else 1)
        ax.set_ylim(y_row[0] - 1, y_row[0] + 1) if len(y_row) > 0 else ax.set_ylim(-1, 1)
        return line,

    def update(frame):
        current_x = x_row[:frame + 1]
        current_y = y_row[:frame + 1]

        line.set_data(current_x, current_y)

        # expand x-axis
        x_min, x_max = min(current_x), max(current_x)
        x_pad = (x_max - x_min) * 0.1 if x_max != x_min else 1
        ax.set_xlim(x_min - x_pad, x_max + x_pad)
        
        # expand y-axis
        y_min, y_max = min(current_y), max(current_y)
        y_pad = (y_max - y_min) * 0.1 if y_max != y_min else 1
        ax.set_ylim(y_min - y_pad, y_max + y_pad)

        return line,

    anim = FuncAnimation(
        fig, 
        update, 
        frames=total_frames, 
        init_func=init, 
        blit=False,
        interval=interval,
        repeat=False
    )

    if save_path:
        print(f"Saving to {save_path}...")
        anim.save(save_path, writer='ffmpeg', fps=60)
    else:
        plt.show()

    return anim

# replace all of this with the data to plot
if __name__ == "__main__":
    import numpy as np

    x_values = np.linspace(0, 10, 100).tolist()
    y_values = np.cos(x_values).tolist()

    dataset = {
        'x': x_values,
        'y': y_values
    }

    anim = animate_data_plot(dataset, "x", "y", "animation", interval=30)
    plt.show()
