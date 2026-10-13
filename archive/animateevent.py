"""
Step 2: Animate a simulated annihilation event so the particle tracks visibly
grow outward from the wall over time, with a "flash" marking the annihilation
moment -- instead of one static all-at-once scatter plot.
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from track_generator import (RADIUS, HALF_HEIGHT, generate_cosmic_track,
                              generate_vertex_event)


def draw_cylinder(ax):
    theta = np.linspace(0, 2 * np.pi, 40)
    z = np.linspace(-HALF_HEIGHT, HALF_HEIGHT, 2)
    theta_grid, z_grid = np.meshgrid(theta, z)
    x_grid = RADIUS * np.cos(theta_grid)
    y_grid = RADIUS * np.sin(theta_grid)
    ax.plot_wireframe(x_grid, y_grid, z_grid, color='lightgray', alpha=0.3, linewidth=0.5)


def animate_annihilation(vertex_tracks, vertex, n_frames=30):
    fig = plt.figure(figsize=(7, 7))
    ax = fig.add_subplot(111, projection='3d')

    n_hits = vertex_tracks[0].shape[0]  # hits per track (all tracks share this)
    # frame budget: first few frames = just the flash, remaining = tracks growing
    flash_frames = 5
    grow_frames = n_frames - flash_frames

    def update(frame):
        ax.clear()
        draw_cylinder(ax)
        ax.set_xlim(-RADIUS * 1.1, RADIUS * 1.1)
        ax.set_ylim(-RADIUS * 1.1, RADIUS * 1.1)
        ax.set_zlim(-HALF_HEIGHT, HALF_HEIGHT)
        ax.set_xlabel('x (cm)')
        ax.set_ylabel('y (cm)')
        ax.set_zlabel('z (cm)')

        if frame < flash_frames:
            # pulsing flash at the annihilation point on the wall
            size = 200 + 300 * np.sin(frame / flash_frames * np.pi)
            ax.scatter(*vertex, color='gold', marker='*', s=size,
                       edgecolor='red', linewidth=1.5, label='annihilation (t=0)')
            ax.set_title("Antihydrogen annihilates on the wall...")
        else:
            g = frame - flash_frames
            frac = (g + 1) / grow_frames
            n_visible = max(1, int(np.ceil(frac * n_hits)))

            ax.scatter(*vertex, color='gold', marker='*', s=150,
                       edgecolor='red', linewidth=1, label='vertex')
            for i, track in enumerate(vertex_tracks):
                visible = track[:n_visible]
                ax.scatter(*visible.T, color='royalblue', s=20,
                           label='pion track' if i == 0 else None)
            ax.set_title("Charged pions propagate outward through the gas")

        ax.legend(loc='upper right')
        return ax,

    ani = animation.FuncAnimation(fig, update, frames=n_frames, interval=120,
                                   repeat=True)
    plt.show()  # opens a live window and plays the animation on loop
    return ani  # keep a reference so it isn't garbage-collected mid-animation


if __name__ == "__main__":
    vertex_tracks, vertex = generate_vertex_event()
    ani = animate_annihilation(vertex_tracks, vertex)