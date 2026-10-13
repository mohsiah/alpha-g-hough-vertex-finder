"""
Step 3: Combine multiple cosmic-ray tracks and multiple annihilation (vertex)
events into a single jumbled point cloud -- mimicking what a real detector
readout window actually looks like: an unlabeled pile of hits from several
overlapping events at once.

Ground-truth labels are kept alongside the hits (in a pandas DataFrame) purely
so *we* can check the Hough transform's answers later. The algorithm itself
will only ever be given the x,y,z columns -- never the label.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from track_generator import (RADIUS, HALF_HEIGHT, generate_cosmic_track,
                              generate_vertex_event)


def generate_mixed_event(n_cosmic=3, n_vertex=2):
    """Returns a pandas DataFrame with one row per detector hit:
    columns = x, y, z, truth_type ('cosmic' or 'vertex'), truth_event_id.

    truth_event_id distinguishes *different* cosmic tracks or different
    annihilation vertices from each other (e.g. cosmic track 0 vs cosmic
    track 1), separate from truth_type which just says which class it is.
    """
    rows = []
    event_id = 0

    for _ in range(n_cosmic):
        track = generate_cosmic_track()
        for hit in track:
            rows.append({
                'x': hit[0], 'y': hit[1], 'z': hit[2],
                'truth_type': 'cosmic',
                'truth_event_id': event_id
            })
        event_id += 1

    for _ in range(n_vertex):
        tracks, vertex = generate_vertex_event()
        for track in tracks:
            for hit in track:
                rows.append({
                    'x': hit[0], 'y': hit[1], 'z': hit[2],
                    'truth_type': 'vertex',
                    'truth_event_id': event_id
                })
        event_id += 1

    return pd.DataFrame(rows)


def draw_cylinder(ax):
    theta = np.linspace(0, 2 * np.pi, 40)
    z = np.linspace(-HALF_HEIGHT, HALF_HEIGHT, 2)
    theta_grid, z_grid = np.meshgrid(theta, z)
    x_grid = RADIUS * np.cos(theta_grid)
    y_grid = RADIUS * np.sin(theta_grid)
    ax.plot_wireframe(x_grid, y_grid, z_grid, color='lightgray', alpha=0.3, linewidth=0.5)


def plot_mixed_event(df, show_truth=False):
    """show_truth=False: plots every hit the same color, exactly what a real
    algorithm would have to work with -- no labels visible.
    show_truth=True: colors hits by their true class, for our own debugging
    only. Never feed this coloring into the actual classifier."""
    fig = plt.figure(figsize=(7, 7)) 
    ax = fig.add_subplot(111, projection='3d')
    draw_cylinder(ax)

    if show_truth:
        for etype, color in [('cosmic', 'crimson'), ('vertex', 'royalblue')]:
            subset = df[df['truth_type'] == etype]
            ax.scatter(subset['x'], subset['y'], subset['z'],
                       color=color, s=15, label=etype)
        ax.legend()
        ax.set_title("Mixed event -- TRUTH LABELS (for our debugging only)")
    else:
        ax.scatter(df['x'], df['y'], df['z'], color='dimgray', s=15)
        ax.set_title("Mixed event -- what the algorithm actually sees")

    ax.set_xlabel('x (cm)')
    ax.set_ylabel('y (cm)')
    ax.set_zlabel('z (cm)')
    return fig


if __name__ == "__main__":
    df = generate_mixed_event(n_cosmic=3, n_vertex=2)

    print(df.head())
    print(f"\nTotal hits: {len(df)}")
    print(df['truth_type'].value_counts())
    print(f"Distinct events: {df['truth_event_id'].nunique()}")

    plot_mixed_event(df, show_truth=False)
    plot_mixed_event(df, show_truth=True)
    plt.show()