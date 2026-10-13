"""
Step 4: Modified Hough transform for separating cosmic-ray background from
antihydrogen annihilation ("vertex") events in the synthetic cylindrical TPC
data from Steps 1-3.

Why this parameterization:
Cosmic tracks are near-vertical (see track_generator.generate_cosmic_track),
so they collapse to almost a single point if you project onto the x-y plane
-- the classic top-down Hough view used for many TPCs doesn't help here.
Instead we use the two *side* projections, x-z and y-z, where a genuinely
straight 3D line always projects to a straight 2D line in BOTH views. We run
the standard image-processing Hough transform (rho = u*cos(theta)+v*sin(theta))
on each side view separately, then only accept a 3D track candidate where a
peak in the x-z accumulator and a peak in the y-z accumulator agree on
(roughly) the same set of hits. This "vote in two views, then intersect"
trick is exactly the kind of projection-based approach real cylindrical TPCs
use, and it keeps every step debuggable as a 2D picture.

Pipeline:
  1. hough_transform_2d / find_peaks_2d   -> classic 2D Hough, pure numpy
  2. find_lines_2d                        -> peaks -> inlier hits -> refit
  3. find_tracks_3d                       -> agree x-z view with y-z view
  4. classify_track                       -> cosmic (long+vertical) vs vertex leg (short)
  5. cluster_vertices                     -> group vertex-leg tracks that share a wall point
  6. score_tracks                         -> compare against Step 3's ground truth

Only numpy / matplotlib / pandas are used -- no scikit-learn, no scipy.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.animation as animation

from track_generator import RADIUS, HALF_HEIGHT
from python_mixedevent import generate_mixed_event


# ---------------------------------------------------------------------------
# 1. Classic 2D Hough transform (fully vectorized -- no per-point python loop)
# ---------------------------------------------------------------------------

def hough_transform_2d(u, v, n_theta=180, n_rho=200, rho_range=None):
    """Standard Hough accumulator for lines in a 2D (u, v) point set, using
    the normal form rho = u*cos(theta) + v*sin(theta). theta in [0, pi)."""
    thetas = np.linspace(0, np.pi, n_theta, endpoint=False)
    cos_t, sin_t = np.cos(thetas), np.sin(thetas)

    # rho_matrix[i, j] = rho of point i at angle theta_j
    rho_matrix = np.outer(u, cos_t) + np.outer(v, sin_t)

    if rho_range is None:
        max_r = np.sqrt(np.max(u**2 + v**2)) * 1.05 + 1e-6
        rho_range = (-max_r, max_r)
    rho_vals = np.linspace(rho_range[0], rho_range[1], n_rho)
    bin_width = (rho_range[1] - rho_range[0]) / n_rho

    rho_idx = np.clip(((rho_matrix - rho_range[0]) / bin_width).astype(int), 0, n_rho - 1)
    theta_idx = np.broadcast_to(np.arange(n_theta), rho_idx.shape)

    accumulator = np.zeros((n_rho, n_theta), dtype=int)
    np.add.at(accumulator, (rho_idx.ravel(), theta_idx.ravel()), 1)
    return accumulator, thetas, rho_vals


def find_peaks_2d(accumulator, thetas, rho_vals, n_peaks=12, min_votes=4,
                   suppress_bins=(6, 6)):
    """Greedy peak-picking with non-max suppression: take the current max,
    record it, zero out a window around it so nearby bins from the same
    physical line don't get picked again, repeat."""
    work = accumulator.copy()
    peaks = []
    for _ in range(n_peaks):
        r_i, t_i = np.unravel_index(np.argmax(work), work.shape)
        votes = work[r_i, t_i]
        if votes < min_votes:
            break
        peaks.append({'theta': thetas[t_i], 'rho': rho_vals[r_i], 'votes': int(votes)})
        r_lo, r_hi = max(0, r_i - suppress_bins[0]), min(work.shape[0], r_i + suppress_bins[0] + 1)
        t_lo, t_hi = max(0, t_i - suppress_bins[1]), min(work.shape[1], t_i + suppress_bins[1] + 1)
        work[r_lo:r_hi, t_lo:t_hi] = 0
    return peaks


def line_inlier_mask(u, v, theta, rho, tol):
    """Perpendicular distance from every (u, v) point to the line, thresholded."""
    dist = np.abs(u * np.cos(theta) + v * np.sin(theta) - rho)
    return dist < tol


def refit_line_tls(u, v):
    """Total-least-squares refit of a line through points already assigned to
    a Hough peak, via eigendecomposition of the 2x2 scatter matrix. Cleans up
    the coarse Hough-bin estimate into a precise (theta, rho)."""
    pts = np.stack([u, v], axis=1)
    centroid = pts.mean(axis=0)
    centered = pts - centroid
    cov = centered.T @ centered
    eigvals, eigvecs = np.linalg.eigh(cov)   # ascending eigenvalues
    normal = eigvecs[:, 0]                   # smallest-variance direction = normal to line
    theta = np.arctan2(normal[1], normal[0]) % np.pi
    n = np.array([np.cos(theta), np.sin(theta)])
    rho = centroid @ n
    return theta, rho


def find_lines_2d(u, v, tol=0.6, n_theta=180, n_rho=200, n_peaks=12, min_votes=4,
                   suppress_bins=(6, 6)):
    """Full 2D pipeline: vote -> find peaks -> collect inliers -> refit -> re-collect inliers."""
    accumulator, thetas, rho_vals = hough_transform_2d(u, v, n_theta, n_rho)
    peaks = find_peaks_2d(accumulator, thetas, rho_vals, n_peaks=n_peaks, min_votes=min_votes,
                           suppress_bins=suppress_bins)

    lines = []
    for peak in peaks:
        mask = line_inlier_mask(u, v, peak['theta'], peak['rho'], tol)
        if mask.sum() < min_votes:
            continue
        theta_r, rho_r = refit_line_tls(u[mask], v[mask])
        mask_refined = line_inlier_mask(u, v, theta_r, rho_r, tol)
        lines.append({'theta': theta_r, 'rho': rho_r, 'mask': mask_refined,
                       'votes': int(mask_refined.sum())})
    return lines, accumulator, thetas, rho_vals


# ---------------------------------------------------------------------------
# 2. Combine the x-z and y-z views into real 3D track candidates
# ---------------------------------------------------------------------------

def find_tracks_3d(df, tol=0.6, min_hits=4, overlap_thresh=0.55,
                    n_theta=360, n_rho=300, suppress_bins=(4, 4), n_peaks=15,
                    **hough_kwargs):
    """A hit set is accepted as a genuine 3D straight track only if it forms
    a line in BOTH the x-z and y-z projections. This cross-view agreement is
    the actual "modified" part of the transform: it rejects the coincidental
    2D alignments that a single projection alone would be fooled by."""
    x, y, z = df['x'].to_numpy(), df['y'].to_numpy(), df['z'].to_numpy()

    lines_xz, acc_xz, th_xz, rho_xz = find_lines_2d(x, z, tol=tol, n_theta=n_theta, n_rho=n_rho,
                                                     suppress_bins=suppress_bins, n_peaks=n_peaks,
                                                     **hough_kwargs)
    lines_yz, acc_yz, th_yz, rho_yz = find_lines_2d(y, z, tol=tol, n_theta=n_theta, n_rho=n_rho,
                                                     suppress_bins=suppress_bins, n_peaks=n_peaks,
                                                     **hough_kwargs)

    candidates = []
    for lxz in lines_xz:
        for lyz in lines_yz:
            overlap = lxz['mask'] & lyz['mask']
            n_overlap = int(overlap.sum())
            if n_overlap < min_hits:
                continue
            union = int((lxz['mask'] | lyz['mask']).sum())
            jaccard = n_overlap / max(1, union)
            if jaccard < overlap_thresh:
                continue
            candidates.append({'hit_mask': overlap, 'n_hits': n_overlap})

    # Greedy claim hits from strongest candidate down, so one hit can't be
    # double-counted into two different reconstructed tracks.
    candidates.sort(key=lambda c: c['n_hits'], reverse=True)
    used = np.zeros(len(df), dtype=bool)
    tracks = []
    for c in candidates:
        available = c['hit_mask'] & ~used
        if available.sum() < min_hits:
            continue
        used |= available
        tracks.append(np.where(available)[0])

    views = {'xz': (acc_xz, th_xz, rho_xz, lines_xz),
             'yz': (acc_yz, th_yz, rho_yz, lines_yz)}
    return tracks, views


# ---------------------------------------------------------------------------
# 3. Classification: cosmic (long + vertical) vs vertex leg (short + steep)
# ---------------------------------------------------------------------------

def classify_track(points_xyz, cosmic_angle_max=0.35, cosmic_length_min=20.0):
    """Mirrors the physical distinction baked into Step 1's generator: cosmic
    muons are long and nearly parallel to z; annihilation pions are short
    (3-12 cm, capped by generate_vertex_event) and fly off at much steeper
    angles. We deliberately use an ABSOLUTE length cutoff rather than "spans
    most of the detector height": a cosmic ray that happens to enter off-
    center can exit through the curved side wall partway down and still be
    a long, clearly-vertical track without crossing the full height."""
    order = np.argsort(points_xyz[:, 2])
    p_lo, p_hi = points_xyz[order[0]], points_xyz[order[-1]]
    direction = p_hi - p_lo
    length = np.linalg.norm(direction)
    if length < 1e-6:
        return 'vertex', 0.0, 0.0

    angle_from_z = np.arccos(np.clip(abs(direction[2]) / length, -1.0, 1.0))
    is_cosmic = (angle_from_z < cosmic_angle_max) and (length > cosmic_length_min)
    return ('cosmic' if is_cosmic else 'vertex'), angle_from_z, length


# ---------------------------------------------------------------------------
# 4. Vertex reconstruction: cluster where several "vertex-leg" tracks meet the wall
# ---------------------------------------------------------------------------

def wall_anchor_point(points_xyz):
    """Per Step 1, annihilation vertices sit ON the cylinder wall, and pion
    tracks are generated starting there. So the hit with the largest radial
    distance from a short track is our best estimate of where it met the wall."""
    r = np.hypot(points_xyz[:, 0], points_xyz[:, 1])
    return points_xyz[np.argmax(r)]


def cluster_vertices(anchor_points, dist_thresh=4.0):
    """Simple greedy single-linkage clustering: walk the anchor points, attach
    each to the nearest existing cluster centroid within dist_thresh, else
    start a new cluster. A vertex is confirmed when >=2 tracks land in the
    same cluster -- one leg alone can't reconstruct a decay vertex."""
    clusters, centroids = [], []
    labels = np.full(len(anchor_points), -1, dtype=int)
    for i, p in enumerate(anchor_points):
        match = next((ci for ci, cen in enumerate(centroids)
                      if np.linalg.norm(p - cen) < dist_thresh), -1)
        if match == -1:
            clusters.append([i])
            centroids.append(p)
            labels[i] = len(clusters) - 1
        else:
            clusters[match].append(i)
            centroids[match] = anchor_points[clusters[match]].mean(axis=0)
            labels[i] = match
    return labels, np.array(centroids)


# ---------------------------------------------------------------------------
# 5. Scoring against Step 3's ground truth (never used by the algorithm above)
# ---------------------------------------------------------------------------

def score_tracks(df, tracks, predicted_labels):
    rows = []
    for hit_idx, pred in zip(tracks, predicted_labels):
        truth_slice = df.iloc[hit_idx]
        truth_type = truth_slice['truth_type'].mode().iloc[0]
        truth_event = truth_slice['truth_event_id'].mode().iloc[0]
        rows.append({'n_hits': len(hit_idx), 'predicted': pred, 'truth': truth_type,
                      'correct': pred == truth_type, 'truth_event_id': truth_event})
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# 6. Visualization
# ---------------------------------------------------------------------------

def draw_cylinder(ax):
    theta = np.linspace(0, 2 * np.pi, 40)
    z = np.linspace(-HALF_HEIGHT, HALF_HEIGHT, 2)
    theta_grid, z_grid = np.meshgrid(theta, z)
    ax.plot_wireframe(RADIUS * np.cos(theta_grid), RADIUS * np.sin(theta_grid), z_grid,
                       color='lightgray', alpha=0.3, linewidth=0.5)


def plot_accumulator(accumulator, thetas, rho_vals, lines, title):
    fig, ax = plt.subplots(figsize=(6, 5))
    extent = [thetas[0], thetas[-1], rho_vals[0], rho_vals[-1]]
    ax.imshow(accumulator, extent=extent, aspect='auto', origin='lower', cmap='viridis')
    for line in lines:
        ax.plot(line['theta'], line['rho'], marker='+', color='red',
                 markersize=14, markeredgewidth=2)
    ax.set_xlabel('theta (rad)')
    ax.set_ylabel('rho (cm)')
    ax.set_title(title)
    fig.colorbar(ax.images[0], ax=ax, label='votes')
    return fig


def plot_before_after(df, tracks, predicted_labels):
    fig = plt.figure(figsize=(13, 6))
    color_map = {'cosmic': 'crimson', 'vertex': 'royalblue'}

    ax1 = fig.add_subplot(121, projection='3d')
    draw_cylinder(ax1)
    ax1.scatter(df['x'], df['y'], df['z'], color='dimgray', s=12)
    ax1.set_title("Before: raw unlabeled hits")

    ax2 = fig.add_subplot(122, projection='3d')
    draw_cylinder(ax2)
    hit_color = np.full(len(df), 'lightgray', dtype=object)
    for hit_idx, pred in zip(tracks, predicted_labels):
        hit_color[hit_idx] = color_map[pred]
    ax2.scatter(df['x'], df['y'], df['z'], color=list(hit_color), s=12)
    ax2.set_title("After: Hough-classified hits")

    for ax in (ax1, ax2):
        ax.set_xlabel('x (cm)'); ax.set_ylabel('y (cm)'); ax.set_zlabel('z (cm)')
    plt.tight_layout()
    return fig


def animate_discovery(df, tracks, predicted_labels, vertex_centroids=None, interval=250):
    """Reveals reconstructed tracks one at a time, colored by the algorithm's
    own classification, then drops in the reconstructed vertex markers --
    the same "grow over time" visual language as Step 2's animation."""
    fig = plt.figure(figsize=(7, 7))
    ax = fig.add_subplot(111, projection='3d')
    color_map = {'cosmic': 'crimson', 'vertex': 'royalblue'}

    n_tracks = len(tracks)
    n_frames = n_tracks + (5 if vertex_centroids is not None and len(vertex_centroids) else 0)

    def update(frame):
        ax.clear()
        draw_cylinder(ax)
        ax.set_xlim(-RADIUS * 1.1, RADIUS * 1.1)
        ax.set_ylim(-RADIUS * 1.1, RADIUS * 1.1)
        ax.set_zlim(-HALF_HEIGHT, HALF_HEIGHT)
        ax.set_xlabel('x (cm)'); ax.set_ylabel('y (cm)'); ax.set_zlabel('z (cm)')

        shown = min(frame + 1, n_tracks)
        for i in range(shown):
            pts = df.iloc[tracks[i]][['x', 'y', 'z']].to_numpy()
            ax.scatter(*pts.T, color=color_map[predicted_labels[i]], s=18,
                       label=predicted_labels[i] if i == 0 else None)

        if frame >= n_tracks and vertex_centroids is not None and len(vertex_centroids):
            ax.scatter(*vertex_centroids.T, color='gold', marker='*', s=250,
                       edgecolor='black', linewidth=1, label='reconstructed vertex')

        ax.set_title(f"Hough reconstruction: track {shown}/{n_tracks}"
                     if frame < n_tracks else "Reconstructed annihilation vertices")
        ax.legend(loc='upper right')
        return ax,

    ani = animation.FuncAnimation(fig, update, frames=n_frames, interval=interval, repeat=True)
    return fig, ani


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    df = generate_mixed_event(n_cosmic=3, n_vertex=2)
    print(f"Mixed event: {len(df)} hits, {df['truth_event_id'].nunique()} truth events")

    tracks, views = find_tracks_3d(df, tol=0.6, min_hits=4)
    print(f"Reconstructed {len(tracks)} candidate 3D tracks")

    predicted_labels = []
    vertex_leg_anchors = []
    vertex_leg_track_idx = []
    for i, hit_idx in enumerate(tracks):
        pts = df.iloc[hit_idx][['x', 'y', 'z']].to_numpy()
        label, angle, length = classify_track(pts)
        predicted_labels.append(label)
        if label == 'vertex':
            vertex_leg_anchors.append(wall_anchor_point(pts))
            vertex_leg_track_idx.append(i)

    vertex_centroids = np.empty((0, 3))
    if vertex_leg_anchors:
        anchor_arr = np.array(vertex_leg_anchors)
        cluster_labels, centroids = cluster_vertices(anchor_arr, dist_thresh=4.0)
        counts = pd.Series(cluster_labels).value_counts()
        confirmed = counts[counts >= 2].index
        vertex_centroids = centroids[confirmed] if len(confirmed) else np.empty((0, 3))
        print(f"Confirmed {len(vertex_centroids)} reconstructed vertices "
              f"from {len(vertex_leg_anchors)} vertex-leg tracks")

    results = score_tracks(df, tracks, predicted_labels)
    print("\nPer-track results:")
    print(results)
    print(f"\nTrack-level accuracy: {results['correct'].mean():.2%}")
    print("\nConfusion matrix (rows=truth, cols=predicted):")
    print(pd.crosstab(results['truth'], results['predicted']))

    plot_accumulator(*views['xz'][:3], views['xz'][3], "Hough accumulator: x-z view")
    plot_accumulator(*views['yz'][:3], views['yz'][3], "Hough accumulator: y-z view")
    plot_before_after(df, tracks, predicted_labels)
    fig, ani = animate_discovery(df, tracks, predicted_labels, vertex_centroids)

    plt.show()