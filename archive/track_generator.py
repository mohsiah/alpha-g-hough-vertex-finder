"""
Step 1: Synthetic track generator for cylindrical TPC background-rejection demo.

Generates two kinds of events inside a cylinder (standing in for the ALPHA-g rTPC):
  1. Cosmic-ray tracks: long, mostly-straight lines that cross the whole cylinder.
  2. Annihilation ("vertex") events: a few short tracks radiating from one random point.

Everything here uses only numpy + matplotlib.
"""

import numpy as np
import matplotlib.pyplot as plt

# ---- Detector geometry (rough stand-in for the rTPC dimensions in the paper) ----
RADIUS = 19.0      # cm, outer cathode wall radius
HALF_HEIGHT = 115.0  # cm, roughly half of the 230 cm active length

rng = np.random.default_rng(seed=42)  # fixed seed = reproducible for now


def sample_points_on_segment(p_start, p_end, n_points, noise_std=0.1):
    """Given two 3D endpoints, sample n_points along the segment with added noise.
    This stands in for discrete detector 'hits' along a real particle track."""
    t = np.linspace(0, 1, n_points)
    line = p_start[None, :] + t[:, None] * (p_end - p_start)[None, :]
    noise = rng.normal(0, noise_std, size=line.shape)
    return line + noise


def line_cylinder_intersection(point, direction):
    """Given a point and direction defining an infinite line, find the t-interval
    where it lies inside the finite cylinder (radius RADIUS, +-HALF_HEIGHT in z).
    We compute the valid t-range from the radial wall AND from the two z-caps
    separately, then intersect those ranges. Returns None if they don't overlap."""
    px, py, pz = point
    dx, dy, dz = direction
    t_min, t_max = -np.inf, np.inf

    # --- radial wall constraint: |xy(t)| <= RADIUS ---
    a = dx**2 + dy**2
    b = 2 * (px * dx + py * dy)
    c = px**2 + py**2 - RADIUS**2
    if a > 1e-12:
        disc = b**2 - 4 * a * c
        if disc < 0:
            return None  # line's xy-projection never reaches the cylinder radius
        sqrt_disc = np.sqrt(disc)
        r_lo, r_hi = sorted([(-b - sqrt_disc) / (2 * a), (-b + sqrt_disc) / (2 * a)])
        t_min, t_max = max(t_min, r_lo), min(t_max, r_hi)
    else:
        if c > 0:
            return None  # constant radius already outside cylinder

    # --- z-cap constraint: -HALF_HEIGHT <= z(t) <= HALF_HEIGHT ---
    if abs(dz) > 1e-12:
        z_lo, z_hi = sorted([(-HALF_HEIGHT - pz) / dz, (HALF_HEIGHT - pz) / dz])
        t_min, t_max = max(t_min, z_lo), min(t_max, z_hi)
    else:
        if abs(pz) > HALF_HEIGHT:
            return None  # horizontal line outside the z-caps entirely

    if t_min >= t_max:
        return None  # empty interval -> line misses the finite cylinder

    p1 = point + t_min * direction
    p2 = point + t_max * direction
    return p1, p2


def generate_cosmic_track(n_hits=15):
    """Cosmic muons arrive mostly vertically (small angle off straight-down),
    unlike annihilation products which fly off in random directions."""
    # mostly-vertical direction with a small random tilt.
    # Real cosmic muons follow roughly a cos^2(theta) zenith-angle distribution,
    # i.e. strongly peaked near vertical -- 0.05 rad std keeps tracks crossing
    # the full detector height rather than exiting through the side wall.
    tilt = rng.normal(0, 0.05, size=2)
    direction = np.array([tilt[0], tilt[1], -1.0])
    direction /= np.linalg.norm(direction)

    # random entry point roughly centered, offset within the cylinder's radius
    offset_r = rng.uniform(0, RADIUS * 0.8)
    offset_angle = rng.uniform(0, 2 * np.pi)
    start_point = np.array([offset_r * np.cos(offset_angle),
                             offset_r * np.sin(offset_angle),
                             HALF_HEIGHT])

    intersection = line_cylinder_intersection(start_point, direction)
    if intersection is None:
        return generate_cosmic_track(n_hits)  # retry if it missed geometry

    p1, p2 = intersection
    return sample_points_on_segment(p1, p2, n_hits)


def generate_vertex_event(n_tracks=None, n_hits_per_track=6):
    """Antihydrogen annihilates ON THE WALL of the apparatus (per the paper's own
    description), not floating inside the gas. So the vertex sits on the cylinder's
    curved wall (r = RADIUS), and the resulting charged-pion tracks fly inward into
    the gas volume where the rTPC can actually detect their ionization trail."""
    if n_tracks is None:
        n_tracks = rng.integers(2, 4)  # 2 or 3 tracks, like real pion multiplicities

    # vertex point ON the wall, not inside the volume
    vtheta = rng.uniform(0, 2 * np.pi)
    vz = rng.uniform(-HALF_HEIGHT * 0.8, HALF_HEIGHT * 0.8)
    vertex = np.array([RADIUS * np.cos(vtheta), RADIUS * np.sin(vtheta), vz])
    outward_normal = np.array([np.cos(vtheta), np.sin(vtheta), 0.0])

    tracks = []
    for _ in range(n_tracks):
        # random isotropic direction, then force it inward (into the gas) --
        # a pion can't usefully travel straight through the solid wall
        direction = rng.normal(size=3)
        direction /= np.linalg.norm(direction)
        if np.dot(direction, outward_normal) > 0:
            direction = -direction

        length = rng.uniform(3, 12)
        end_point = vertex + direction * length

        tracks.append(sample_points_on_segment(vertex, end_point, n_hits_per_track))
    return tracks, vertex


def plot_event(cosmic_hits=None, vertex_tracks=None, vertex=None, title=""):
    fig = plt.figure(figsize=(7, 7))
    ax = fig.add_subplot(111, projection='3d')

    # draw cylinder wireframe for reference
    theta = np.linspace(0, 2 * np.pi, 40)
    z = np.linspace(-HALF_HEIGHT, HALF_HEIGHT, 2)
    theta_grid, z_grid = np.meshgrid(theta, z)
    x_grid = RADIUS * np.cos(theta_grid)
    y_grid = RADIUS * np.sin(theta_grid)
    ax.plot_wireframe(x_grid, y_grid, z_grid, color='lightgray', alpha=0.3, linewidth=0.5)

    if cosmic_hits is not None:
        ax.scatter(*cosmic_hits.T, color='crimson', label='cosmic track', s=15)

    if vertex_tracks is not None:
        for i, track in enumerate(vertex_tracks):
            ax.scatter(*track.T, color='royalblue', s=15,
                       label='annihilation track' if i == 0 else None)
        if vertex is not None:
            ax.scatter(*vertex, color='black', marker='*', s=100, label='vertex')

    ax.set_xlabel('x (cm)')
    ax.set_ylabel('y (cm)')
    ax.set_zlabel('z (cm)')
    ax.set_title(title)
    ax.legend()
    plt.tight_layout()
    return fig


if __name__ == "__main__":
    cosmic = generate_cosmic_track()
    vertex_tracks, vertex = generate_vertex_event()

    plot_event(cosmic_hits=cosmic, title="Simulated cosmic-ray track")
    plot_event(vertex_tracks=vertex_tracks, vertex=vertex,
               title="Simulated annihilation (vertex) event")

    print(f"Cosmic track: {cosmic.shape[0]} hits")
    print(f"Vertex event: {len(vertex_tracks)} tracks, "
          f"{sum(t.shape[0] for t in vertex_tracks)} total hits")

    plt.show()  # opens both figure windows; script pauses here until you close them