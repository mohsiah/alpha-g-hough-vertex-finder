"""
ag_synth.py -- Step 5a: a MORE REALISTIC synthetic generator (replaces Step 1's toy one).

What changed vs Step 1, and why (all of it from the ALPHA-g papers):
  * UNITS ARE mm (the open Zenodo dataset is in mm; Step 1-4 used cm).
  * Annihilation vertices sit on the Penning-trap wall NEAR THE AXIS (r_vertex), inside the
    TPC's inner cathode. Pions fly OUTWARD through the gas. (Step 1 put the vertex on the
    outer wall pointing inward -- that is not the real geometry.)
  * Charged pions are HELICES in the 1 T axial field, not straight lines.
  * The TPC only records the annulus r_in < r < r_out; the hollow centre records nothing.
  * Random noise hits are added.
  * Cosmic muons come from above (zenith-angle pdf ~ cos^2), are straight, and are only
    "seen" where they cross the annulus.

!!! The numbers in Det are PLACEHOLDERS. Replace them with values measured from the real
    data (ag_data.estimate_geometry). The point of this file is to give a testbed whose
    *structure* matches reality, so the pipeline's logic can be debugged before real data.
"""
import numpy as np
from dataclasses import dataclass


@dataclass
class Det:
    r_in: float = 110.0        # mm  inner edge of recorded region (PLACEHOLDER)
    r_out: float = 190.0       # mm  outer edge (PLACEHOLDER)
    half_len: float = 1152.0   # mm  (2304 mm active length, from the paper)
    r_vertex: float = 25.0     # mm  trap-wall radius where annihilations happen (PLACEHOLDER)
    b_field: float = 1.0       # T   (from the paper)
    sigma_xy: float = 1.5      # mm  spacepoint smearing (PLACEHOLDER)
    sigma_z: float = 2.0       # mm  (PLACEHOLDER)
    step: float = 4.0          # mm  3D spacing between spacepoints along a track
    n_noise: float = 15.0      # mean number of random noise hits per event
    vz_range: float = 1000.0   # mm  vertices uniform in +-vz_range


def _smear(rng, det, pts):
    pts = pts.copy()
    pts[:, :2] += rng.normal(0, det.sigma_xy, size=(len(pts), 2))
    pts[:, 2] += rng.normal(0, det.sigma_z, size=len(pts))
    return pts


def _in_acceptance(det, pts):
    r = np.hypot(pts[:, 0], pts[:, 1])
    return (r >= det.r_in) & (r <= det.r_out) & (np.abs(pts[:, 2]) <= det.half_len)


def noise_hits(rng, det, mean=None):
    n = rng.poisson(det.n_noise if mean is None else mean)
    r = np.sqrt(rng.uniform(det.r_in**2, det.r_out**2, n))
    ph = rng.uniform(0, 2 * np.pi, n)
    z = rng.uniform(-det.half_len, det.half_len, n)
    return np.column_stack([r * np.cos(ph), r * np.sin(ph), z])


def pion_track(rng, det, vertex):
    """One charged pion from `vertex`: helix in the axial field, sampled only inside the annulus."""
    cos_t = rng.uniform(-1, 1)
    sin_t = np.sqrt(1 - cos_t**2)
    psi0 = rng.uniform(0, 2 * np.pi)
    p = rng.uniform(0.15, 0.60)                      # GeV/c, rough pion momentum range
    pt = max(p * sin_t, 1e-3)
    R = pt / (0.3 * det.b_field) * 1000.0            # mm   (R[m] = pt[GeV]/(0.3 B[T]))
    q = rng.choice([-1.0, 1.0])
    kappa = q / R
    dl = max(det.step * sin_t, 0.3)                  # transverse arc step -> ~step mm in 3D
    l = np.arange(0.0, 700.0, dl)
    psi = psi0 + kappa * l
    x = vertex[0] + (np.sin(psi) - np.sin(psi0)) / kappa
    y = vertex[1] - (np.cos(psi) - np.cos(psi0)) / kappa
    z = vertex[2] + l * cos_t / max(sin_t, 1e-6)
    pts = np.column_stack([x, y, z])
    return _smear(rng, det, pts[_in_acceptance(det, pts)])


def make_annihilation(rng, det=Det()):
    n_trk = rng.choice([2, 3, 4, 5], p=[0.30, 0.35, 0.25, 0.10])
    ph_v = rng.uniform(0, 2 * np.pi)
    vz = rng.uniform(-det.vz_range, det.vz_range)
    vertex = np.array([det.r_vertex * np.cos(ph_v), det.r_vertex * np.sin(ph_v), vz])
    tracks = [pion_track(rng, det, vertex) for _ in range(n_trk)]
    n_vis = sum(len(t) >= 6 for t in tracks)
    noise = noise_hits(rng, det)
    pts = np.vstack([t for t in tracks if len(t)] + [noise])
    labels = np.concatenate([np.full(len(t), i) for i, t in enumerate(tracks)] + [np.full(len(noise), -1)])
    return dict(pts=pts, vz_true=vz, vz_helix=np.nan, kind='annihilation',
                n_truth_tracks=int(n_vis), n_cosmics=0, labels=labels)


def make_cosmic_hits(rng, det=Det(), step=5.0):
    """Straight muon, zenith pdf ~ cos^2 (c = u^(1/3)), uniform azimuth, crossing z=0 inside
    the outer radius. Re-drawn until it leaves hits in the annulus (invisible ones don't exist
    in real data either)."""
    while True:
        c = rng.uniform(0, 1) ** (1 / 3)
        s = np.sqrt(1 - c * c)
        az = rng.uniform(0, 2 * np.pi)
        d = np.array([s * np.cos(az), s * np.sin(az), -c])
        rr = det.r_out * np.sqrt(rng.uniform())
        ph = rng.uniform(0, 2 * np.pi)
        p0 = np.array([rr * np.cos(ph), rr * np.sin(ph), 0.0])
        T = det.half_len / c
        t = np.arange(-T, T, step)
        pts = p0[None, :] + t[:, None] * d[None, :]
        pts = pts[_in_acceptance(det, pts)]
        if len(pts) >= 8:
            return _smear(rng, det, pts)


def make_cosmic_event(rng, det=Det(), n_cosmics=1):
    cos = [make_cosmic_hits(rng, det) for _ in range(n_cosmics)]
    noise = noise_hits(rng, det)
    labels = np.concatenate([np.full(len(c), 100 + i) for i, c in enumerate(cos)] + [np.full(len(noise), -1)])
    return dict(pts=np.vstack(cos + [noise]), vz_true=np.nan, vz_helix=np.nan, kind='cosmic',
                n_truth_tracks=0, n_cosmics=n_cosmics, labels=labels)



def overlay_cosmics(rng, event, det=Det(), n_cosmics=1):
    """Same idea as make_mixed_event but starting from ANY existing event (e.g. a real
    simulated annihilation loaded from the Zenodo file). This is the 'embedding' trick."""
    ev = dict(event)
    ev['pts'] = np.vstack([event['pts']] + [make_cosmic_hits(rng, det) for _ in range(n_cosmics)])
    ev['kind'] = 'mixed'
    ev['n_cosmics'] = n_cosmics
    return ev