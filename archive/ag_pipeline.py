"""
ag_pipeline.py -- Step 5b: the FINAL algorithm.

Input : one event = (n, 3) array of spacepoints (x, y, z) in mm.  NO labels, ever.
Output: Result(tracks, vertex, label, reason)

Pipeline (each stage maps onto something you already built):

  1. FIND TRACKS      Your Step-4 2D Hough (find_lines_2d) is reused UNCHANGED, but applied
                      in the detector's natural views:  hits are first grouped by azimuth
                      phi (a 1-D vote), then a Hough line is searched in the (r, z) plane of
                      each phi-sector.  A candidate must ALSO be a straight line in (r, r*dphi)
                      -> the same "two views must agree" idea as your (x-z, y-z) agreement,
                      but with views that are never degenerate for tracks leaving the axis.
  2. FIT TRACKS       Total-least-squares line in (r,z) [reuses refit_line_tls] and in (x,y).
                      From these: z-extrapolation to the trap wall, and d0 = distance of the
                      xy-line from the z axis ("does this track point back at the trap?").
  3. FIND VERTEX      Tracks that point at the axis (d0 small) and agree on the same z at the
                      trap radius form a vertex. >= 2 tracks are required.
  4. CLASSIFY         vertex found -> 'annihilation';  otherwise -> 'background'.
                      (Your Step-4 classify_track idea survives as a per-track TAG.)

Why a *vertex* is the discriminator and not "is the track long and vertical":
  a near-vertical pion that crosses the whole TPC looks like a cosmic by shape alone, while a
  cosmic that happens to be tilted radially looks like a pion by shape alone. What separates
  them physically is that annihilation tracks all come from ONE point on the trap wall.
"""
from dataclasses import dataclass, field, replace
import numpy as np

# >>> If your Step-4 file has a different name, change THIS ONE LINE. <<<
from hough_detector import find_lines_2d, refit_line_tls

TWO_PI = 2 * np.pi


@dataclass
class Params:
    # ---- geometry ----
    r_vertex: float = 25.0        # mm, trap-wall radius used to extrapolate tracks (CALIBRATE)
    # ---- track finding ----
    phi_bin: float = 0.03         # rad, bin of the 1-D azimuth vote
    phi_window: float = 0.15      # rad, half-width of a sector (must cover track curvature)
    n_phi_peaks: int = 6          # sectors tried per round
    max_tracks: int = 12
    min_hits: int = 6             # minimum spacepoints for a track
    rz_tol: float = 6.0           # mm, inlier distance in the (r,z) view
    rphi_tol: float = 5.0         # mm, inlier distance in the (r, r*dphi) view
    hough_rho_bin: float = 3.0    # mm, accumulator bin ~ half the inlier tolerance (see note)
    hough_peaks: int = 4
    hough_suppress: tuple = (3, 3)
    # ---- vertexing ----
    d0_max: float = 60.0          # mm, track must point this close to the axis (CALIBRATE)
    min_l_extent: float = 40.0    # mm, path-length lever arm needed to extrapolate to the trap
    k_max: float = 5.0            # |dz/dl| limit: steeper tracks cannot constrain z at the wall
    s_z_max: float = 8.0          # mm, max rms of z about the track's (l,z) line (fit quality)
    collinear_rms: float = 5.0    # mm, two tracks on ONE straight 3-D line => through-going muon
    opposite_min: float = 2.0     # rad, ...and on opposite sides of the axis
    sigma_hit: float = 1.0        # mm, floor on per-hit residual
    sigma_z0: float = 4.0         # mm, z error floor of an extrapolated track (CALIBRATE)
    sigma_l: float = 10.0         # mm, path-length error of the extrapolation; z error = k*sigma_l (CALIBRATE)
    sigma_xy: float = 10.0        # mm, xy error of the extrapolated wall point (CALIBRATE)
    pull_max: float = 3.0         # tracks must meet at one 3-D point within this many sigma
    min_vertex_tracks: int = 2
    # ---- per-track tag (your Step-4 rule, now in mm) ----
    cosmic_angle_max: float = 0.35
    cosmic_length_min: float = 200.0


# --------------------------------------------------------------------------------------
# helpers
# --------------------------------------------------------------------------------------
def to_cyl(pts):
    x, y, z = pts[:, 0], pts[:, 1], pts[:, 2]
    return np.hypot(x, y), np.arctan2(y, x) % TWO_PI, z


def circ_diff(a, b):
    """a - b wrapped to (-pi, pi]."""
    return (a - b + np.pi) % TWO_PI - np.pi


@dataclass
class Track:
    """A track = a straight line in the xy plane (normal form n.p = rho) plus z linear in the
    path length l measured along that line from the foot of the perpendicular to the axis."""
    hits: np.ndarray
    n: int = 0
    z_bar: float = 0.0
    l_bar: float = 0.0
    l_extent: float = 0.0
    k: float = np.nan              # dz/dl
    s_z: float = np.nan            # residual of z about the (l,z) line (mm)
    sxx: float = np.nan            # sum (l - l_bar)^2
    d0: float = np.inf             # mm, distance of the xy line from the z axis
    xy_theta: float = 0.0
    xy_rho: float = 0.0
    r_mean: float = 0.0
    angle_from_z: float = np.nan
    length: float = 0.0
    extrapolable: bool = False
    cosmic_like: bool = False      # your Step-4 rule, kept as a tag
    through_going: bool = False    # set by the collinearity veto

    def wall_candidates(self, P):
        """A straight line through the trap region crosses the wall cylinder r = r_vertex at TWO
        points; the pion may have started at either (inward-emitted pions cross the hollow and
        come out the far side). So return both 3-D candidates as (xy, z, sigma_z); the vertex
        finder lets the OTHER tracks decide which one is right."""
        half = np.sqrt(max(P.r_vertex**2 - self.d0**2, 0.0))
        ls = [half, -half] if half > 1e-6 else [0.0]
        nvec = np.array([np.cos(self.xy_theta), np.sin(self.xy_theta)])
        uvec = np.array([-np.sin(self.xy_theta), np.cos(self.xy_theta)])
        out = []
        for l_v in ls:
            z = self.z_bar + self.k * (l_v - self.l_bar)
            var = self.s_z**2 * (1.0 / self.n + (l_v - self.l_bar) ** 2 / self.sxx)
            sig = np.sqrt(var + P.sigma_z0**2 + (self.k * P.sigma_l) ** 2)
            out.append((self.xy_rho * nvec + l_v * uvec, float(z), float(sig)))
        return out


@dataclass
class Vertex:
    z: float
    sigma: float
    n_tracks: int
    chi2_ndf: float
    track_ids: list
    xy: np.ndarray = None


@dataclass
class Result:
    tracks: list
    vertex: Vertex = None
    label: str = 'background'
    reason: str = ''


# --------------------------------------------------------------------------------------
# 1. TRACK FINDING
# --------------------------------------------------------------------------------------
def _phi_centres(phi, P):
    """1-D Hough vote in azimuth: circular histogram, smoothed by the sector width."""
    nb = int(round(TWO_PI / P.phi_bin))
    w = max(1, int(round(P.phi_window / P.phi_bin)))
    h, _ = np.histogram(phi, bins=nb, range=(0, TWO_PI))
    hp = np.concatenate([h[-w:], h, h[:w]]).astype(float)
    sm = np.convolve(hp, np.ones(2 * w + 1), mode='valid')
    centres = []
    for _ in range(P.n_phi_peaks):
        i = int(np.argmax(sm))
        if sm[i] < P.min_hits:
            break
        centres.append((i + 0.5) * TWO_PI / nb)
        sm[np.arange(i - w, i + w + 1) % nb] = 0
    return centres


def _phi_consistent(hits, r, phi, c, P):
    """Second view: arc displacement s = r*dphi must be a straight line in r.
    Iteratively drops outliers; returns surviving hit indices or None."""
    h = hits
    for _ in range(4):
        if len(h) < P.min_hits:
            return None
        s = r[h] * circ_diff(phi[h], c)
        A = np.column_stack([np.ones(len(h)), r[h]])
        coef = np.linalg.lstsq(A, s, rcond=None)[0]
        res = s - A @ coef
        keep = np.abs(res) < P.rphi_tol
        if keep.all():
            return h
        h = h[keep]
    return h if len(h) >= P.min_hits else None


def _best_track_in_sector(r, phi, z, sel, c, P):
    u, v = r[sel] - r[sel].mean(), z[sel] - z[sel].mean()     # centre the cloud
    # Choose the accumulator size so a bin is ~hough_rho_bin mm wide in rho AND the angular
    # bin moves the line's far end by about the same amount. (Fixed n_rho/n_theta would make
    # bins far finer than the hit noise for small clouds -> votes smear -> no peak.)
    max_r = np.sqrt(np.max(u**2 + v**2)) * 1.05 + 1e-6
    n_rho = int(np.clip(np.ceil(2 * max_r / P.hough_rho_bin), 40, 2000))
    n_theta = int(np.clip(np.ceil(np.pi * max_r / P.hough_rho_bin), 90, 720))
    lines, *_ = find_lines_2d(u, v, tol=P.rz_tol, n_theta=n_theta, n_rho=n_rho,
                              n_peaks=P.hough_peaks, min_votes=P.min_hits,
                              suppress_bins=P.hough_suppress)
    best = None
    for ln in lines:
        h = _phi_consistent(sel[ln['mask']], r, phi, c, P)
        if h is not None and (best is None or len(h) > len(best)):
            best = h
    return best


def find_tracks(pts, P):
    r, phi, z = to_cyl(pts)
    remaining = np.ones(len(pts), bool)
    found = []
    for _ in range(P.max_tracks):
        idx = np.flatnonzero(remaining)
        if len(idx) < P.min_hits:
            break
        got = None
        for c in _phi_centres(phi[idx], P):
            sel = idx[np.abs(circ_diff(phi[idx], c)) <= P.phi_window]
            if len(sel) < P.min_hits:
                continue
            c = (c + circ_diff(phi[sel], c).mean()) % TWO_PI      # re-centre on the hits
            sel = idx[np.abs(circ_diff(phi[idx], c)) <= P.phi_window]
            got = _best_track_in_sector(r, phi, z, sel, c, P)
            if got is not None:
                break
        if got is None:
            break
        found.append(got)
        remaining[got] = False
    return found


# --------------------------------------------------------------------------------------
# 2. TRACK FITTING
# --------------------------------------------------------------------------------------
def _line_rms3d(p):
    """RMS perpendicular distance of 3-D points to their best-fit straight line (PCA)."""
    c = p - p.mean(0)
    w = np.linalg.eigvalsh(c.T @ c)
    return float(np.sqrt((w[0] + w[1]) / len(p)))


def fit_track(pts, hits, P):
    p = pts[hits]
    x, y, z = p.T
    n = len(hits)
    t = Track(hits=hits, n=n, z_bar=z.mean(), r_mean=float(np.hypot(x, y).mean()))

    # (x, y): straight line. d0 = does it point back at the axis?
    if np.hypot(np.ptp(x), np.ptp(y)) > 15.0:
        t.xy_theta, t.xy_rho = refit_line_tls(x, y)
        t.d0 = abs(t.xy_rho)
        l = -x * np.sin(t.xy_theta) + y * np.cos(t.xy_theta)     # path length along the line
        t.l_bar = l.mean()
        t.l_extent = np.percentile(l, 95) - np.percentile(l, 5)
        t.sxx = max(np.sum((l - t.l_bar) ** 2), 1e-9)
        t.k = np.sum((l - t.l_bar) * (z - t.z_bar)) / t.sxx       # OLS slope dz/dl
        res = z - t.z_bar - t.k * (l - t.l_bar)
        t.s_z = max(np.sqrt(np.sum(res**2) / max(n - 2, 1)), P.sigma_hit)
        t.extrapolable = bool(t.l_extent >= P.min_l_extent)
    else:                      # (almost) vertical: xy direction meaningless, cannot extrapolate
        t.d0 = t.r_mean

    # 3-D direction (PCA) for the Step-4-style cosmic tag
    c = p - p.mean(0)
    w, V = np.linalg.eigh(c.T @ c)
    d = V[:, -1]
    t.angle_from_z = float(np.arccos(np.clip(abs(d[2]), 0, 1)))
    t.length = float(np.ptp(c @ d))
    t.cosmic_like = bool(t.angle_from_z < P.cosmic_angle_max and t.length > P.cosmic_length_min)
    return t


def _mean_phi(p):
    return np.arctan2(np.mean(p[:, 1] / np.hypot(p[:, 0], p[:, 1])),
                      np.mean(p[:, 0] / np.hypot(p[:, 0], p[:, 1])))


def mark_through_going(tracks, pts, P):
    """A straight muon that crosses the hollow centre is recorded as TWO half-tracks on OPPOSITE
    sides of the axis, each of which 'points at the trap' and looks like an outgoing pion.
    They give themselves away by lying on ONE straight 3-D line. Pions from a vertex do not
    (different directions, opposite curvature, ...). The 'opposite sides' condition stops us
    from vetoing two fragments of the same pion."""
    mphi = [_mean_phi(pts[t.hits]) for t in tracks]
    for i in range(len(tracks)):
        for j in range(i + 1, len(tracks)):
            if abs(circ_diff(mphi[i], mphi[j])) < P.opposite_min:
                continue
            h = np.concatenate([tracks[i].hits, tracks[j].hits])
            if _line_rms3d(pts[h]) < P.collinear_rms:
                tracks[i].through_going = tracks[j].through_going = True


# --------------------------------------------------------------------------------------
# 3. VERTEX + 4. CLASSIFICATION   (cheap: re-run with different cuts without re-tracking)
# --------------------------------------------------------------------------------------
def is_usable(t, P):
    """Quality gate: only tracks that can really constrain a vertex may vote for one."""
    return bool(t.extrapolable and abs(t.k) <= P.k_max and t.s_z <= P.s_z_max
                and t.d0 <= P.d0_max and not t.through_going)


def find_vertex(tracks, P):
    ids = [i for i, t in enumerate(tracks) if is_usable(t, P)]
    if len(ids) < P.min_vertex_tracks:
        return None, f'only {len(ids)} usable track(s) (extrapolable, good fit, pointing at axis, not through-going)'
    cands = {i: tracks[i].wall_candidates(P) for i in ids}

    def chi2(a, b):
        return (np.sum((a[0] - b[0]) ** 2) / (2 * P.sigma_xy**2)
                + (a[1] - b[1]) ** 2 / (a[2] ** 2 + b[2] ** 2))

    best = None
    for i in ids:                                  # seed on every candidate of every track
        for a in cands[i]:
            members, tot = [(i, a)], 0.0
            for j in ids:
                if j == i:
                    continue
                c2, b_ = min(((chi2(a, b), b) for b in cands[j]), key=lambda q: q[0])
                if c2 < P.pull_max**2:
                    members.append((j, b_)); tot += c2
            score = (len(members), -tot)
            if best is None or score > best[0]:
                best = (score, members)
    members = best[1]
    if len(members) < P.min_vertex_tracks:
        return None, 'tracks do not meet at a common point on the trap wall'
    z = np.array([m[1][1] for m in members]); s_ = np.array([m[1][2] for m in members])
    w = 1 / s_**2
    zbar = float(np.sum(w * z) / np.sum(w))
    xy = np.mean([m[1][0] for m in members], axis=0)
    chi2_ndf = float(np.sum(((z - zbar) / s_) ** 2) / max(len(members) - 1, 1))
    return Vertex(zbar, float(1 / np.sqrt(np.sum(w))), len(members), chi2_ndf,
                  [m[0] for m in members], xy), ''


def classify(tracks, P):
    if not tracks:
        return Result(tracks, None, 'background', 'no tracks found')
    v, why = find_vertex(tracks, P)
    if v is None:
        return Result(tracks, None, 'background', why)
    return Result(tracks, v, 'annihilation', '')


def build_tracks(pts, P=Params()):
    """The EXPENSIVE half: find + fit + veto. Cache this, then re-run classify() with
    different cuts for free (that is how the ROC scan in ag_evaluate works)."""
    pts = np.asarray(pts, float)
    if len(pts) < P.min_hits:
        return []
    tracks = [fit_track(pts, h, P) for h in find_tracks(pts, P)]
    mark_through_going(tracks, pts, P)
    return tracks


def reconstruct(pts, P=Params()):
    """The whole pipeline for one event."""
    return classify(build_tracks(pts, P), P)