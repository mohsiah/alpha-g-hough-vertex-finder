"""
pipeline.py -- the SIMPLE, EXPLAINABLE version of the final algorithm.

THE WHOLE IDEA IN FOUR SENTENCES:
  1. Every annihilation track starts at ONE point on the trap wall, very close to the central axis.
  2. A cosmic muon is one straight line that does NOT point at the axis.
  3. So: find the tracks (Hough), fit each with a straight line, and keep only tracks whose line
     passes close to the axis.
  4. An event is an annihilation if at least TWO such tracks reach the axis at the SAME height z.

Every number in CFG has a physical meaning, written next to it.
"""
import numpy as np
from hough_detector import find_lines_2d, refit_line_tls

CFG = dict(
    R_VERTEX=25.0,     # mm  radius of the trap wall where annihilations happen
    D0_MAX=60.0,       # mm  a track "points at the axis" if its line passes closer than this
    PHI_WINDOW=0.15,   # rad half-width of an angular slice (tracks leave the axis -> same angle)
    PHI_BIN=0.03,      # rad bin of the angle histogram
    MIN_HITS=6,        # a track needs at least this many spacepoints
    TOL_RZ=6.0,        # mm  hit is "on the Hough line" if within this distance in the (r,z) view
    TOL_XY=5.0,        # mm  ...and within this of the straight line in the (x,y) view (2nd view)
    MIN_LEN=40.0,      # mm  track must be at least this long (in x-y) to be trusted
    MAX_SLOPE=5.0,     # |dz/dl| steeper than this cannot fix z -> track may not vote
    MAX_RMS_Z=8.0,     # mm  fit quality: z scatter about the track's line
    SIGMA_Z=3.0,       # mm  base z error of a track
    NSIG=3.0,          # two tracks "agree" if their z differ by < NSIG standard errors
    COLLINEAR=5.0,     # mm  two half-tracks on ONE straight 3-D line = a muon through the hole
)
TWO_PI = 2 * np.pi


def wrap(a):                                   # angle difference wrapped into (-pi, pi]
    return (a + np.pi) % TWO_PI - np.pi


# ---------------------------------------------------------------- 1. find tracks
def angle_peaks(phi, c):
    """Histogram of hit angles. Each track piles hits into a few neighbouring bins."""
    nb = int(round(TWO_PI / c['PHI_BIN'])); w = max(1, int(round(c['PHI_WINDOW'] / c['PHI_BIN'])))
    h, _ = np.histogram(phi, bins=nb, range=(0, TWO_PI))
    smooth = np.convolve(np.concatenate([h[-w:], h, h[:w]]), np.ones(2 * w + 1), mode='valid').astype(float)
    peaks = []
    for _ in range(6):
        i = int(np.argmax(smooth))
        if smooth[i] < c['MIN_HITS']:
            break
        peaks.append((i + 0.5) * TWO_PI / nb)
        smooth[np.arange(i - w, i + w + 1) % nb] = 0     # don't pick the same pile twice
    return peaks


def xy_line_rms(x, y):
    """Straight-line fit in x-y (your refit_line_tls); returns (theta, rho, perpendicular distances)."""
    theta, rho = refit_line_tls(x, y)
    return theta, rho, x * np.cos(theta) + y * np.sin(theta) - rho


def find_tracks(pts, c=CFG):
    x, y, z = pts.T
    r, phi = np.hypot(x, y), np.arctan2(y, x) % TWO_PI
    left = np.ones(len(pts), bool); tracks = []
    for _ in range(12):
        idx = np.flatnonzero(left)
        if len(idx) < c['MIN_HITS']:
            break
        found = None
        for centre in angle_peaks(phi[idx], c):
            sl = idx[np.abs(wrap(phi[idx] - centre)) <= c['PHI_WINDOW']]       # one angular slice
            if len(sl) < c['MIN_HITS']:
                continue
            u, v = r[sl] - r[sl].mean(), z[sl] - z[sl].mean()
            m = np.sqrt(np.max(u**2 + v**2)) * 1.05 + 1e-6                     # Hough in the (r,z) plane
            n_rho = int(np.clip(2 * m / 3.0, 40, 2000)); n_th = int(np.clip(np.pi * m / 3.0, 90, 720))
            lines, *_ = find_lines_2d(u, v, tol=c['TOL_RZ'], n_theta=n_th, n_rho=n_rho, n_peaks=4,
                                      min_votes=c['MIN_HITS'], suppress_bins=(3, 3))
            for ln in lines:                                                   # 2nd view: x-y straightness
                h = sl[ln['mask']]
                for _ in range(4):
                    if len(h) < c['MIN_HITS']:
                        break
                    _, _, d = xy_line_rms(x[h], y[h])
                    keep = np.abs(d) < c['TOL_XY']
                    if keep.all():
                        break
                    h = h[keep]
                if len(h) >= c['MIN_HITS'] and (found is None or len(h) > len(found)):
                    found = h
            if found is not None:
                break
        if found is None:
            break
        tracks.append(found); left[found] = False
    return tracks


# ---------------------------------------------------------------- 2. fit each track
def fit_track(pts, hits, c=CFG):
    x, y, z = pts[hits].T
    t = dict(hits=hits, n=len(hits), usable=False, thru=False)
    if np.hypot(np.ptp(x), np.ptp(y)) < 15:                  # vertical: x-y direction is meaningless
        return t
    theta, rho, _ = xy_line_rms(x, y)
    t['d0'] = abs(rho)                                       # distance of the line from the axis
    l = -x * np.sin(theta) + y * np.cos(theta)               # position ALONG the line (0 = closest to axis)
    lbar, zbar = l.mean(), z.mean()
    sxx = np.sum((l - lbar) ** 2)
    k = np.sum((l - lbar) * (z - zbar)) / sxx                # slope dz/dl
    s_z = max(np.sqrt(np.sum((z - zbar - k * (l - lbar)) ** 2) / max(len(z) - 2, 1)), 1.0)
    t['z0'] = zbar - k * lbar                                # z where the track is closest to the axis
    # error on z0: line-fit error + "the vertex may be up to R_VERTEX of path away from the closest point"
    t['sig'] = np.sqrt(c['SIGMA_Z'] ** 2 + s_z**2 * (1 / len(z) + lbar**2 / sxx) + (k * c['R_VERTEX']) ** 2)
    t['usable'] = bool(np.percentile(l, 95) - np.percentile(l, 5) >= c['MIN_LEN'] and abs(k) <= c['MAX_SLOPE']
                       and s_z <= c['MAX_RMS_Z'] and t['d0'] <= c['D0_MAX'])
    t['phi'] = np.arctan2(np.mean(np.sin(np.arctan2(y, x))), np.mean(np.cos(np.arctan2(y, x))))
    return t


def line3d_rms(p):                                            # scatter of 3-D points about their best line
    w = np.linalg.eigvalsh(np.cov(p.T) * (len(p) - 1))
    return np.sqrt((w[0] + w[1]) / len(p))


def mark_through_going(tracks, pts, c=CFG):
    """Muon crossing the hollow centre = 2 half-tracks on opposite sides, lying on ONE 3-D line."""
    for i in range(len(tracks)):
        for j in range(i + 1, len(tracks)):
            if 'phi' in tracks[i] and 'phi' in tracks[j] and abs(wrap(tracks[i]['phi'] - tracks[j]['phi'])) > 2.0:
                if line3d_rms(pts[np.concatenate([tracks[i]['hits'], tracks[j]['hits']])]) < c['COLLINEAR']:
                    tracks[i]['thru'] = tracks[j]['thru'] = True


# ---------------------------------------------------------------- 3. decide
def classify(pts, c=CFG):
    tracks = [fit_track(pts, h, c) for h in find_tracks(pts, c)]
    mark_through_going(tracks, pts, c)
    good = [t for t in tracks if t['usable'] and not t['thru']]
    best = []
    for a in good:                                            # largest group of tracks that agree on z
        grp = [b for b in good if abs(a['z0'] - b['z0']) < c['NSIG'] * np.hypot(a['sig'], b['sig'])]
        if len(grp) > len(best):
            best = grp
    if len(best) >= 2:
        w = np.array([1 / t['sig'] ** 2 for t in best])
        return 'annihilation', float(np.sum(w * [t['z0'] for t in best]) / w.sum()), tracks
    return 'background', None, tracks


# ---------------------------------------------------------------- 4. check it
def wilson(k, n, z=1.96):
    p = k / n; d = 1 + z * z / n; m = (p + z * z / (2 * n)) / d
    h = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return max(0, m - h), min(1, m + h)


def report(name, k, n):
    lo, hi = wilson(k, n)
    print(f'{name:42s} {k:5d}/{n:<5d} = {k / n:.4f}   95% CI [{lo:.4f}, {hi:.4f}]' + (f'   (0 seen: 95% upper limit 3/N = {3 / n:.4f})' if k == 0 else ''))


if __name__ == '__main__':
    import ag_synth as syn
    rng, det = np.random.default_rng(21), syn.Det()
    N = 2000
    sig = [syn.make_annihilation(rng, det) for _ in range(N)]
    out = [classify(e['pts']) for e in sig]
    ok = [o[0] == 'annihilation' for o in out]
    report('signal efficiency (all events)', sum(ok), N)
    res = np.array([o[1] - e['vz_true'] for o, e in zip(out, sig) if o[1] is not None])
    print(f'vertex z error: median {np.median(res):+.1f} mm, sigma68 {0.5 * np.subtract(*np.percentile(res, [84, 16])):.1f} mm, '
          f'95% within {np.percentile(abs(res), 95):.0f} mm, worst {abs(res).max():.0f} mm')
    cos = [syn.make_cosmic_event(rng, det, 1 + i % 3) for i in range(N)]
    report('cosmic leakage (1-3 muons/event)', sum(classify(e['pts'])[0] == 'annihilation' for e in cos), N)
    for m in (1, 2):
        mix = [syn.overlay_cosmics(rng, e, det, m) for e in sig[:500]]
        report(f'annihilation + {m} cosmic overlay: kept', sum(classify(e['pts'])[0] == 'annihilation' for e in mix), 500)
