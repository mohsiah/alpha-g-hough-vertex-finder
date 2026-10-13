import numpy as np
import pipeline as sp
import ag_synth as syn

DET = syn.Det(r_in=109.0, r_out=181.0, half_len=1134.0)
CFG = dict(sp.CFG, D0_MAX=70.0)


def muon(rng, zen='cos2', sxy=1.5, sz=2.0, step=5.0):
    while True:
        c = {'cos2': rng.uniform() ** (1 / 3), 'iso': rng.uniform(), 'horiz': rng.uniform(0, 0.3)}[zen]
        s = np.sqrt(1 - c * c); az = rng.uniform(0, 2 * np.pi)
        d = np.array([s * np.cos(az), s * np.sin(az), -c])
        rr = DET.r_out * np.sqrt(rng.uniform()); ph = rng.uniform(0, 2 * np.pi)
        p0 = np.array([rr * np.cos(ph), rr * np.sin(ph), 0.0]); T = DET.half_len / max(c, 1e-3)
        t = np.arange(-T, T, step); pts = p0 + t[:, None] * d
        r = np.hypot(pts[:, 0], pts[:, 1]); pts = pts[(r >= DET.r_in) & (r <= DET.r_out) & (np.abs(pts[:, 2]) <= DET.half_len)]
        if len(pts) >= 8:
            pts = pts + rng.normal(0, 1, pts.shape) * [sxy, sxy, sz]
            return pts, d


def deltas(rng, pts, n=6):
    out = []
    for _ in range(n):
        base = pts[rng.integers(len(pts))]; dd = rng.normal(size=3); dd /= np.linalg.norm(dd)
        l = np.arange(0, rng.uniform(20, 70), 4.0); out.append(base + l[:, None] * dd + rng.normal(0, 1.5, (len(l), 3)))
    return np.vstack(out)


def shower(rng, pts, n=80, spread=15.0):
    c = pts[rng.integers(len(pts))]; return c + rng.normal(0, spread, (n, 3))


def event(rng, kind):
    n_mu = 1 + rng.integers(3); parts = []
    for _ in range(n_mu):
        zen = {'horizontal-ish (zenith pdf c<0.3)': 'horiz', 'isotropic zenith': 'iso'}.get(kind, 'cos2')
        sxy, sz = (4.0, 5.0) if kind == 'coarse resolution (4/5 mm)' else (1.5, 2.0)
        m, _ = muon(rng, zen, sxy, sz); parts.append(m)
        if kind == 'muon + delta-ray branches': parts.append(deltas(rng, m))
        if kind == 'muon + shower blobs': parts.append(shower(rng, m))
    parts.append(syn.noise_hits(rng, DET, 60.0 if kind == 'heavy noise (60 hits)' else None))
    return np.vstack(parts)


def wilson(k, n, z=1.96):
    p = k / n; d = 1 + z * z / n; m = (p + z * z / (2 * n)) / d; h = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return max(0, m - h), min(1, m + h)


if __name__ == '__main__':
    N = 400; rng = np.random.default_rng(11)
    for kind in ['design case (cos^2, ideal)', 'isotropic zenith', 'horizontal-ish (zenith pdf c<0.3)',
                 'muon + delta-ray branches', 'muon + shower blobs', 'coarse resolution (4/5 mm)', 'heavy noise (60 hits)']:
        k = sum(sp.classify(event(rng, kind), CFG)[0] == 'annihilation' for _ in range(N))
        lo, hi = wilson(k, N); print(f'{kind:36s} leakage {k:3d}/{N} = {k / N:.3f}  [{lo:.3f}, {hi:.3f}]')
