"""
ag_evaluate.py -- the test. Nothing here is part of the algorithm; it only
MEASURES the algorithm.
Usage
-----
  python ag_evaluate.py --synthetic                         # full report on the testbed
  python ag_evaluate.py --h5 test.h5 --cal-h5 val.h5        # real simulation data (Zenodo)
  python ag_evaluate.py --h5 test.h5 --cal-h5 val.h5 --show # + event displays

"""
import argparse
import numpy as np
from dataclasses import replace

from ag_pipeline import Params, build_tracks, classify, is_usable
import ag_synth as syn


# ----------------------------------------------------------------------------- statistics
def wilson(k, n, z=1.96):
    """95% Wilson score interval for a binomial fraction. Works for k=0 and k=n."""
    if n == 0:
        return (np.nan, np.nan)
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return max(0.0, c - h), min(1.0, c + h)


def fmt_frac(k, n):
    lo, hi = wilson(k, n)
    extra = f'   (zero seen -> 95% upper limit ~ 3/N = {3 / n:.2e})' if k == 0 and n else ''
    return f'{k}/{n} = {k / max(n, 1):.4f}  [95% CI {lo:.4f}, {hi:.4f}]{extra}'


def res_stats(res):
    res = np.asarray(res)
    if len(res) == 0:
        return 'n=0'
    s68 = 0.5 * (np.percentile(res, 84) - np.percentile(res, 16))
    return (f'n={len(res)}  median={np.median(res):+.2f}  sigma68={s68:.2f}  '
            f'95%|res|={np.percentile(np.abs(res), 95):.1f}  frac>|50mm|={np.mean(np.abs(res) > 50):.4f}  mm')


# ----------------------------------------------------------------------------- running
def cache_tracks(events, P):
    return [build_tracks(e['pts'], P) for e in events]


def labels(cached, P):
    return [classify(t, P) for t in cached]


# ----------------------------------------------------------------------------- calibration
def calibrate(events, P, verbose=True):
    """Set the few geometry/error parameters from events that have a TRUTH vertex z.
    Uses only z_true (which the real Zenodo files provide) -- no per-track truth needed."""
    cached = cache_tracks(events, P)
    zt = np.array([e['vz_true'] for e in events])

    # (1) r_vertex: the wall radius that makes the single-track z estimates most self-consistent
    best = None
    for rv in np.arange(0.0, 61.0, 5.0):
        Pt = replace(P, r_vertex=float(rv))
        err = []
        for tr, z0 in zip(cached, zt):
            for t in tr:
                if is_usable(t, Pt):
                    err.append(min(abs(c[1] - z0) for c in t.wall_candidates(Pt)))
        if len(err) > 50:
            m = np.median(err)
            if best is None or m < best[0]:
                best = (m, rv)
    P = replace(P, r_vertex=float(best[1]))

    # (2) error model sigma_total^2 = sigma_z0^2 + (|k| sigma_l)^2 from the nearest-candidate error
    rows = []
    for tr, z0 in zip(cached, zt):
        for t in tr:
            if is_usable(t, P):
                rows.append((abs(t.k), min((c[1] - z0 for c in t.wall_candidates(P)), key=abs)))
    A = np.array(rows)
    flat = A[A[:, 0] < 0.3, 1]
    steep = A[A[:, 0] > 0.7]
    s68 = lambda e: 0.5 * (np.percentile(e, 84) - np.percentile(e, 16))
    sz0 = max(s68(flat), 1.0) if len(flat) > 20 else P.sigma_z0
    sl = float(np.median([s68(steep[(steep[:, 0] >= lo) & (steep[:, 0] < hi), 1]) /
                          steep[(steep[:, 0] >= lo) & (steep[:, 0] < hi), 0].mean()
                          for lo, hi in [(0.7, 1.2), (1.2, 2.0), (2.0, 5.0)]
                          if ((steep[:, 0] >= lo) & (steep[:, 0] < hi)).sum() > 20] or [P.sigma_l]))
    P = replace(P, sigma_z0=float(sz0), sigma_l=sl)

    # (3) d0 cut: just above the 99th percentile of d0 for tracks of events whose vertex is found
    d0s = []
    for tr, z0 in zip(cached, zt):
        res = classify(tr, replace(P, d0_max=1e9))
        if res.vertex is not None and abs(res.vertex.z - z0) < 30:
            d0s += [tr[i].d0 for i in res.vertex.track_ids]
    if len(d0s) > 50:
        P = replace(P, d0_max=float(1.15 * np.percentile(d0s, 99)))
    if verbose:
        print(f'CALIBRATED on {len(events)} events: r_vertex={P.r_vertex:.0f} mm, sigma_z0={P.sigma_z0:.1f}, '
              f'sigma_l={P.sigma_l:.1f}, d0_max={P.d0_max:.0f} mm')
    return P


# ----------------------------------------------------------------------------- the tests
def test_signal(events, P, cached=None):
    cached = cached or cache_tracks(events, P)
    R = labels(cached, P)
    n = len(events)
    passed = [r.label == 'annihilation' for r in R]
    print(f'  signal efficiency, all events      : {fmt_frac(sum(passed), n)}')
    has_truth = [e['n_truth_tracks'] >= 0 for e in events]
    if all(has_truth):
        acc = [e['n_truth_tracks'] >= 2 for e in events]
        k = sum(p for p, a in zip(passed, acc) if a)
        print(f'  ...among events with >=2 visible pion tracks: {fmt_frac(k, sum(acc))}')
        print(f'  (events with <2 visible tracks cannot be reconstructed by ANY method: {n - sum(acc)}/{n})')
    res = [r.vertex.z - e['vz_true'] for r, e in zip(R, events) if r.vertex is not None]
    print('  vertex z residual (this algorithm) :', res_stats(res))
    hz = np.array([e['vz_helix'] for e in events])
    both = [(r.vertex.z - e['vz_true'], e['vz_helix'] - e['vz_true'])
            for r, e in zip(R, events) if r.vertex is not None and np.isfinite(e['vz_helix'])]
    if both:
        b = np.array(both)
        print('  same events, experiment Helix Fit :', res_stats(b[:, 1]))
        print('  same events, this algorithm       :', res_stats(b[:, 0]))
        print(f'  (Helix-Fit exact zeros: {np.mean(hz == 0):.3f} -- if large, zeros encode "failed")')
    reasons = {}
    for r in R:
        if r.label != 'annihilation':
            reasons[r.reason[:60]] = reasons.get(r.reason[:60], 0) + 1
    if reasons:
        print('  why signal events were lost:', dict(sorted(reasons.items(), key=lambda kv: -kv[1])[:4]))
    return cached


def test_background(cosmic_events, P):
    cached = cache_tracks(cosmic_events, P)
    R = labels(cached, P)
    print(f'  pure-cosmic events passing (= leakage): {fmt_frac(sum(r.label == "annihilation" for r in R), len(R))}')
    for m in sorted({e['n_cosmics'] for e in cosmic_events}):
        idx = [i for i, e in enumerate(cosmic_events) if e['n_cosmics'] == m]
        print(f'     {m} muon(s) per event: {fmt_frac(sum(R[i].label == "annihilation" for i in idx), len(idx))}')
    return cached


def test_mixed(events, rng, det, P, n_cosmics=(1, 2)):
    for m in n_cosmics:
        mixed = [syn.overlay_cosmics(rng, e, det, m) for e in events]
        R = labels(cache_tracks(mixed, P), P)
        ok = [r.label == 'annihilation' for r in R]
        res = [r.vertex.z - e['vz_true'] for r, e in zip(R, mixed) if r.vertex is not None]
        print(f'  annihilation + {m} cosmic overlay: eff {fmt_frac(sum(ok), len(ok))}')
        print(f'      z residual: {res_stats(res)}')


def test_null(events, rng, P):
    """Structureless inputs. ANY pass here is a pure false-positive-by-chance estimate."""
    shuffled, noise = [], []
    for e in events:
        p = e['pts'].copy()
        r, ph = np.hypot(p[:, 0], p[:, 1]), rng.uniform(0, 2 * np.pi, len(p))   # destroy phi structure
        shuffled.append(dict(e, pts=np.column_stack([r * np.cos(ph), r * np.sin(ph), p[:, 2]])))
        rr = np.hypot(p[:, 0], p[:, 1])
        u = np.sqrt(rng.uniform(rr.min() ** 2, rr.max() ** 2, len(p))); ph2 = rng.uniform(0, 2 * np.pi, len(p))
        noise.append(dict(e, pts=np.column_stack([u * np.cos(ph2), u * np.sin(ph2),
                                                  rng.uniform(p[:, 2].min(), p[:, 2].max(), len(p))])))
    for name, evs in [('phi-shuffled signal (same r,z, no tracks)', shuffled), ('pure uniform noise', noise)]:
        R = labels(cache_tracks(evs, P), P)
        print(f'  {name}: passing {fmt_frac(sum(r.label == "annihilation" for r in R), len(R))}')


def roc_scan(sig_cached, bkg_cached, P):
    """Sweep the two cuts that matter. Re-uses cached tracks, so it is instant."""
    print('  d0_max  pull_max | signal eff | cosmic pass')
    for d0 in (0.6 * P.d0_max, 0.8 * P.d0_max, P.d0_max, 1.5 * P.d0_max, 2.5 * P.d0_max):
        for pm in (2.0, 3.0, 4.0):
            Q = replace(P, d0_max=d0, pull_max=pm)
            s = np.mean([classify(t, Q).label == 'annihilation' for t in sig_cached])
            b = np.mean([classify(t, Q).label == 'annihilation' for t in bkg_cached])
            print(f'  {d0:6.0f}  {pm:8.1f} | {s:10.3f} | {b:11.4f}')


def show_event(ev, P, title=''):
    import matplotlib.pyplot as plt
    tr = build_tracks(ev['pts'], P); R = classify(tr, P)
    fig = plt.figure(figsize=(7, 7)); ax = fig.add_subplot(111, projection='3d')
    p = ev['pts']; ax.scatter(*p.T, s=6, c='lightgray')
    cols = plt.cm.tab10.colors
    for i, t in enumerate(tr):
        q = p[t.hits]; inv = R.vertex is not None and i in R.vertex.track_ids
        ax.scatter(*q.T, s=14, color=cols[i % 10], marker='o' if inv else 'x')
    if R.vertex is not None:
        ax.scatter(*R.vertex.xy, R.vertex.z, marker='*', s=250, c='gold', edgecolor='k')
    ax.set_title(f'{title} -> {R.label} {R.reason}'); return fig


# ----------------------------------------------------------------------------- main
def make_synthetic_sets(rng, det, n_cal, n_eval):
    cal = [syn.make_annihilation(rng, det) for _ in range(n_cal)]
    sig = [syn.make_annihilation(rng, det) for _ in range(n_eval)]
    bkg = [syn.make_cosmic_event(rng, det, 1 + i % 3) for i in range(n_eval)]
    return cal, sig, bkg


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--synthetic', action='store_true')
    ap.add_argument('--h5'); ap.add_argument('--cal-h5')
    ap.add_argument('--n-cal', type=int, default=1500); ap.add_argument('--n-eval', type=int, default=3000)
    ap.add_argument('--seed', type=int, default=0); ap.add_argument('--show', action='store_true')
    a = ap.parse_args()
    rng = np.random.default_rng(a.seed)
    P = Params()

    if a.synthetic or not a.h5:
        det = syn.Det()
        cal, sig, bkg = make_synthetic_sets(rng, det, a.n_cal, a.n_eval)
        print(f'SYNTHETIC testbed (placeholder geometry): {len(cal)} cal / {len(sig)} signal / {len(bkg)} cosmic events')
    else:
        import ag_data as D
        if a.cal_h5:
            cal = D.load_events(a.cal_h5, 0, a.n_cal); sig = D.load_events(a.h5, 0, a.n_eval)
        else:                                           # disjoint slices of ONE file
            allev = D.load_events(a.h5, 0, a.n_cal + a.n_eval)
            cal, sig = allev[:a.n_cal], allev[a.n_cal:]
        g = D.estimate_geometry(cal)
        print('MEASURED geometry (mm):', {k: round(v) for k, v in g.items()})
        det = syn.Det(r_in=g['r_in'], r_out=g['r_out'], half_len=g['half_len'])
        bkg = [syn.make_cosmic_event(rng, det, 1 + i % 3) for i in range(len(sig))]
        print('NOTE: cosmics are SYNTHETIC (the open data has none) -> rejection numbers are '
              'an optimistic bound.')

    print('\n== 1. Calibration (cal set only) =='); P = calibrate(cal, P)
    print('\n== 2. Signal: held-out annihilation events =='); sc = test_signal(sig, P)
    print('\n== 3. Background: cosmic-only events =='); bc = test_background(bkg, P)
    print('\n== 4. Annihilation with cosmic overlay =='); test_mixed(sig[:min(500, len(sig))], rng, det, P)
    print('\n== 5. Null tests =='); test_null(sig[:min(500, len(sig))], rng, P)
    print('\n== 6. Cut scan (efficiency vs leakage) =='); roc_scan(sc, bc, P)
    if a.show:
        import matplotlib.pyplot as plt
        show_event(sig[0], P, 'signal'); show_event(bkg[0], P, 'cosmic'); plt.show()


if __name__ == '__main__':
    main()
