import sys
import numpy as np
import pipeline as sp
import ag_data as D
import ag_synth as syn

GRID = [40, 50, 60, 70, 80, 100, 130, 170]


def rate(events, d0):
    cfg = dict(sp.CFG, D0_MAX=float(d0))
    return sum(sp.classify(e['pts'], cfg)[0] == 'annihilation' for e in events) / len(events)


def main():
    if len(sys.argv) < 3:
        print('usage: python run.py inspect FILE.h5   |   python run.py run VAL.h5 TEST.h5 --n 500')
        return
    mode, n = sys.argv[1], 500
    if '--n' in sys.argv:
        n = int(sys.argv[sys.argv.index('--n') + 1])
    if mode == 'inspect':
        D.inspect_h5(sys.argv[2]); return

    cal = D.load_events(sys.argv[2], 0, n)
    ev = D.load_events(sys.argv[3], 0, n)
    g = D.estimate_geometry(cal)
    print('\nMeasured geometry from the data (mm):', {k: round(v) for k, v in g.items()})
    rng = np.random.default_rng(0)
    det = syn.Det(r_in=g['r_in'], r_out=g['r_out'], half_len=g['half_len'])

    # ---- choose D0_MAX on the VALIDATION file only
    m = min(300, len(cal))
    cos_cal = [syn.make_cosmic_event(rng, det, 1 + i % 3) for i in range(m)]
    print(f'\nChoosing D0_MAX on {m} validation events (signal) + {m} synthetic cosmics:')
    print('  D0_MAX | signal eff | cosmic leakage')
    table = []
    for d0 in GRID:
        s, c = rate(cal[:m], d0), rate(cos_cal, d0)
        table.append((d0, s, c)); print(f'  {d0:6d} | {s:10.3f} | {c:.3f}')
    ok = [t for t in table if t[2] <= 0.01]
    best = max(ok, key=lambda t: t[1]) if ok else min(table, key=lambda t: t[2])
    if not ok:
        print('  WARNING: no cut reached 1% leakage; using the tightest one. Tell your advisor / me.')
    d0max = best[0]
    print(f'  -> chosen D0_MAX = {d0max} mm  (rule: highest signal efficiency with leakage <= 1%)')
    cfg = dict(sp.CFG, D0_MAX=float(d0max))

    print(f'\n=== TEST FILE: {len(ev)} annihilation events (not used for any choice) ===')
    out = [sp.classify(e['pts'], cfg) for e in ev]
    sp.report('signal efficiency', sum(o[0] == 'annihilation' for o in out), len(ev))
    res = np.array([o[1] - e['vz_true'] for o, e in zip(out, ev) if o[1] is not None])
    s68 = lambda a: 0.5 * np.subtract(*np.percentile(a, [84, 16]))
    print(f'vertex z error (mine): median {np.median(res):+.1f} mm, sigma68 {s68(res):.1f} mm, '
          f'95% within {np.percentile(abs(res), 95):.0f} mm')
    both = [(o[1] - e['vz_true'], e['vz_helix'] - e['vz_true']) for o, e in zip(out, ev)
            if o[1] is not None and np.isfinite(e['vz_helix'])]
    if both:
        b = np.array(both)
        print(f'same events, experiment Helix Fit: median {np.median(b[:, 1]):+.1f} mm, sigma68 {s68(b[:, 1]):.1f} mm')
    print(f'Helix Fit failures in test set: {np.mean([not np.isfinite(e["vz_helix"]) for e in ev]):.3f}')

    cos = [syn.make_cosmic_event(rng, det, 1 + i % 3) for i in range(len(ev))]
    print('\n=== SYNTHETIC cosmics (optimistic bound, the open data has none) ===')
    sp.report('cosmic leakage', sum(sp.classify(e['pts'], cfg)[0] == 'annihilation' for e in cos), len(cos))
    for k in (1, 2):
        mix = [syn.overlay_cosmics(rng, e, det, k) for e in ev[:300]]
        sp.report(f'real annihilation + {k} cosmic(s): kept',
                  sum(sp.classify(e['pts'], cfg)[0] == 'annihilation' for e in mix), 300)


if __name__ == '__main__':
    main()