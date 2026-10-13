import sys
import numpy as np
from hough_detector import find_lines_2d

WIRE_MM = 3.0
TICK_MM_DEFAULT = 0.55
BIN_MM, TOL_MM = 6.0, 8.0
MIN_HITS, MIN_LEN_MM, MIN_OCC, MAX_ITER = 100, 500.0, 0.6, 12
NU_FRAC = 0.5


def load_event(f, i):
    def span(t):
        c = f[f'{t}/event_id.seq_cnt'][:, 1]; return int(c[:i].sum()), int(c[i])
    s, n = span('hit_table')
    g = lambda k: f[f'hit_table/{k}'][s:s + n, 0]
    plane = g('local_plane'); m = plane == 2
    hid, w, t = g('hit_id')[m], g('local_wire')[m], g('local_time')[m]
    es, en = span('edep_table')
    eh = f['edep_table/hit_id'][es:es + en, 0]; ef = f['edep_table/energy_fraction'][es:es + en, 0]
    allhid = g('hit_id'); pos = np.full(allhid.max() + 1, -1); pos[allhid] = np.arange(len(allhid))
    frac_all = np.zeros(len(allhid)); ok = eh < len(pos)
    idx = pos[eh[ok]]; good = idx >= 0
    np.add.at(frac_all, idx[good], ef[ok][good])
    ps, pn = span('particle_table')
    prim = f['particle_table/parent_id'][ps:ps + pn, 0] == 0
    vw = f['particle_table/start_wire_pos'][ps:ps + pn][prim][:, 2].mean()
    vt = f['particle_table/start_wire_time'][ps:ps + pn, 0][prim].mean()
    return w, t, frac_all[m], (vw, vt)


def find_long_lines(x, y):
    idx = np.arange(len(x)); lines = []; banned = np.zeros(len(x), bool)
    for _ in range(MAX_ITER):
        avail = idx[~banned]
        if len(avail) < MIN_HITS:
            break
        u, v = x[avail] - x[avail].mean(), y[avail] - y[avail].mean()
        m = np.sqrt(np.max(u**2 + v**2)) * 1.05 + 1e-6
        n_rho = int(np.clip(2 * m / BIN_MM, 40, 4000)); n_th = int(np.clip(np.pi * m / BIN_MM, 90, 720))
        found, *_ = find_lines_2d(u, v, tol=TOL_MM, n_theta=n_th, n_rho=n_rho, n_peaks=1, min_votes=MIN_HITS)
        if not found:
            break
        ln = found[0]; hits = avail[ln['mask']]
        s = -u[ln['mask']] * np.sin(ln['theta']) + v[ln['mask']] * np.cos(ln['theta'])
        lo, hi = np.percentile(s, [2, 98]); length = hi - lo
        occ = len(np.unique(np.floor((s - lo) / 10.0))) / max(length / 10.0, 1.0)
        if len(hits) >= MIN_HITS and length >= MIN_LEN_MM and occ >= MIN_OCC:
            lines.append(hits); banned[hits] = True
        else:
            banned[hits] = True
    return lines


def score(f, i, tick_mm):
    w, t, frac, (vw, vt) = load_event(f, i)
    x, y = w * WIRE_MM, t * tick_mm
    lines = find_long_lines(x, y)
    on = np.zeros(len(x), bool)
    for h in lines:
        on[h] = True
    nu = frac >= NU_FRAC
    cos = ~nu
    pur = [np.mean(~nu[h]) for h in lines]
    return dict(n_nu=int(nu.sum()), n_cos=int(cos.sum()), n_lines=len(lines),
                capture=on[cos].sum() / max(cos.sum(), 1), contam=on[nu].sum() / max(nu.sum(), 1) if nu.sum() else np.nan,
                cos_lines=sum(p >= 0.9 for p in pur), x=x, y=y, nu=nu, on=on, lines=lines, vtx=(vw * WIRE_MM, vt * tick_mm))


def mean_ci(a):
    a = np.asarray(a, float); a = a[np.isfinite(a)]
    return a.mean(), 1.96 * a.std(ddof=1) / np.sqrt(len(a)), len(a)


def sanity(f, n=5):
    print('Label check: neutrino-labelled hits should sit near the true primary vertex')
    for i in range(n):
        r = score(f, i, TICK_MM_DEFAULT)
        d = np.hypot(r['x'] - r['vtx'][0], r['y'] - r['vtx'][1])
        print(f'  event {i}: {r["n_nu"]} neutrino hits; {np.mean(d[r["nu"]] < 1500):.2f} of them within 1.5 m of the vertex '
              f'(cosmic hits: {np.mean(d[~r["nu"]] < 1500):.2f})')


def main():
    import h5py
    dev, test = h5py.File(sys.argv[1], 'r'), h5py.File(sys.argv[2], 'r')
    n = int(sys.argv[sys.argv.index('--n') + 1]) if '--n' in sys.argv else 50
    nfig = int(sys.argv[sys.argv.index('--figs') + 1]) if '--figs' in sys.argv else 0
    print('--- label sanity on the development file'); sanity(dev)
    print('\n--- do the two files share cosmic backgrounds? (first 15 events each, plane 2)')
    def hs(f, i):
        w, t, _, _ = load_event(f, i); return set(zip(w.tolist(), np.round(t, 1).tolist()))
    A = [hs(dev, i) for i in range(15)]; B = [hs(test, i) for i in range(15)]
    worst = max(len(a & b) / max(min(len(a), len(b)), 1) for a in A for b in B)
    print(f'   largest overlap between any dev event and any test event: {worst:.3f} of the smaller event')
    for tick in (0.4, TICK_MM_DEFAULT, 0.7):
        res = [score(test, i, tick) for i in range(n)]
        print(f'\n=== TEST FILE, {n} events, tick = {tick} mm  (events are the independent units; mean +- 95% CI) ===')
        for key, label in (('capture', 'cosmic hits found on long lines'), ('contam', 'neutrino hits wrongly on long lines')):
            m, ci, k = mean_ci([r[key] for r in res]); print(f'   {label:40s} {m:.3f} +- {ci:.3f}   ({k} events)')
        tot_l = sum(r['n_lines'] for r in res); tot_c = sum(r['cos_lines'] for r in res)
        print(f'   long lines found: {tot_l}, of which mostly cosmic (>=90% non-neutrino hits): {tot_c} ({tot_c / max(tot_l, 1):.3f})')
        if tick == TICK_MM_DEFAULT and nfig:
            import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
            for i in range(nfig):
                r = res[i]; plt.figure(figsize=(12, 5))
                plt.scatter(r['x'] / WIRE_MM, r['y'] / tick, s=2, c='lightgray', label='other hits')
                for k, h in enumerate(r['lines']):
                    plt.scatter(r['x'][h] / WIRE_MM, r['y'][h] / tick, s=3, label='Hough long line' if k == 0 else None, c=f'C{k % 10}')
                plt.scatter(r['x'][r['nu']] / WIRE_MM, r['y'][r['nu']] / tick, s=4, c='red', label='neutrino hits (simulation truth)')
                plt.xlabel('wire (plane 2)'); plt.ylabel('time tick'); plt.title(f'test event {i}'); plt.legend(loc='upper right')
                plt.savefig(f'ub_fig_{i}.png', dpi=150); plt.close()
            print(f'   saved {nfig} figure(s): ub_fig_*.png')


if __name__ == '__main__':
    main()