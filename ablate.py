"""
ablate.py -- switches each part of the algorithm off, one at a time, and measures what breaks.
  python ablate.py val.h5 test.h5 --n 3000
(Also prints WHY signal events are lost.)  Cut D0_MAX is fixed at 70 mm (chosen on val in run.py).
"""
import sys
import numpy as np
import pipeline as sp
import ag_data as D
import ag_synth as syn

BASE = dict(sp.CFG, D0_MAX=70.0)
VARIANTS = [
    ('full method',                         {}),
    ('no pointing cut (D0_MAX off)',        dict(D0_MAX=1e9)),
    ('no through-going veto',               dict(COLLINEAR=0.0)),
    ('no second view (x-y straightness)',   dict(TOL_XY=1e9)),
    ('no z-agreement (any 2 tracks)',       dict(NSIG=1e9)),
    ('NAIVE BASELINE (all rules off)',      dict(D0_MAX=1e9, COLLINEAR=0.0, NSIG=1e9, TOL_XY=1e9)),
]


def rate(events, cfg):
    return sum(sp.classify(e['pts'], cfg)[0] == 'annihilation' for e in events)


def loss_reasons(events, cfg):
    names = ['no track found', 'only 1 track found', '>=2 tracks, <2 usable (pointing/length/slope/fit/veto)',
             '>=2 usable tracks but they disagree on z']
    cnt = dict.fromkeys(names, 0); veto = 0
    for e in events:
        lab, z, tr = sp.classify(e['pts'], cfg)
        veto += any(t['thru'] for t in tr)
        if lab == 'annihilation':
            continue
        usable = sum(t['usable'] and not t['thru'] for t in tr)
        cnt[names[0] if len(tr) == 0 else names[1] if len(tr) == 1 else names[2] if usable < 2 else names[3]] += 1
    return cnt, veto


def main():
    n = int(sys.argv[sys.argv.index('--n') + 1]) if '--n' in sys.argv else 1000
    cal = D.load_events(sys.argv[1], 0, 300); ev = D.load_events(sys.argv[2], 0, n)
    g = D.estimate_geometry(cal); rng = np.random.default_rng(0)
    det = syn.Det(r_in=g['r_in'], r_out=g['r_out'], half_len=g['half_len'])
    cos = [syn.make_cosmic_event(rng, det, 1 + i % 3) for i in range(n)]
    print(f'\n{"variant":42s} {"signal eff":>22s} {"cosmic leakage (synthetic)":>30s}')
    for name, change in VARIANTS:
        if change is None:
            continue
        cfg = dict(BASE, **change); ks, kc = rate(ev, cfg), rate(cos, cfg)
        ls, hs = sp.wilson(ks, n); lc, hc = sp.wilson(kc, n)
        print(f'{name:42s} {ks / n:7.3f} [{ls:.3f},{hs:.3f}]   {kc / n:7.4f} [{lc:.4f},{hc:.4f}]')
    cnt, veto = loss_reasons(ev, BASE)
    lost = sum(cnt.values())
    print(f'\nSignal events lost with the full method: {lost}/{n} = {lost / n:.3f}')
    for k, v in cnt.items():
        print(f'   {v:5d} ({v / n:6.3f})  {k}')
    print(f'Signal events in which the through-going veto fired on a track: {veto}/{n}')


if __name__ == '__main__':
    main()