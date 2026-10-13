"""
animate_real.py -- your Step-2 animation, driven by a REAL simulated ALPHA-g event.
  python animate_real.py test.h5 --event 5                 # play on screen
  python animate_real.py test.h5 --event 5 --save e5.gif   # save a GIF for slides
  python animate_real.py test.h5 --event 5 --cosmic        # add a SYNTHETIC cosmic muon (shown in red)
  python animate_real.py --synthetic --save demo.gif       # no data file needed
"""
import sys
import numpy as np
import matplotlib
if '--save' in sys.argv:
    matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.animation as animation
try:
    import pipeline as sp
except ImportError:
    import pipeline as sp
try:
    from hough_detector import refit_line_tls
except ImportError:
    from hough_detector import refit_line_tls

D0_MAX = 70.0                                   # the cut chosen on the validation file


def analyse(pts):
    """Run the pipeline and split tracks into 'vertex tracks' and 'rejected tracks'."""
    cfg = dict(sp.CFG, D0_MAX=D0_MAX)
    label, zv, tracks = sp.classify(pts, cfg)
    vtx, rejected = [], []
    for t in tracks:
        ok = label == 'annihilation' and t['usable'] and not t['thru'] and abs(t['z0'] - zv) < 3 * t['sig']
        (vtx if ok else rejected).append(t)
    vertex = None
    if vtx:                                      # xy of the vertex: mean closest-approach point of its tracks
        feet = []
        for t in vtx:
            p = pts[t['hits']]
            th, rho = refit_line_tls(p[:, 0], p[:, 1])
            feet.append(rho * np.array([np.cos(th), np.sin(th)]))
        vertex = np.array([*np.mean(feet, axis=0), zv])
    return label, vertex, vtx, rejected


def make_animation(ev, r_in, r_out, half_len, title_extra='', zoom=True):
    pts = ev['pts']
    label, vertex, vtx, rejected = analyse(pts)
    z_lo, z_hi = -half_len, half_len
    if zoom and vertex is not None and not rejected:      # zoom on the event unless a long cosmic must be shown
        z_lo, z_hi = vertex[2] - 300, vertex[2] + 300
    used = np.zeros(len(pts), bool)
    groups = []                                  # (hit indices ordered outward, colour, legend text)
    cols = plt.cm.Blues(np.linspace(0.55, 0.95, max(len(vtx), 1)))
    for c, t in zip(cols, vtx):
        h = t['hits']; d = np.linalg.norm(pts[h] - vertex, axis=1)
        groups.append((h[np.argsort(d)], c, 'annihilation track')); used[h] = True
    for t in rejected:
        h = t['hits']; groups.append((h[np.argsort(-pts[h, 2])], 'crimson', 'rejected as background')); used[h] = True
    other = np.flatnonzero(~used)

    FLASH, GROW, HOLD = 6, 36, 12
    fig = plt.figure(figsize=(8, 8)); ax = fig.add_subplot(111, projection='3d')
    th = np.linspace(0, 2 * np.pi, 48); zz = np.linspace(z_lo, z_hi, 2)
    T, Z = np.meshgrid(th, zz)

    def update(f):
        ax.clear()
        for r in (r_in, r_out):
            ax.plot_wireframe(r * np.cos(T), r * np.sin(T), Z, color='gray', alpha=0.3, linewidth=0.5)
        ax.set_xlim(-r_out, r_out); ax.set_ylim(-r_out, r_out); ax.set_zlim(z_lo, z_hi)
        ax.set_xlabel('x (mm)'); ax.set_ylabel('y (mm)'); ax.set_zlabel('z (mm)')
        if vertex is not None and f < FLASH:
            ax.scatter(*vertex, color='gold', marker='*', s=200 + 300 * np.sin(f / FLASH * np.pi),
                       edgecolor='red', linewidth=1.5, label='annihilation')
            ax.set_title('Antihydrogen annihilates on the trap wall...')
        elif f < FLASH + GROW + HOLD:
            frac = min(1.0, (f - FLASH + 1) / GROW)
            if vertex is not None:
                ax.scatter(*vertex, color='gold', marker='*', s=150, edgecolor='red', linewidth=1,
                           label=f'reconstructed vertex z = {vertex[2]:.0f} mm')
            seen = set()
            for h, c, name in groups:
                k = max(1, int(np.ceil(frac * len(h))))
                ax.scatter(*pts[h[:k]].T, color=c, s=10, label=None if name in seen else name); seen.add(name)
            if frac >= 1.0 and len(other):
                ax.scatter(*pts[other].T, color='gray', s=3, alpha=0.5, label='unassigned hits')
            ax.set_title('Charged pions propagate outward through the gas' + title_extra)
        ax.legend(loc='upper right', fontsize=8)
        return ax,

    info = f"verdict: {label}"
    if vertex is not None and np.isfinite(ev['vz_true']):
        info += f" | true z {ev['vz_true']:.0f} mm, ours {vertex[2]:.0f} mm"
        if np.isfinite(ev['vz_helix']):
            info += f", Helix Fit {ev['vz_helix']:.0f} mm"
    fig.suptitle(info, fontsize=10)
    return fig, animation.FuncAnimation(fig, update, frames=FLASH + GROW + HOLD, interval=110, repeat=True)


def main():
    a = sys.argv
    get = lambda k, d: type(d)(a[a.index(k) + 1]) if k in a else d
    event_no = get('--event', 0)
    rng = np.random.default_rng(event_no)
    if '--synthetic' in a:
        import ag_synth as syn
        det = syn.Det(); ev = syn.make_annihilation(rng, det)
        r_in, r_out, half = det.r_in, det.r_out, det.half_len
    else:
        import ag_data as D, ag_synth as syn
        ev = D.load_events(a[1], event_no, 20)[0]
        g = D.estimate_geometry([ev]); r_in, r_out, half = 109.0, 181.0, 1134.0
        det = syn.Det(r_in=r_in, r_out=r_out, half_len=half)
    extra = ''
    if '--cosmic' in a:
        ev = syn.overlay_cosmics(rng, ev, det, 1); extra = '  (red = SYNTHETIC cosmic muon)'
    fig, ani = make_animation(ev, r_in, r_out, half, extra)
    if '--save' in a:
        ani.save(a[a.index('--save') + 1], writer=animation.PillowWriter(fps=9)); print('saved')
    else:
        plt.show()


if __name__ == '__main__':
    main()