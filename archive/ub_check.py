import sys
import numpy as np
import h5py
import matplotlib.pyplot as plt

f = h5py.File(sys.argv[1], 'r')
PROBE = {'hit_table': 'hit_id', 'edep_table': 'energy', 'particle_table': 'momentum'}

print('A) what does event_id.seq_cnt really contain?')
for t in PROBE:
    sc = f[f'{t}/event_id.seq_cnt'][:]
    print(f'   {t}: first rows {sc[:5].tolist()} | sum of column 1 = {int(sc[:, 1].sum())} | rows in table = {f[f"{t}/{PROBE[t]}"].shape[0]}')


def span(table, i):
    cnt = f[f'{table}/event_id.seq_cnt'][:, 1]
    return int(cnt[:i].sum()), int(cnt[i])


def col(table, name, i):
    s, n = span(table, i)
    return f[f'{table}/{name}'][s:s + n, 0]


print('\nB) hits per plane and hit-id uniqueness, first 6 events')
for i in range(6):
    p, hid = col('hit_table', 'local_plane', i), col('hit_table', 'hit_id', i)
    print(f'   event {i}: planes {np.bincount(p, minlength=3)}  hit_id range {hid.min()}..{hid.max()}, {len(np.unique(hid))} unique of {len(hid)}')

print('\nC) do different events share identical hits?')
def hitset(i):
    p, w, t = (col('hit_table', k, i) for k in ('local_plane', 'local_wire', 'local_time'))
    return set(zip(p.tolist(), w.tolist(), np.round(t, 1).tolist()))
H = [hitset(i) for i in range(4)]
for a in range(4):
    for b in range(a + 1, 4):
        print(f'   events {a},{b}: {len(H[a] & H[b])} shared of {len(H[a])} and {len(H[b])}')

print('\nD) plots: plane-2 hits (gray) with TRUE particle paths (red); bold = primary particles')
events = [int(a) for a in sys.argv[2:]] or [0, 1, 2]
for i in events:
    m = col('hit_table', 'local_plane', i) == 2
    w, t = col('hit_table', 'local_wire', i)[m], col('hit_table', 'local_time', i)[m]
    ps, pn = span('particle_table', i)
    sw = f['particle_table/start_wire_pos'][ps:ps + pn]; ew = f['particle_table/end_wire_pos'][ps:ps + pn]
    st = f['particle_table/start_wire_time'][ps:ps + pn, 0]; et = f['particle_table/end_wire_time'][ps:ps + pn, 0]
    prim = f['particle_table/parent_id'][ps:ps + pn, 0] == 0
    plt.figure(figsize=(12, 6)); plt.scatter(w, t, s=2, c='lightgray')
    for k in range(pn):
        plt.plot([sw[k, 2], ew[k, 2]], [st[k], et[k]], 'r-', lw=2.5 if prim[k] else 0.6, alpha=1.0 if prim[k] else 0.3)
    plt.plot(f['event_table/nu_vtx_wire_pos'][i][2], float(f['event_table/nu_vtx_wire_time'][i, 0]), 'bx', ms=14, mew=3, label='event-table vertex (not used)')
    plt.scatter([], [], s=6, c='lightgray', label='hits (mostly real cosmic)')
    plt.plot([], [], 'r-', label='true simulated particle paths')
    plt.ylim(t.min() - 300, t.max() + 300); plt.xlim(-100, 3600)
    plt.xlabel('wire (plane 2)'); plt.ylabel('time tick'); plt.title(f'event {i}'); plt.legend(loc='upper right')
plt.show()