import sys
import numpy as np
import h5py
import matplotlib.pyplot as plt


def rows(f, table, field, start, count):
    return f[f'{table}/{field}'][start:start + count, 0]


def load_event(f, i):
    s, n = f['hit_table/event_id.seq_cnt'][i]
    hit = {k: rows(f, 'hit_table', k, s, n) for k in ('hit_id', 'local_plane', 'local_wire', 'local_time', 'integral')}
    es, en = f['edep_table/event_id.seq_cnt'][i]
    edep_hit = rows(f, 'edep_table', 'hit_id', es, en)
    edep_frac = rows(f, 'edep_table', 'energy_fraction', es, en)
    ps, pn = f['particle_table/event_id.seq_cnt'][i]
    parent = rows(f, 'particle_table', 'parent_id', ps, pn)
    start_t = rows(f, 'particle_table', 'start_wire_time', ps, pn)
    return hit, edep_hit, edep_frac, parent, start_t


def neutrino_fraction(hit_id, edep_hit, edep_frac):
    """Per hit: summed fraction of its energy that came from the simulated neutrino interaction."""
    lookup = {h: k for k, h in enumerate(hit_id)}
    frac = np.zeros(len(hit_id))
    for h, fr in zip(edep_hit, edep_frac):
        k = lookup.get(h)
        if k is not None:
            frac[k] += fr
    return frac


def main():
    path = sys.argv[1]
    events = [int(a) for a in sys.argv[2:]] or [0]
    with h5py.File(path, 'r') as f:
        for i in events:
            hit, eh, ef, parent, start_t = load_event(f, i)
            wire_vtx = f['event_table/nu_vtx_wire_pos'][i]
            t_doc = float(f['event_table/nu_vtx_wire_time'][i, 0])
            t_prim = float(np.mean(start_t[parent == 0])) if np.any(parent == 0) else np.nan
            frac = neutrino_fraction(hit['hit_id'], eh, ef)
            print(f'event {i}: hits per plane {np.bincount(hit["local_plane"], minlength=3)}, '
                  f'hits mostly neutrino {int(np.sum(frac >= 0.5))}, '
                  f'truth vertex plane-2 wire {wire_vtx[2]}, time (event_table) {t_doc:.0f}, time (primary particle) {t_prim:.0f}')
            m = hit['local_plane'] == 2
            w, t = hit['local_wire'][m], hit['local_time'][m]
            nu = frac[m] >= 0.5
            plt.figure(figsize=(12, 6))
            plt.scatter(w[~nu], t[~nu], s=2, c='lightgray', label='other (mostly real cosmic)')
            plt.scatter(w[nu], t[nu], s=4, c='red', label='neutrino interaction')
            plt.plot(wire_vtx[2], t_doc, 'bx', ms=14, mew=3, label='vertex time from event_table')
            plt.plot(wire_vtx[2], t_prim, 'g+', ms=16, mew=3, label='vertex time from primary particle')
            plt.xlabel('wire number (plane 2)'); plt.ylabel('time tick'); plt.title(f'event {i}'); plt.legend()
    plt.show()


if __name__ == '__main__':
    main()