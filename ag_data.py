import numpy as np

HELIX_FAILED = -9999999.0


def clean_points(pts):
    pts = np.asarray(pts, dtype=float).reshape(-1, 3)
    pts = pts[np.isfinite(pts).all(axis=1)]
    pts = pts[(pts[:, 0] != 0) | (pts[:, 1] != 0)]
    return np.unique(pts, axis=0)


def inspect_h5(path):
    import h5py
    with h5py.File(path, 'r') as f:
        print('file:', path)

        def show(name, obj):
            if isinstance(obj, h5py.Dataset):
                print(f'  {name:40s} shape={obj.shape} dtype={obj.dtype}')
                if obj.shape and obj.shape[0] > 0:
                    flat = np.asarray(obj[0]).ravel()
                    print(f'      first row, first 8 values: {flat[:8]}   (row has {flat.size} values)')
        f.visititems(show)


def _decode(arr, layout='pts5'):
    arr = np.asarray(arr, dtype=float)
    out = []
    for a in arr:
        pts = a[:, :3]
        vz, vh = a[0, 3], a[0, 4]
        if vh <= HELIX_FAILED / 2:
            vh = np.nan
        out.append((pts, vz, vh))
    return out


def _candidate_layouts(shape):
    return ['pts5'] if len(shape) == 3 and shape[2] == 5 else []


def sanity_check(events, z_half_len=1300.0, r_max=400.0, min_corr=0.9, verbose=True):
    mz, vz, ok = [], [], True
    for ev in events:
        p = ev['pts']
        if len(p) == 0:
            continue
        r = np.hypot(p[:, 0], p[:, 1])
        if r.max() > r_max or np.abs(p[:, 2]).max() > z_half_len:
            ok = False
        mz.append(p[:, 2].mean()); vz.append(ev['vz_true'])
    mz, vz = np.array(mz), np.array(vz)
    good = np.isfinite(vz)
    corr = np.corrcoef(mz[good], vz[good])[0, 1] if good.sum() > 10 else np.nan
    if verbose:
        print(f'  sanity: geometry plausible={ok}, corr(mean spacepoint z, true vertex z)={corr:.3f}')
    return bool(ok and (np.isnan(corr) or corr >= min_corr))


def load_events(path, start=0, count=1000, layout='auto', dataset='data', min_points=10):
    import h5py
    with h5py.File(path, 'r') as f:
        arr = f[dataset][start:start + count]
    if layout == 'auto' and not _candidate_layouts(arr.shape):
        raise RuntimeError(f'Unexpected array shape {arr.shape}; expected (events, 800, 5).')
    events = []
    for pts, vz, vh in _decode(arr):
        pts = clean_points(pts)
        if len(pts) >= min_points:
            events.append(dict(pts=pts, vz_true=float(vz), vz_helix=float(vh),
                               kind='annihilation', n_truth_tracks=-1, n_cosmics=0))
    print(f'{len(events)}/{len(arr)} events kept after padding removal '
          f'(mean {np.mean([len(e["pts"]) for e in events]):.0f} spacepoints/event)')
    if not sanity_check(events):
        raise RuntimeError('Sanity check failed: geometry or vertex-z correlation implausible.')
    return events


def estimate_geometry(events, lo=0.5, hi=99.5):
    r = np.concatenate([np.hypot(e['pts'][:, 0], e['pts'][:, 1]) for e in events])
    z = np.concatenate([e['pts'][:, 2] for e in events])
    g = dict(r_in=float(np.percentile(r, lo)), r_out=float(np.percentile(r, hi)),
             z_min=float(np.percentile(z, lo)), z_max=float(np.percentile(z, hi)))
    g['half_len'] = max(abs(g['z_min']), abs(g['z_max']))
    return g