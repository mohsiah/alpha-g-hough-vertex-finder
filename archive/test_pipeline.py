"""
test_pipeline.py -- unit tests with KNOWN answers. Run:  python test_pipeline.py
Every test builds an input whose correct output you can derive by hand, so a failure means a
bug, not "a tuning issue". Run these before touching real data and after every code change.
"""
import numpy as np
from ag_pipeline import Params, find_tracks, fit_track, build_tracks, reconstruct
from ag_data import clean_points, _decode, _candidate_layouts, sanity_check
import ag_synth as syn

P = Params()
rng = np.random.default_rng(0)


def straight_pion(vertex, phi, cot, n=30, r0=110, r1=190):
    """Noiseless straight track leaving `vertex` radially-ish at azimuth phi, slope dz/dl=cot."""
    l = np.linspace(r0, r1, n)
    return np.column_stack([vertex[0] + l * np.cos(phi), vertex[1] + l * np.sin(phi), vertex[2] + l * cot])


def test_two_noiseless_tracks_recover_vertex_z():
    v = np.array([25.0, 0.0, 300.0])   # exactly on the assumed wall radius
    pts = np.vstack([straight_pion(v, 1.0, 0.8), straight_pion(v, 3.5, -1.1)])
    R = reconstruct(pts, P)
    assert R.label == 'annihilation', R.reason
    assert abs(R.vertex.z - 300.0) < 5.0, R.vertex.z


def test_single_track_is_not_a_vertex():
    pts = straight_pion(np.array([20.0, 0, 0]), 1.0, 0.5)
    assert reconstruct(pts, P).label == 'background'


def test_vertical_muon_rejected():
    z = np.linspace(-1000, 1000, 400)
    pts = np.column_stack([np.full_like(z, 150.0), np.zeros_like(z), z]) + rng.normal(0, 1, (400, 3))
    assert reconstruct(pts, P).label == 'background'


def test_tilted_muon_through_hollow_rejected():
    """The classic impostor: one straight line recorded as two half-tracks on opposite sides."""
    t = np.linspace(-1, 1, 200)
    line = np.column_stack([300 * t, 5 + 0 * t, 400 * t])          # passes ~5 mm from the axis
    r = np.hypot(line[:, 0], line[:, 1])
    pts = line[(r > 110) & (r < 190)] + rng.normal(0, 1.0, ((r > 110) & (r < 190)).sum() * 3).reshape(-1, 3)
    assert reconstruct(pts, P).label == 'background'


def test_tracks_that_miss_each_other_in_z_rejected():
    a = straight_pion(np.array([20.0, 0, 300.0]), 1.0, 0.8)
    b = straight_pion(np.array([-20.0, 0, -500.0]), 4.0, 0.8)     # same geometry, vertex 800 mm away
    assert reconstruct(np.vstack([a, b]), P).label == 'background'


def test_hit_order_does_not_matter():
    ev = syn.make_annihilation(np.random.default_rng(5))
    a = reconstruct(ev['pts'], P); b = reconstruct(ev['pts'][np.random.default_rng(1).permutation(len(ev['pts']))], P)
    assert a.label == b.label
    if a.vertex is not None:
        assert abs(a.vertex.z - b.vertex.z) < 1e-6


def test_global_z_shift_equivariance():
    ev = syn.make_annihilation(np.random.default_rng(6)); sh = ev['pts'] + np.array([0, 0, 123.0])
    a, b = reconstruct(ev['pts'], P), reconstruct(sh, P)
    assert a.label == b.label
    if a.vertex is not None:
        assert abs((b.vertex.z - a.vertex.z) - 123.0) < 1e-6


def test_azimuth_rotation_invariance():
    ev = syn.make_annihilation(np.random.default_rng(7)); p = ev['pts']; c, s = np.cos(1.234), np.sin(1.234)
    rot = np.column_stack([c * p[:, 0] - s * p[:, 1], s * p[:, 0] + c * p[:, 1], p[:, 2]])
    a, b = reconstruct(p, P), reconstruct(rot, P)
    assert a.label == b.label
    if a.vertex is not None:
        assert abs(a.vertex.z - b.vertex.z) < 3.0


def test_padding_removal_both_styles():
    real = rng.normal(100, 30, (50, 3))
    zero_pad = np.vstack([real, np.zeros((750, 3))])
    dup_pad = np.vstack([real, real[rng.integers(0, 50, 750)]])
    assert len(clean_points(zero_pad)) == 50 and len(clean_points(dup_pad)) == 50


def test_loader_layouts_and_sanity():
    evs = [syn.make_annihilation(np.random.default_rng(i)) for i in range(60)]
    P_ = 800; rows_b, rows_i = [], []
    for e in evs:
        p = np.zeros((P_, 3)); p[:len(e['pts'])] = e['pts']
        rows_b.append(np.concatenate([p[:, 0], p[:, 1], p[:, 2], [e['vz_true'], e['vz_true']]]))
        rows_i.append(np.concatenate([p.ravel(), [e['vz_true'], e['vz_true']]]))
    for lay, rows in [('blocks', rows_b), ('interleaved', rows_i)]:
        arr = np.array(rows)
        good = [dict(pts=clean_points(p), vz_true=v, vz_helix=h) for p, v, h in _decode(arr, lay)]
        assert sanity_check(good, verbose=False), lay
    wrong = [dict(pts=clean_points(p), vz_true=v, vz_helix=h) for p, v, h in _decode(np.array(rows_b), 'interleaved')]
    assert not sanity_check(wrong, verbose=False), 'sanity_check failed to catch a wrong layout'
    assert _candidate_layouts((10, 2402)) == ['blocks', 'interleaved']


if __name__ == '__main__':
    tests = [v for k, v in sorted(globals().items()) if k.startswith('test_')]
    fails = 0
    for t in tests:
        try:
            t(); print('PASS', t.__name__)
        except AssertionError as e:
            fails += 1; print('FAIL', t.__name__, e)
    print(f'\n{len(tests) - fails}/{len(tests)} passed')