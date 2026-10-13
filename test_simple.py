"""Unit tests for the FINAL algorithm (pipeline.py). Run: python test_simple.py"""
import numpy as np
import pipeline as sp
import ag_synth as syn
from ag_data import clean_points

rng = np.random.default_rng(0)
CFG = dict(sp.CFG, D0_MAX=70.0)
label = lambda pts, cfg=CFG: sp.classify(pts, cfg)[0]


def pion(vertex, phi, cot, n=30):
    l = np.linspace(110, 180, n)
    return np.column_stack([vertex[0] + l * np.cos(phi), vertex[1] + l * np.sin(phi), vertex[2] + l * cot])


def test_two_noiseless_tracks_recover_vertex_z():
    v = np.array([25.0, 0, 300.0])
    lab, z, _ = sp.classify(np.vstack([pion(v, 1.0, 0.8), pion(v, 3.5, -1.1)]), CFG)
    assert lab == 'annihilation' and abs(z - 300) < 25, (lab, z)

def test_single_track_is_background():
    assert label(pion(np.array([25.0, 0, 0]), 1.0, 0.5)) == 'background'

def test_vertical_muon_rejected():
    z = np.linspace(-1000, 1000, 400)
    pts = np.column_stack([np.full_like(z, 150.0), np.zeros_like(z), z]) + rng.normal(0, 1, (400, 3))
    assert label(pts) == 'background'

def test_tilted_muon_through_hollow_rejected():
    t = np.linspace(-1, 1, 400); line = np.column_stack([300 * t, 5 + 0 * t, 400 * t]); r = np.hypot(line[:, 0], line[:, 1])
    m = (r > 109) & (r < 181); pts = line[m] + rng.normal(0, 1.0, (m.sum(), 3))
    assert label(pts) == 'background'

def test_veto_is_what_rejects_it():
    """Ablation sanity: with the through-going veto switched off the same muon must pass."""
    t = np.linspace(-1, 1, 400); line = np.column_stack([300 * t, 5 + 0 * t, 400 * t]); r = np.hypot(line[:, 0], line[:, 1])
    m = (r > 109) & (r < 181); pts = line[m] + rng.normal(0, 1.0, (m.sum(), 3))
    assert label(pts, dict(CFG, COLLINEAR=0.0)) == 'annihilation'

def test_tracks_far_apart_in_z_rejected():
    a = pion(np.array([25.0, 0, 300.0]), 1.0, 0.8); b = pion(np.array([-25.0, 0, -500.0]), 4.0, 0.8)
    assert label(np.vstack([a, b])) == 'background'

def test_hit_order_irrelevant():
    ev = syn.make_annihilation(np.random.default_rng(5))
    a = sp.classify(ev['pts'], CFG); b = sp.classify(ev['pts'][np.random.default_rng(1).permutation(len(ev['pts']))], CFG)
    assert a[0] == b[0] and (a[1] is None or abs(a[1] - b[1]) < 1e-6)

def test_z_shift_equivariance():
    ev = syn.make_annihilation(np.random.default_rng(6)); a = sp.classify(ev['pts'], CFG); b = sp.classify(ev['pts'] + [0, 0, 123.0], CFG)
    assert a[0] == b[0] and (a[1] is None or abs(b[1] - a[1] - 123.0) < 1e-6)

def test_padding_removal():
    real = rng.normal(100, 30, (50, 3)); real[:, :2] += 5
    padded = np.vstack([real, np.column_stack([np.zeros(300), np.zeros(300), rng.choice([-26, 30, 0], 300)])])
    assert len(clean_points(padded)) == 50

if __name__ == '__main__':
    ts = [v for k, v in sorted(globals().items()) if k.startswith('test_')]; bad = 0
    for t in ts:
        try: t(); print('PASS', t.__name__)
        except AssertionError as e: bad += 1; print('FAIL', t.__name__, e)
    print(f'{len(ts) - bad}/{len(ts)} passed')