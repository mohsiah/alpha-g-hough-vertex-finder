"""Hough-transform line finder (2-D) used by pipeline.py. Only numpy is required."""
import numpy as np


def hough_transform_2d(u, v, n_theta=180, n_rho=200, rho_range=None):
    """Standard Hough accumulator for lines in a 2D (u, v) point set, using
    the normal form rho = u*cos(theta) + v*sin(theta). theta in [0, pi)."""
    thetas = np.linspace(0, np.pi, n_theta, endpoint=False)
    cos_t, sin_t = np.cos(thetas), np.sin(thetas)

    # rho_matrix[i, j] = rho of point i at angle theta_j
    rho_matrix = np.outer(u, cos_t) + np.outer(v, sin_t)

    if rho_range is None:
        max_r = np.sqrt(np.max(u**2 + v**2)) * 1.05 + 1e-6
        rho_range = (-max_r, max_r)
    rho_vals = np.linspace(rho_range[0], rho_range[1], n_rho)
    bin_width = (rho_range[1] - rho_range[0]) / n_rho

    rho_idx = np.clip(((rho_matrix - rho_range[0]) / bin_width).astype(int), 0, n_rho - 1)
    theta_idx = np.broadcast_to(np.arange(n_theta), rho_idx.shape)

    accumulator = np.zeros((n_rho, n_theta), dtype=int)
    np.add.at(accumulator, (rho_idx.ravel(), theta_idx.ravel()), 1)
    return accumulator, thetas, rho_vals


def find_peaks_2d(accumulator, thetas, rho_vals, n_peaks=12, min_votes=4,
                   suppress_bins=(6, 6)):
    """Greedy peak-picking with non-max suppression: take the current max,
    record it, zero out a window around it so nearby bins from the same
    physical line don't get picked again, repeat."""
    work = accumulator.copy()
    peaks = []
    for _ in range(n_peaks):
        r_i, t_i = np.unravel_index(np.argmax(work), work.shape)
        votes = work[r_i, t_i]
        if votes < min_votes:
            break
        peaks.append({'theta': thetas[t_i], 'rho': rho_vals[r_i], 'votes': int(votes)})
        r_lo, r_hi = max(0, r_i - suppress_bins[0]), min(work.shape[0], r_i + suppress_bins[0] + 1)
        t_lo, t_hi = max(0, t_i - suppress_bins[1]), min(work.shape[1], t_i + suppress_bins[1] + 1)
        work[r_lo:r_hi, t_lo:t_hi] = 0
    return peaks


def line_inlier_mask(u, v, theta, rho, tol):
    """Perpendicular distance from every (u, v) point to the line, thresholded."""
    dist = np.abs(u * np.cos(theta) + v * np.sin(theta) - rho)
    return dist < tol


def refit_line_tls(u, v):
    """Total-least-squares refit of a line through points already assigned to
    a Hough peak, via eigendecomposition of the 2x2 scatter matrix. Cleans up
    the coarse Hough-bin estimate into a precise (theta, rho)."""
    pts = np.stack([u, v], axis=1)
    centroid = pts.mean(axis=0)
    centered = pts - centroid
    cov = centered.T @ centered
    eigvals, eigvecs = np.linalg.eigh(cov)   # ascending eigenvalues
    normal = eigvecs[:, 0]                   # smallest-variance direction = normal to line
    theta = np.arctan2(normal[1], normal[0]) % np.pi
    n = np.array([np.cos(theta), np.sin(theta)])
    rho = centroid @ n
    return theta, rho


def find_lines_2d(u, v, tol=0.6, n_theta=180, n_rho=200, n_peaks=12, min_votes=4,
                   suppress_bins=(6, 6)):
    """Full 2D pipeline: vote -> find peaks -> collect inliers -> refit -> re-collect inliers."""
    accumulator, thetas, rho_vals = hough_transform_2d(u, v, n_theta, n_rho)
    peaks = find_peaks_2d(accumulator, thetas, rho_vals, n_peaks=n_peaks, min_votes=min_votes,
                           suppress_bins=suppress_bins)

    lines = []
    for peak in peaks:
        mask = line_inlier_mask(u, v, peak['theta'], peak['rho'], tol)
        if mask.sum() < min_votes:
            continue
        theta_r, rho_r = refit_line_tls(u[mask], v[mask])
        mask_refined = line_inlier_mask(u, v, theta_r, rho_r, tol)
        lines.append({'theta': theta_r, 'rho': rho_r, 'mask': mask_refined,
                       'votes': int(mask_refined.sum())})
    return lines, accumulator, thetas, rho_vals


# ---------------------------------------------------------------------------
# 2. Combine the x-z and y-z views into real 3D track candidates
# ---------------------------------------------------------------------------
