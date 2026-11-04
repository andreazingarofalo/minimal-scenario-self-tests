"""NPA figure of Appendix C (figures/fam_npa_levels.pdf), six values of theta.

Compares the exact section A_theta = P_theta with the region certified by the almost-quantum level 1+A+B+AB.
The boundary of that region is found by bisection along 180 rays from the centroid of P_theta (181 angles,
the last one closing the curve), with MOSEK via ncpol2sdpa. The boundaries are cached in
data/npa_regions_cache.npz and only missing entries are computed, so with the cache in place the script only
draws. Rebuilding the cache takes one or two minutes on eight cores. Also writes a PNG copy.
"""
import os

import matplotlib
import numpy as np

matplotlib.use('Agg')
import matplotlib.pyplot as plt
from joblib import Parallel, delayed

from common import beta_vec
from lines_and_thetabar import THETA_BAR, polygon
from npa import npa_dual

plt.rcParams.update({'font.family': 'serif', 'mathtext.fontset': 'cm', 'font.size': 9, 'axes.linewidth': 0.6,
                     'xtick.major.width': 0.6, 'ytick.major.width': 0.6, 'axes.edgecolor': '#52514e',
                     'xtick.color': '#52514e', 'ytick.color': '#52514e', 'axes.labelcolor': '#0b0b0b'})
FILL = '#cde2fb'
EDGE = '#184f95'
GRID = '#e4e3df'
ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..')
CACHE = os.path.join(ROOT, 'data', 'npa_regions_cache.npz')
OUT = os.path.join(ROOT, 'figures', 'fam_npa_levels')

PANELS = [(np.pi / 8 + np.pi / 64, r'$\theta=\pi/8+\pi/64$'),
          (np.pi / 8 + 3 * np.pi / 64, r'$\theta=\pi/8+3\pi/64$'),
          (np.pi / 8 + 5 * np.pi / 64, r'$\theta=\pi/8+5\pi/64$'),
          (THETA_BAR, r'$\theta=\bar\theta$'),
          (np.pi / 8 + 7 * np.pi / 64, r'$\theta=\pi/8+7\pi/64$'),
          (np.pi / 4, r'$\theta=\pi/4$')]
# (level, colour, line style, legend label) of the NPA regions drawn
LEVELS = [('AB', '#52514e', '--', r'almost-quantum level $1{+}A{+}B{+}AB$')]
PHIS = np.linspace(0, 2 * np.pi, 181)


def closed(P):
    """Points of P sorted by angle around their mean, with the first point repeated at the end."""
    c = P.mean(0)
    P = P[np.argsort(np.arctan2(P[:, 1] - c[1], P[:, 0] - c[0]))]
    return np.vstack([P, P[:1]])


def boundary_point(theta, level, phi, cen):
    """Point of the boundary of {NPA bound <= 1} on the ray from cen at angle phi (bisection in [0, 0.6])."""
    d = np.array([np.cos(phi), np.sin(phi)])
    lo, hi = 0.0, 0.6
    for _ in range(26):
        m = (lo + hi) / 2
        if npa_dual(beta_vec(theta, *(cen + m * d)), level) <= 1 + 1e-7:
            lo = m
        else:
            hi = m
    return cen + lo * d


def load_or_compute_boundaries():
    """Region boundaries per panel and level, from the cache, computing and saving the missing ones."""
    os.makedirs(os.path.dirname(CACHE), exist_ok=True)
    cache = dict(np.load(CACHE)) if os.path.exists(CACHE) else {}
    for k, (theta, _) in enumerate(PANELS):
        cen = polygon(theta).mean(0)
        for level, *_ in LEVELS:
            key = f'p{k}_{level}'
            if key not in cache:
                cache[key] = np.array(Parallel(n_jobs=8)(delayed(boundary_point)(theta, level, phi, cen)
                                                         for phi in PHIS))
                np.savez(CACHE, **cache)
                print('done', key, flush=True)
    return cache


def main():
    cache = load_or_compute_boundaries()

    # same square window and scale in every panel, centred on the polygon
    half = 1.1 * max(max(np.ptp(polygon(theta)[:, 0]), np.ptp(polygon(theta)[:, 1]))
                     for theta, _ in PANELS) / 2
    fig, axs = plt.subplots(2, 3, figsize=(6.8, 5.0))
    for k, (ax, (theta, title)) in enumerate(zip(axs.flat, PANELS)):
        C = closed(polygon(theta))
        ax.fill(C[:, 0], C[:, 1], color=FILL, lw=0, label=r'$\mathcal{A}_\theta\equiv\mathcal{P}_\theta$')
        ax.plot(C[:, 0], C[:, 1], '-', color=EDGE, lw=1.0)
        for level, colour, style, label in LEVELS:
            pts = cache[f'p{k}_{level}']
            ax.plot(pts[:, 0], pts[:, 1], style, color=colour, lw=1.1, label=label)
        cx = 0.5 * (C[:, 0].min() + C[:, 0].max())
        ax.set_xlim(cx - half, cx + half)
        ax.set_ylim(-half, half)
        ax.set_aspect('equal')
        ax.grid(color=GRID, lw=0.5)
        ax.set_axisbelow(True)
        ax.set_title(title, fontsize=9)
        ax.tick_params(labelsize=7)
    for ax in axs[1]:
        ax.set_xlabel(r'$r_0$')
    for ax in axs[:, 0]:
        ax.set_ylabel(r'$r_1$')
    handles, labels = axs[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, fontsize=8, frameon=False, loc='lower center', ncol=2,
               bbox_to_anchor=(0.5, -0.02))
    fig.tight_layout(rect=(0, 0.05, 1, 1))
    fig.savefig(OUT + '.pdf', bbox_inches='tight')
    fig.savefig(OUT + '.png', dpi=150, bbox_inches='tight')
    print('figure written')


if __name__ == '__main__':
    main()
