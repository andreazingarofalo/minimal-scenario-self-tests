"""Figure of Appendix C with the local polygon P_theta and its vertex labels (figures/fam_vertices.pdf).

Three panels: a hexagon below theta_bar, theta_bar with the two triple vertices circled, and the regular
octagon at pi/4. Each edge is labelled by the deterministic point whose line carries it, and the dashed lines
are the lines of the eight CHSH-saturating deterministic points. Also writes a PNG copy.
"""
import os

import matplotlib
import numpy as np

matplotlib.use('Agg')
import matplotlib.pyplot as plt

from lines_and_thetabar import PAIRS_6, PAIRS_8, THETA_BAR, lines, vert

plt.rcParams.update({'font.family': 'serif', 'mathtext.fontset': 'cm', 'font.size': 8, 'axes.linewidth': 0.6,
                     'xtick.major.width': 0.6, 'ytick.major.width': 0.6, 'axes.edgecolor': '#52514e',
                     'xtick.color': '#52514e', 'ytick.color': '#52514e', 'axes.labelcolor': '#0b0b0b'})
FILL = '#cde2fb'
EDGE = '#184f95'
LINE = '#b9b8b3'
ORANGE = '#eb6834'
TEXT2 = '#52514e'
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'figures', 'fam_vertices')

PANELS = [(np.pi / 8 + 3 * np.pi / 64, r'$\theta=\pi/8+3\pi/64$ (hexagon)'),
          (THETA_BAR, r'$\theta=\bar\theta$'),
          (np.pi / 4, r'$\theta=\pi/4$ (octagon)')]
# Vertex labels of the paper, in the order of PAIRS_6 and PAIRS_8 in lines_and_thetabar.py.
LABELS_6 = [r'$V_1$', r'$V_2$', r'$V_3$', r'$V_4$', r"$V_3'$", r"$V_2'$"]
LABELS_8 = [r'$V_1$', r'$V_5$', r'$V_6$', r'$V_3$', r'$V_4$', r"$V_3'$", r"$V_6'$", r"$V_5'$"]
TRIPLE_VERTICES = [(('I', 1, 1), ('II', -1, 1)), (('I', 1, -1), ('II', 1, 1))]
HALF = 0.40    # half width of each panel


def name(key):
    """Label of a deterministic point, e.g. ('I', 1, -1) -> I(+,-)."""
    kind, a, b = key
    return r'$\mathrm{%s}(%s,%s)$' % (kind, '+' if a > 0 else '-', '+' if b > 0 else '-')


def draw_line(ax, g, h, lo, hi):
    """Line g.r = h clipped to the box [lo, hi]."""
    if abs(g[1]) > abs(g[0]):
        x = np.array([lo[0], hi[0]])
        y = (h - g[0] * x) / g[1]
    else:
        y = np.array([lo[1], hi[1]])
        x = (h - g[1] * y) / g[0]
    ax.plot(x, y, '--', color=LINE, lw=0.6, zorder=1)


def draw_panel(ax, theta, title):
    L = lines(theta)
    pairs = PAIRS_6 if theta <= THETA_BAR else PAIRS_8
    labels = LABELS_6 if len(pairs) == 6 else LABELS_8
    n = len(pairs)
    V = np.array([vert(L[p], L[q]) for p, q in pairs])
    cen = V.mean(0)
    cx = 0.5 * (V[:, 0].min() + V[:, 0].max())
    lo = np.array([cx - HALF, -HALF])
    hi = np.array([cx + HALF, HALF])

    for g, h in L.values():
        draw_line(ax, g, h, lo, hi)
    C = np.vstack([V, V[:1]])
    ax.fill(C[:, 0], C[:, 1], color=FILL, lw=0, zorder=2)
    ax.plot(C[:, 0], C[:, 1], '-', color=EDGE, lw=1.1, zorder=3)
    ax.plot(V[:, 0], V[:, 1], 'o', color=EDGE, ms=2.4, zorder=4)

    # vertex labels, pushed away from the centre
    for v, label in zip(V, labels):
        d = (v - cen) / np.linalg.norm(v - cen)
        ax.text(*(v + 0.065 * d), label, ha='center', va='center', fontsize=8, zorder=5)

    # Edge k joins vertices k and k+1, which share one line. Its label goes just inside the polygon.
    for k in range(n):
        w0, w1 = V[k], V[(k + 1) % n]
        shared = (set(pairs[k]) & set(pairs[(k + 1) % n])).pop()
        m = 0.5 * (w0 + w1)
        e = w1 - w0
        normal = np.array([e[1], -e[0]]) / np.linalg.norm(e)
        if normal @ (cen - m) < 0:
            normal = -normal
        ax.text(*(m + 0.055 * normal), name(shared), ha='center', va='center', fontsize=5.8, color=TEXT2,
                zorder=5)

    if abs(theta - THETA_BAR) < 1e-12:
        for p, q in TRIPLE_VERTICES:
            ax.plot(*vert(L[p], L[q]), 'o', ms=7, mfc='none', mec=ORANGE, mew=0.9, zorder=6)
    if abs(theta - np.pi / 4) < 1e-12:
        ax.plot(0, 0, '+', ms=5, mew=0.9, color=EDGE, zorder=5)
        ax.annotate(r'$\vec{\beta}_{\rm CHSH}$', (0, 0), xytext=(0, -9), textcoords='offset points',
                    ha='center', fontsize=7, color=EDGE)

    ax.set_xlim(lo[0], hi[0])
    ax.set_ylim(lo[1], hi[1])
    ax.set_aspect('equal')
    ax.set_title(title, fontsize=8.5)
    ax.set_xlabel(r'$r_0$')
    ax.tick_params(labelsize=7)


def main():
    fig, axs = plt.subplots(1, 3, figsize=(6.8, 2.75))
    for ax, (theta, title) in zip(axs, PANELS):
        draw_panel(ax, theta, title)
    axs[0].set_ylabel(r'$r_1$')
    fig.tight_layout()
    fig.savefig(OUT + '.pdf', bbox_inches='tight')
    fig.savefig(OUT + '.png', dpi=200, bbox_inches='tight')
    print('saved', OUT)


if __name__ == '__main__':
    main()
