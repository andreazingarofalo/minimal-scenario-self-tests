"""Figure of the orthogonal faces in the main text (figures/fam_faces.pdf), one column wide.

Draws the local polygon P_theta in the (r0, r1) plane for five values of theta in (pi/8, pi/4], the paths of
its vertices as theta varies (grey), the segment it collapses onto as theta -> pi/8, and circles the two
triple vertices at theta_bar. Also writes a PNG copy.
"""
import os

import matplotlib
import numpy as np

matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

from lines_and_thetabar import PAIRS_6, PAIRS_8, THETA_BAR, lines, polygon, vert

plt.rcParams.update({'font.family': 'serif', 'mathtext.fontset': 'cm', 'font.size': 8, 'axes.linewidth': 0.6,
                     'xtick.major.width': 0.6, 'ytick.major.width': 0.6, 'axes.edgecolor': '#52514e',
                     'xtick.color': '#52514e', 'ytick.color': '#52514e', 'axes.labelcolor': '#0b0b0b'})
BLUES = ['#a9cbf3', '#5f9de6', '#2462b5', '#0b1f3a']
ORANGE = '#eb6834'
GREY = '#b9b8b3'
GRID = '#e4e3df'
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'figures', 'fam_faces')

# (theta, colour, legend label) of the five polygons
POLYGONS = [(np.pi / 8 + np.pi / 64, BLUES[0], r'$\pi/8+\pi/64$'),
            (np.pi / 8 + 3 * np.pi / 64, BLUES[1], r'$\pi/8+3\pi/64$'),
            (THETA_BAR, ORANGE, r'$\bar\theta$'),
            (np.pi / 8 + 6 * np.pi / 64, BLUES[2], r'$\pi/8+6\pi/64$'),
            (np.pi / 4, BLUES[3], r'$\pi/4$')]
# The two vertices at theta_bar where three lines meet.
TRIPLE_VERTICES = [(('I', 1, 1), ('II', -1, 1)), (('I', 1, -1), ('II', 1, 1))]


def line_width(colour):
    return 1.4 if colour == ORANGE else 1.0


def vertex(theta, p, q):
    """Intersection of the lines of the deterministic points p and q."""
    L = lines(theta)
    return vert(L[p], L[q])


def ordered_vertices(theta):
    """Vertices of P_theta in counterclockwise order."""
    V = polygon(theta)
    c = V.mean(0)
    return V[np.argsort(np.arctan2(V[:, 1] - c[1], V[:, 0] - c[0]))]


def draw_polygon(ax, theta, colour):
    V = ordered_vertices(theta)
    C = np.vstack([V, V[:1]])
    ax.plot(C[:, 0], C[:, 1], color=colour, lw=line_width(colour), solid_joinstyle='round', zorder=3)
    ax.plot(V[:, 0], V[:, 1], 'o', ms=2.4, color=colour, zorder=4)


def main():
    fig, ax = plt.subplots(figsize=(3.4, 2.75))

    # vertex paths over (pi/8, pi/4], hexagon up to theta_bar and octagon after
    for thetas, pairs in [(np.linspace(np.pi / 8 + 2e-4, THETA_BAR, 400), PAIRS_6),
                          (np.linspace(THETA_BAR, np.pi / 4, 200), PAIRS_8)]:
        for p, q in pairs:
            path = np.array([vertex(theta, p, q) for theta in thetas])
            ax.plot(path[:, 0], path[:, 1], color=GREY, lw=0.5, zorder=1)

    # limit theta -> pi/8: the segment r0 = 1/sqrt2, |r1| <= sqrt2 - 1
    s = np.sqrt(2) - 1
    ax.plot([1 / np.sqrt(2)] * 2, [-s, s], color=GREY, lw=1.6, solid_capstyle='round', zorder=1)

    for theta, colour, _ in POLYGONS:
        draw_polygon(ax, theta, colour)

    for p, q in TRIPLE_VERTICES:
        ax.plot(*vertex(THETA_BAR, p, q), 'o', ms=7, mfc='none', mec=ORANGE, mew=0.9, zorder=5)

    # CHSH sits at the centre of the regular octagon
    ax.plot(0, 0, '+', ms=5, mew=0.9, color=BLUES[3], zorder=5)
    ax.annotate(r'$\vec{\beta}_{\rm CHSH}$', (0, 0), xytext=(3, -10), textcoords='offset points',
                fontsize=7.5, color=BLUES[3])

    handles = [Line2D([], [], color=colour, lw=line_width(colour), label=label)
               for _, colour, label in POLYGONS[::-1]]
    ax.legend(handles=handles, title=r'$\theta$', fontsize=6.5, title_fontsize=7, frameon=False,
              loc='center left', bbox_to_anchor=(1.0, 0.5), handlelength=2.2)

    ax.set_xlabel(r'$r_0$')
    ax.set_ylabel(r'$r_1$')
    ax.set_aspect('equal')
    ax.grid(color=GRID, lw=0.5)
    ax.set_axisbelow(True)
    fig.tight_layout()
    fig.savefig(OUT + '.pdf', bbox_inches='tight')
    fig.savefig(OUT + '.png', dpi=200, bbox_inches='tight')
    print('saved', OUT)


if __name__ == '__main__':
    main()
