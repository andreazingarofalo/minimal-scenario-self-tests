"""Lines of the deterministic points, the vertex table of the local polygon, and theta_bar.

Each of the eight CHSH-saturating deterministic points gives a constraint g0 r0 + g1 r1 <= h in the plane
of candidates (Appendix C, local constraints in the plane). The vertex table gives each vertex of the local
polygon as the intersection of two such lines, as in Proposition (Local polygon): a hexagon for
theta <= theta_bar and an octagon above. Run as a script, it checks the table against common.local_polygon
and derives the equation of theta_bar, where three lines meet.
"""
import sys

import numpy as np
import sympy as sp

from common import SQ2, local_polygon

# theta_bar = arcsin(s_bar)/2, with s_bar the root of 2 s^3 + 1 = sqrt2 (1 + s) in (1/sqrt2, 1)
THETA_BAR = 0.6445395332517548


def line_coefficients(kind, a, b, c, s, r2):
    """Coefficients (g0, g1, h) of the constraint g0 r0 + g1 r1 <= h of the point kind(a, b).

    c = cos(2 theta), s = sin(2 theta) and r2 = sqrt2, as floats or as sympy expressions.
    I(a,b): A0 = A1 = B0 = a, B1 = b. II(a,b): A0 = -A1 = B1 = a, B0 = b.
    """
    D = 1 + s*s
    if kind == 'I':
        return a*(r2 - (2 - r2)*s*s) - r2*c, b*D*(r2*a*c - 1), D - a*c - r2*s*s
    return s*(r2*c - 2*b*s), a*D*(r2*s - 1), D - b*c - r2*s


def lines(th):
    """Dictionary (kind, a, b) -> (g, h) of the eight constraints g . r <= h at theta."""
    c, s = np.cos(2*th), np.sin(2*th)
    out = {}
    for a in (1, -1):
        for b in (1, -1):
            for kind in ('I', 'II'):
                g0, g1, h = line_coefficients(kind, a, b, c, s, SQ2)
                out[(kind, a, b)] = (np.array([g0, g1]), h)
    return out


def vert(l1, l2):
    """Intersection of the two lines l1 = (g1, h1) and l2 = (g2, h2), by Cramer's rule."""
    (g1, h1), (g2, h2) = l1, l2
    det = g1[0]*g2[1] - g1[1]*g2[0]
    return np.array([h1*g2[1] - h2*g1[1], g1[0]*h2 - g2[0]*h1])/det


# Vertex table: hexagon V1, V2, V3, V4, V3', V2' and octagon V1, V5, V6, V3, V4, V3', V6', V5'.
PAIRS_6 = [(('II', 1, 1), ('II', -1, 1)), (('I', 1, 1), ('II', -1, 1)), (('I', 1, 1), ('II', -1, -1)),
           (('II', 1, -1), ('II', -1, -1)), (('I', 1, -1), ('II', 1, -1)), (('I', 1, -1), ('II', 1, 1))]
PAIRS_8 = [(('II', 1, 1), ('II', -1, 1)), (('II', -1, 1), ('I', -1, 1)), (('I', 1, 1), ('I', -1, 1)),
           (('I', 1, 1), ('II', -1, -1)), (('II', 1, -1), ('II', -1, -1)), (('I', 1, -1), ('II', 1, -1)),
           (('I', 1, -1), ('I', -1, -1)), (('II', 1, 1), ('I', -1, -1))]


def polygon(th):
    """Vertices of the local polygon from the vertex table, in that order, as an array of shape (n, 2)."""
    L = lines(th)
    pairs = PAIRS_6 if th <= THETA_BAR else PAIRS_8
    return np.array([vert(L[p], L[q]) for p, q in pairs])


def main():
    max_err = 0
    for th in np.linspace(np.pi/8 + 1e-4, np.pi/4, 300):
        found = np.array([v for v, _ in local_polygon(th)])
        table = polygon(th)
        assert len(found) == len(table), th
        max_err = max(max_err, max(min(np.linalg.norm(found - v, axis=1)) for v in table))
    print('compact lines + vertex table reproduce the polygon, max error', max_err)
    if max_err > 1e-8:
        sys.exit('vertex table does not reproduce the polygon')

    # At theta_bar the lines of I(+1,+1), II(-1,+1), I(-1,+1) meet: the 3x3 determinant of their
    # coefficients (g0, g1, h) vanishes. Eliminate c with c^2 + s^2 = 1.
    c, s = sp.symbols('c s')
    q = sp.sqrt(2)
    M = sp.Matrix([line_coefficients('I', 1, 1, c, s, q),
                   line_coefficients('II', -1, 1, c, s, q),
                   line_coefficients('I', -1, 1, c, s, q)])
    res = sp.factor(sp.expand(sp.resultant(sp.expand(M.det()), c**2 + s**2 - 1, c)), extension=q)
    print('resultant in s = sin(2 theta), factored over Q(sqrt2):', res)
    s_bar = sp.nsolve(2*s**3 + 1 - q*(1 + s), s, 0.96)
    print('root of 2s^3+1 = sqrt2(1+s) in (1/sqrt2,1):', s_bar, ' theta_bar =', sp.asin(s_bar)/2)


if __name__ == '__main__':
    main()
