"""The plane of candidates and the local polygon, numerically.

The family P_theta uses the state cos(theta)|00> + sin(theta)|11> and the measurement angles of the
Tsirelson point, (a0, a1, b0, b1) = (pi/4, -pi/4, 0, pi/2). beta_vec gives the plane of candidates
beta_theta(r0, r1) of Appendix C, and local_polygon the local polygon, the set of (r0, r1) where
beta_theta(r0, r1) . D <= 1 for all 16 deterministic points D. Correlation vectors are ordered as
(A0, A1, B0, B1, A0B0, A0B1, A1B0, A1B1).
"""
import itertools

import numpy as np

SQ2 = np.sqrt(2)


def beta_vec(th, r0, r1):
    """beta_theta(r0, r1), written as in the Master's thesis (p4 = r1/2)."""
    c2, c4, c6 = np.cos(2*th), np.cos(4*th), np.cos(6*th)
    s2, s4 = np.sin(2*th), np.sin(4*th)
    p4 = r1/2
    k = 1/(6 - 2*c4)
    beta = np.zeros(8)
    beta += p4*np.array([2*SQ2*(3 - c4)*s2, -2*SQ2*(3 - c4)*s2, 0, -4*(3 - c4),
                         0, SQ2*(5*c2 - c6), 0, SQ2*(5*c2 - c6)])
    beta += r0*np.array([SQ2*(3 - c4), SQ2*(3 - c4), -4*(1 - c4), 0,
                         -2*SQ2*c2, SQ2*s4, -2*SQ2*c2, -SQ2*s4])
    beta += np.array([0, 0, 4*c2, 0,
                      SQ2*(1 - c4), 2*SQ2*s2, SQ2*(1 - c4), -2*SQ2*s2])
    return k*beta


def beta_affine(th):
    """beta^(0), beta^(1), beta^(2), so that beta_theta(r0, r1) = beta^(0) + r0 beta^(1) + r1 beta^(2)."""
    b0 = beta_vec(th, 0, 0)
    return b0, beta_vec(th, 1, 0) - b0, beta_vec(th, 0, 1) - b0


# The 16 deterministic points, one per row
DET = np.array([[A0, A1, B0, B1, A0*B0, A0*B1, A1*B0, A1*B1]
                for A0, A1, B0, B1 in itertools.product([1, -1], repeat=4)], float)


def P_point(th, a0=np.pi/4, a1=-np.pi/4, b0=0., b1=np.pi/2):
    """Correlations of cos(theta)|00> + sin(theta)|11> for the observables cos(a) Z + sin(a) X."""
    c2, s2 = np.cos(2*th), np.sin(2*th)
    E = [np.cos(a)*np.cos(b) + s2*np.sin(a)*np.sin(b) for a in (a0, a1) for b in (b0, b1)]
    return np.array([c2*np.cos(a0), c2*np.cos(a1), c2*np.cos(b0), c2*np.cos(b1)] + E)


def local_polygon(th, tol=1e-10):
    """Vertices of the local polygon, counterclockwise.

    Returns a list of pairs (vertex, indices of the rows of DET whose constraint is active there).
    """
    b0, e0, e1 = beta_affine(th)
    # beta . D <= 1 for all rows D of DET reads G r <= h
    G = np.stack([DET @ e0, DET @ e1], 1)
    h = 1 - DET @ b0
    verts = []
    for i, j in itertools.combinations(range(16), 2):
        M = G[[i, j]]
        if abs(np.linalg.det(M)) < 1e-12:
            continue
        x = np.linalg.solve(M, h[[i, j]])
        if np.all(G @ x <= h + tol) and not any(np.linalg.norm(x - v) < 1e-8 for v, _ in verts):
            active = tuple(np.where(np.abs(G @ x - h) < 1e-8)[0])
            verts.append((x, active))
    centre = np.mean([v for v, _ in verts], 0)
    verts.sort(key=lambda t: np.arctan2(t[0][1] - centre[1], t[0][0] - centre[0]))
    return verts
