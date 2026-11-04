"""First level 1+A+B and almost-quantum level 1+A+B+AB on the plane of candidates beta_theta(r0, r1).

For three values of theta the script minimises the NPA bound over (r0, r1) with Nelder-Mead, starting from the
centroid of P_theta. When the minimum is one, it measures the set where the bound is one by bisection along
eight rays from the minimiser.
These are the numbers of the subsection "NPA relaxations and numerical methods" of Appendix C.
The SDPs are solved with MOSEK via ncpol2sdpa. The script takes about half a minute.
"""
import numpy as np
from scipy.optimize import minimize

from common import beta_vec, local_polygon
from npa import npa_dual

THETAS = [np.pi / 8 + np.pi / 64, np.pi / 8 + 5 * np.pi / 64, np.pi / 4]
LEVELS = ['0', 'AB']
TOL = 1e-7    # the bound counts as one when it is at most 1 + TOL, as in make_fam_npa_figure.py


def npa_bound(theta, level):
    """The NPA bound as a function of r = (r0, r1)."""
    return lambda r: npa_dual(beta_vec(theta, *r), level)


def sublevel_radii(f, center):
    """Distance from center to the boundary of {f <= 1 + TOL} along 8 rays, by bisection in [0, 1]."""
    out = []
    for phi in np.linspace(0, 2 * np.pi, 8, endpoint=False):
        direction = np.array([np.cos(phi), np.sin(phi)])
        lo, hi = 0., 1.
        for _ in range(30):
            mid = (lo + hi) / 2
            if f(center + mid * direction) <= 1 + TOL:
                lo = mid
            else:
                hi = mid
        out.append(lo)
    return out


def main():
    for theta in THETAS:
        vertices = np.array([r for r, _ in local_polygon(theta)])
        centroid = vertices.mean(0)
        for level in LEVELS:
            f = npa_bound(theta, level)
            res = minimize(f, centroid, method='Nelder-Mead', options=dict(xatol=1e-7, fatol=1e-10))
            # if the minimum is above one, no inequality of the plane is certified at this level
            rad = None if f(res.x) > 1 + TOL else np.round(sublevel_radii(f, res.x), 4)
            print(f'theta={theta:.4f} level 1+A+B+{level:3s}: min value {res.fun:.9f} '
                  f'at {np.round(res.x, 5)}, centroid {np.round(centroid, 5)} value {f(centroid):.6f}; '
                  f'radii of set: {rad}')


if __name__ == '__main__':
    main()
