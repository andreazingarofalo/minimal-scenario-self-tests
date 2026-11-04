"""NPA bounds at the vertices of the local polygon P_theta, the table of NPA levels in Appendix C.

For theta = pi/8 + k pi/64, k = 1, ..., 8, the script solves the NPA relaxation of every vertex inequality at
seven levels and prints the largest value of omega_q - 1 over the vertices. The SDPs are solved with MOSEK
via ncpol2sdpa, eight in parallel. The script stops with an error if a solve is not optimal.
"""
import collections
import sys

import numpy as np
from joblib import Parallel, delayed

from common import beta_vec, local_polygon
from npa import npa

LEVELS = ['0', 'AB', 'AA+BB+AB', 'AB+ABB', 'BB+AB+ABB', 'AB+AAB+ABB', 'AA+BB+AB+ABB']
THETAS = list(np.pi / 8 + np.arange(1, 9) * np.pi / 64)


def solve(theta, r, level):
    """Dual value and solver status for the inequality beta_theta(r0, r1)."""
    _, dual, status = npa(beta_vec(theta, *r), level)
    return dual, status


def main():
    jobs = [(theta, i, tuple(r), level)
            for theta in THETAS
            for i, (r, _) in enumerate(local_polygon(theta))
            for level in LEVELS]
    results = Parallel(n_jobs=8)(delayed(solve)(theta, r, level) for theta, _, r, level in jobs)

    failed = [(theta, i, level, status)
              for (theta, i, _, level), (_, status) in zip(jobs, results) if status != 'optimal']
    if failed:
        sys.exit(f'{len(failed)} solves without an optimal solution, '
                 f'e.g. (theta, vertex, level, status) = {failed[:5]}')

    duals = collections.defaultdict(list)    # (theta, level) -> dual values at the vertices
    for (theta, _, _, level), (dual, _) in zip(jobs, results):
        duals[theta, level].append(dual)

    print('max over vertices of (NPA dual value - 1), per theta (rows) and level (cols)')
    print('theta    ' + ' '.join(f'{level:>13s}' for level in LEVELS))
    for theta in THETAS:
        print(f'{theta:.4f} ' + ' '.join(f'{max(duals[theta, level]) - 1:+13.2e}' for level in LEVELS))
    print('min over vertices of (NPA dual - 1):', min(min(d) for d in duals.values()) - 1)
    print('statuses:', set(status for _, status in results))


if __name__ == '__main__':
    main()
