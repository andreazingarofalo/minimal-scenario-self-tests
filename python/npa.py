"""NPA bounds on the quantum value of a correlator Bell expression in the minimal scenario.

npa(beta, level) and npa_dual(beta, level) solve the NPA relaxation at the levels used in Appendix C,
with ncpol2sdpa and the MOSEK solver. Run as a script, the module prints every level for the vertex
(r0, r1) = (0, 1 - 1/sqrt2) of the octagon at theta = pi/4.
"""
import sys

import ncpol2sdpa as ncp
import numpy as np

from common import beta_vec

# One operator per measurement. With X^2 = 1 they are the +-1 observables A_x and B_y, and Alice's
# operators commute with Bob's.
A = ncp.generate_measurements([2, 2], 'A')
B = ncp.generate_measurements([2, 2], 'B')
A_OPS = ncp.flatten(A)
B_OPS = ncp.flatten(B)
OPS = ncp.flatten([A, B])
SUBSTITUTIONS = {op**2: 1 for op in OPS}
for a in A_OPS:
    for b in B_OPS:
        SUBSTITUTIONS[b * a] = a * b

A0, A1 = A_OPS
B0, B1 = B_OPS
# Same order as the vector beta: (A0, A1, B0, B1, A0B0, A0B1, A1B0, A1B1).
CORRELATORS = [A0, A1, B0, B1, A0 * B0, A0 * B1, A1 * B0, A1 * B1]


def extra_monomials(level):
    """Monomials added to 1, A, B. For example level = 'AB+ABB' gives the level 1+A+B+AB+ABB'."""
    if level in [0, '0', None]:
        return []
    parts = level.split('+')
    monomials = []
    if 'AA' in parts:
        monomials += [a1 * a2 for a1 in A_OPS for a2 in A_OPS]
    if 'BB' in parts:
        monomials += [b1 * b2 for b1 in B_OPS for b2 in B_OPS]
    if 'AB' in parts:
        monomials += [a * b for a in A_OPS for b in B_OPS]
    if 'AAB' in parts:
        monomials += [a1 * a2 * b for a1 in A_OPS for a2 in A_OPS for b in B_OPS]
    if 'ABB' in parts:
        monomials += [a * b1 * b2 for a in A_OPS for b1 in B_OPS for b2 in B_OPS]
    return monomials


def npa(beta, level):
    """Primal value, dual value and solver status of max beta.P over the NPA level 1+A+B+level."""
    objective = sum(float(b) * m for b, m in zip(beta, CORRELATORS))
    sdp = ncp.SdpRelaxation(OPS, verbose=0, normalized=True, parallel=0)
    # Level 0 of ncpol2sdpa already contains the operators of degree one, so the moment matrix is
    # indexed by 1, A, B and the extra monomials. ncpol2sdpa minimises, hence the sign.
    sdp.get_relaxation(level=0, objective=-objective, substitutions=SUBSTITUTIONS,
                       extramonomials=extra_monomials(level))
    sdp.solve(solver='mosek')
    # With the MOSEK backend, ncpol2sdpa returns the value of the sum-of-squares problem in sdp.primal and
    # that of the moment problem in sdp.dual, the opposite of its SDPA convention. We swap them back, so
    # that the dual value is the sum-of-squares one, the upper bound on beta.P.
    return -sdp.dual, -sdp.primal, sdp.status


def npa_dual(beta, level):
    """Upper bound (sum-of-squares value) of the NPA level. Raises RuntimeError if the solver is not optimal."""
    _, dual, status = npa(beta, level)
    if status != 'optimal':
        raise RuntimeError(f'NPA level 1+A+B+{level}: solver status {status}')
    return dual


def main():
    levels = ['0', 'AA', 'BB', 'AB', 'AAB', 'ABB', 'AB+ABB', 'AA+AB+ABB', 'BB+AB+ABB', 'AA+BB+AB+ABB',
              'AB+AAB+ABB', 'AA+BB+AB']
    print('NPA levels at theta=pi/4, vertex (r0,r1)=(0,1-1/sqrt2)')
    beta = beta_vec(np.pi / 4, 0, 1 - 1 / np.sqrt(2))
    failed = []
    for level in levels:
        primal, dual, status = npa(beta, level)
        print(f'   1+A+B+{level:14s} primal {primal:.10f} dual {dual:.10f} {status}')
        if status != 'optimal':
            failed.append(level)
    if failed:
        sys.exit(f'solver did not reach an optimal solution at levels {failed}')


if __name__ == '__main__':
    main()
