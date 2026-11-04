# Code for "All Bell inequalities with a quantum advantage in the minimal scenario are self-testing"

M.-E. Liu, A. Zingarofalo, V. Barizien, J.-D. Bancal (2026).

Code for Appendix C of the paper, on the partially entangled Tsirelson points $\vec P_\theta$: state
$\cos\theta|00\rangle+\sin\theta|11\rangle$, measurement angles $(a_0,a_1,b_0,b_1)=(\pi/4,-\pi/4,0,\pi/2)$,
$\theta\in(\pi/8,\pi/4]$.

| In the paper | Files |
|---|---|
| Plane of candidate Bell functionals | `mathematica/Orthogonal.nb` |
| Exact proof of the vertex table of the local polygon (Proposition "Local polygon") | `mathematica/Local_Polygon_Proof.nb` |
| Sum-of-squares certificates and their exact verification (Proposition "Sum-of-squares certificates", Table I) | `mathematica/Uniform_SOS_Verification.nb`, with the certificates in `mathematica/UniformSOS_data.wl` |
| NPA relaxations (Table II and the numbers of the subsection "NPA relaxations and numerical methods") | `python/npa_vertices.py`, `python/lowlevels.py` |
| Figures 2, 3 and 4 (`figures/`) | `python/make_fam_faces_figure.py`, `python/make_fam_vertices_figure.py`, `python/make_fam_npa_figure.py` |

`python/common.py` (the plane and the local polygon), `python/lines_and_thetabar.py` (the vertex table and
$\bar\theta$) and `python/npa.py` (the NPA levels) are modules used by these scripts. Correlations are ordered as
$(\langle A_0\rangle,\langle A_1\rangle,\langle B_0\rangle,\langle B_1\rangle,\langle A_0B_0\rangle,\langle A_0B_1\rangle,\langle A_1B_0\rangle,\langle A_1B_1\rangle)$.
The exact computations use $v\in(0,1]$ with $\tan\theta=(\sqrt2-1)+(2-\sqrt2)\,v$, the parameter $\xi$ of the paper.

## Mathematica

`Orthogonal.nb` is the notebook of A. Zingarofalo's Master's thesis, where the plane of candidate inequalities is
derived (in its notation, `r0 = p[1]` and `r1 = 2 p[4]`). `Local_Polygon_Proof.nb` proves the vertex table with
`Resolve` and runs in about ten seconds. `Uniform_SOS_Verification.nb` checks the six certificates exactly and needs
NCAlgebra (https://github.com/NCAlgebra/NC); it reads `UniformSOS_data.wl` from its own folder. All three notebooks
are saved with their outputs.

## Python

`pip install -r requirements.txt` (Python 3.13; MOSEK needs a licence, free for academic use), then run any script,
e.g. `python python/npa_vertices.py`. The NPA scripts stop with an error if a solve is not optimal. The bounds they
report are the values of the sum-of-squares (dual) problems, and the region certified by a level is the numerical
set where this bound is at most $1+10^{-7}$. `lowlevels.py` takes about 20 seconds, the others a few seconds. The
first run of `make_fam_npa_figure.py` computes the region boundaries (one or two minutes) and stores them in `data/`.

## Licence

MIT (`LICENSE`). If you use this code, please cite the paper.
