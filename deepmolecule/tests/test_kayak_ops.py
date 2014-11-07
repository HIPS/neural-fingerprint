import numpy as np
import numpy.random as npr
import sys
sys.path.append('../../Kayak/')
import kayak as ky
import kayak_ops

MAX_GRAD_DIFF  = 1e-7
npr.seed(1)

def test_NeighborMatMult():
    N_atoms = 12
    N_bonds = 9
    D1 = 10
    D2 = 11
    N_trials = 5

    features = ky.Parameter(npr.randn(N_bonds, D1))
    weights = {i : ky.Parameter(npr.randn(i * D1, D2)) for i in [1, 2, 3, 4]}
    neighbors = ky.Parameter([list(npr.randint(N_bonds, size=npr.randint(1, 5)))
                              for i in xrange(N_atoms)])
    out_weights = ky.Parameter(None)

    out = ky.MatSum(ky.MatElemMult(
        kayak_ops.NeighborMatMult(features, neighbors, weights),
        out_weights))

    for i in xrange(N_trials):
        out_weights.value = npr.randn(N_atoms, D2)
        assert ky.util.checkgrad(features, out) < MAX_GRAD_DIFF
        for i_w in [1, 2, 3, 4]:
            assert ky.util.checkgrad(weights[i_w], out) < MAX_GRAD_DIFF

