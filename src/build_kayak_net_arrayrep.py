import numpy as np
import numpy.random as npr
import itertools as it
import kayak as ky
import kayak_ops as mk
from features import N_atom_features, N_bond_features

def all_permutations(N):
    return [permutation for permutation in it.permutations(range(N))]

def softened_max(X_list):
    X_cat = ky.ListToArray(*X_list)
    X_feature_0 = ky.Take(X_cat, slice(0,1), axis=2)
    return ky.MatSum(ky.MatElemMult(X_cat, ky.SoftMax(X_feature_0, axis=0)),
                     axis=0, keepdims=False)

def matmult_neighbors(mol_graph, self_ntype, other_ntypes, feature_sets,
                      weights_gen, permutations=False):
    def neighbor_list(degree, other_ntype):
        return mk.get_neighbor_list(mol_graph, ((self_ntype, degree), other_ntype))
    result_by_degree = []
    for degree in [1, 2, 3, 4]:
        # dims of stacked_neighbors are (atoms, neighbors, features)
        stacked_neighbors = ky.Concatenate(2,
            *[mk.NeighborStack(neighbor_list(degree, other_ntype), features)
              for other_ntype, features in zip(other_ntypes, feature_sets)])
        if permutations:
            weightses = [weights_gen() for i in range(degree)]
            neighbors = [ky.Take(stacked_neighbors, i, axis=1) for i in range(degree)]
            products = [[ky.MatMult(n, w) for w in weightses] for n in neighbors]
            candidates = [ky.MatAdd(*[products[i][j] for i, j in enumerate(p)])
                          for p in all_permutations(degree)]
            # dims of candidates are (atoms, features)
            result_by_degree.append(softened_max(candidates))
        else:
            result_by_degree.append(ky.MatSum(
                ky.TensorMult(stacked_neighbors, weights_gen(), axes=((2,), (0,))),
                axis=1, keepdims=False))

    return ky.Concatenate(0, *result_by_degree)

def softmax_neighbors(mol_graph, ntypes, features):
    idxs = mk.get_neighbor_list(mol_graph, ntypes)
    return mk.NeighborSoftenedMax(idxs, features)

def build_universal_net(num_hidden, param_scale, permutations=False):
    # Derived parameters
    layer_sizes = [N_atom_features] + num_hidden
    k_weights = []
    def weights_gen(shape):
        def new_weights():
            new = ky.Parameter(np.random.randn(*shape) * param_scale)
            k_weights.append(new)
            return new
        return new_weights

    mols = ky.Blank()
    cur_atoms = mk.get_feature_array(mols, 'atom')
    cur_bonds = mk.get_feature_array(mols, 'bond')
    for N_prev, N_cur in zip(layer_sizes[:-1], layer_sizes[1:]):
        cur_atoms = ky.Logistic(ky.MatAdd(
            ky.MatMult(cur_atoms, weights_gen((N_prev, N_cur))()),
            matmult_neighbors(mols, 'atom', ('atom', 'bond'), (cur_atoms, cur_bonds),
                              weights_gen((N_prev + N_bond_features, N_cur)),
                              permutations)))

    fixed_sized_output = softmax_neighbors(mols, ('molecule', 'atom'), cur_atoms)
    output = ky.MatMult(fixed_sized_output, weights_gen((N_cur, 1))())
    target = ky.Blank()
    loss = ky.L2Loss(output, target)
    return mols, target, loss, output, k_weights
