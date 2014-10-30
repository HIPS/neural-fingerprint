import numpy as np
import numpy.random as npr
import kayak as ky
import kayak_ops as mk
from features import N_atom_features, N_bond_features

def matmult_neighbors(mol_graph, ntypes, features, weights_gen):
    def neighbor_list(degree):
        return mk.get_neighbor_list(mol_graph, ((ntypes[0], degree), ntypes[1]))

    return ky.Concatenate(0,
        *[ky.MatMult(mk.NeighborCat(neighbor_list(degree), features),
                     ky.Concatenate(0, *([weights_gen()] * degree)))
          for degree in [1, 2, 3, 4]])

def softmax_neighbors(mol_graph, ntypes, features):
    idxs = mk.get_neighbor_list(mol_graph, ntypes)
    return mk.NeighborSoftenedMax(idxs, features)

def build_universal_net(num_hidden, param_scale):
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
            matmult_neighbors(mols, ('atom', 'atom'), cur_atoms,
                              weights_gen((N_prev, N_cur))),
            matmult_neighbors(mols, ('atom', 'bond'), cur_bonds,
                              weights_gen((N_bond_features, N_cur)))))

    fixed_sized_output = softmax_neighbors(mols, ('molecule', 'atom'), cur_atoms)
    output = ky.MatMult(fixed_sized_output, weights_gen((N_cur, 1))())
    target = ky.Blank()
    loss = ky.L2Loss(output, target)
    return mols, target, loss, output, k_weights
