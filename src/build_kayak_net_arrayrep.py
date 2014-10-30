import numpy as np
import numpy.random as npr
import kayak as ky
import kayak_ops as mk
from features import N_atom_features, N_bond_features

def cat_weights(w_list):
    return {i + 1: ky.Concatenate(0, *([w] * (i + 1)))
            for i, w in enumerate(w_list)}

def get_feature_array(ky_mol_graph, ntype):
    return ky.Blank([ky_mol_graph], lambda p : p[0].value.feature_array(ntype))

def get_neighbor_list(ky_mol_graph, ntypes):
    return ky.Blank([ky_mol_graph], lambda p : p[0].value.neighbor_list(*ntypes))

def softened_max(features):
    return ky.MatSum(ky.ElemMult(features, ky.SoftMax(features, axis=0)), axis=0)

def build_universal_net(num_hidden, param_scale):
    # Derived parameters
    layer_sizes = [N_atom_features] + num_hidden
    k_weights = []
    def new_weights(shape):
        new = ky.Parameter(np.random.randn(*shape) * param_scale)
        k_weights.append(new)
        return new

    mols = ky.Blank()
    cur_atoms = get_feature_array(mols, 'atom')
    cur_bonds = get_feature_array(mols, 'bond')
    atom_atom_neighbors = get_neighbor_list(mols, ('atom', 'atom'))
    atom_bond_neighbors = get_neighbor_list(mols, ('atom', 'bond'))
    mol_atoms = get_neighbor_list(mols, ('molecule', 'atom'))
    for N_prev, N_curr in zip(layer_sizes[:-1], layer_sizes[1:]):
        w_self = new_weights((N_prev, N_curr))
        w_atom_cat = cat_weights([new_weights((N_prev, N_curr)) for i in range(4)])
        w_bond_cat = cat_weights([new_weights((N_bond_features, N_curr)) for i in range(4)])
        cur_atoms = ky.Logistic(ky.MatAdd(
            ky.MatMult(cur_atoms, w_self),
            mk.NeighborMatMult(atom_atom_neighbors, cur_atoms, w_atom_cat),
            mk.NeighborMatMult(atom_bond_neighbors, cur_bonds, w_bond_cat)))

    fixed_sized_output = mk.NeighborSoftenedMax(mol_atoms, cur_atoms)
    output = ky.MatMult(fixed_sized_output, new_weights((N_curr, 1)))
    target = ky.Blank()
    loss = ky.L2Loss(output, target)
    return mols, target, loss, output, k_weights
