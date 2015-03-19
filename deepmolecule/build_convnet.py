import itertools as it
import kayak as ky
import kayak_ops as mk
import numpy as np
from features import N_atom_features, N_bond_features
from util import WeightsContainer, c_value, c_grad, memoize
from mol_graph import graph_from_smiles_tuple

def all_permutations(N):
    return [permutation for permutation in it.permutations(range(N))]

def neighbor_stack(idxs, features, num_neighbors):
    """dims of result are (atoms, neighbors, features)"""
    result_rows = np.zeros((len(idxs), num_neighbors, features.shape[1]))
    for i, idx_list in enumerate(idxs):
        result_rows[i, :, :] = features[idx_list, :]
    return result_rows

def neighbor_cat(idxs, features):
    result_rows = []
    for idx_list in idxs:
        result_rows.append(np.concatenate(
            [features[i, :] for i in idx_list]))
    return np.array(result_rows)

def softened_max(X):
    exp_X = np.exp(X)
    return np.sum(exp_X * X, axis=0) / np.sum(exp_X, axis=0)

def neighbor_softened_max(idxs, features):
    result_rows = []
    for idx_list in idxs:
        result_rows.append(softened_max(features[idx_list, :]))
    return np.array(result_rows)

def neighbor_sum(idxs, features):
    result_rows = []
    for idx_list in idxs:
        result_rows.append(np.sum(features[idx_list, :], axis=0))
    return np.array(result_rows)


def softened_max(X_list):
    X_cat = ky.ListToArray(*X_list)
    X_feature_0 = ky.Take(X_cat, slice(0,1), axis=2)
    return ky.MatSum(ky.MatElemMult(X_cat, ky.SoftMax(X_feature_0, axis=0)),
                     axis=0, keepdims=False)


def matmult_neighbors_ag(mol_nodes, other_ntypes, feature_sets, weights_gen, permutations=False):
    def neighbor_list(degree, other_ntype):
        return mol_nodes[(other_ntype + '_neighbors', degree)]
    result_by_degree = []
    for degree in [1, 2, 3, 4]:
        # dims of stacked_neighbors are [atoms, neighbors (as in atom-bond pairs), features]
        stacked_neighbors = ky.Concatenate(2,
            *[mk.NeighborStack(neighbor_list(degree, other_ntype), features, degree)
              for other_ntype, features in zip(other_ntypes, feature_sets)])
        if permutations:
            weightses = [weights_gen(degree, " neighbour " + str(n)) for n in range(degree)]
            neighbors = [ky.Take(stacked_neighbors, d, axis=1) for d in range(degree)]
            products = [[ky.MatMult(n, w) for w in weightses] for n in neighbors]
            candidates = [ky.MatAdd(*[products[i][j] for i, j in enumerate(p)])
                          for p in all_permutations(degree)]
            # dims of candidates are (atoms, features)
            result_by_degree.append(softened_max(candidates))
        else:
            result_by_degree.append(ky.MatSum(
                ky.TensorMult(stacked_neighbors, weights_gen(degree), axes=((2,), (0,))),
                axis=1, keepdims=False))

    # This is brittle! Relies on atoms being sorted by degree in the first place,
    # in Node.graph_from_smiles_tuple()
    return ky.Concatenate(0, *result_by_degree)


def build_universal_net_ag(bond_vec_dim=1, num_hidden_features=[20, 50, 50],
                        permutations=False, l2_penalty=0.0):
    """Sets up a Kayak graph to compute convnets over all molecules in a minibatch together.
       The number of hidden layers is the length of num_hidden_features - 1."""
    mol_nodes = {'atom_features' : ky.Blank(),
                 'bond_features' : ky.Blank(),
                 'mol_atom_neighbors' : ky.Blank()}
    for degree in [1, 2, 3, 4]:
        mol_nodes[('atom_neighbors', degree)] = ky.Blank()
        mol_nodes[('bond_neighbors', degree)] = ky.Blank()

    weights = WeightsContainer()
    cur_atoms = ky.MatMult(mol_nodes['atom_features'],
        weights.new((N_atom_features, num_hidden_features[0]), name='atom2vec'))
    cur_bonds = ky.MatMult(mol_nodes['bond_features'],
        weights.new((N_bond_features, bond_vec_dim), name='bond2vec'))

    in_and_out_sizes = zip(num_hidden_features[:-1], num_hidden_features[1:])
    for layer, (N_prev, N_cur) in enumerate(in_and_out_sizes):
        def new_weights_func(degree, neighbourstr=""):
            return weights.new((N_prev + bond_vec_dim, N_cur),
                name="layer " + str(layer) + " degree " + str(degree) + neighbourstr + " filter" )
        layer_bias = weights.new((1, N_cur), name="layer " + str(layer) + " biases")
        self_activations = ky.MatMult(cur_atoms,
            weights.new((N_prev, N_cur), name="layer " + str(layer) + " self filter"))
        neighbour_activations = matmult_neighbors_ag(mol_nodes, ('atom', 'bond'),
            (cur_atoms, cur_bonds), new_weights_func, permutations)
        cur_atoms = ky.TanH(layer_bias + self_activations + neighbour_activations)

    # Include both a softened-max and a sum node to pool all atom features together.
    mol_atom_neighbors = mol_nodes['mol_atom_neighbors']
    fixed_sized_softmax = mk.NeighborSoftenedMax(mol_atom_neighbors, cur_atoms)
    fixed_sized_sum = mk.NeighborSum(mol_atom_neighbors, cur_atoms)
    fixed_sized_output = ky.Concatenate(1, fixed_sized_softmax, fixed_sized_sum)
    output_bias = weights.new((1, ), "output bias")
    output_weights = weights.new((N_cur * 2, ), "output weights")
    output = ky.MatMult(fixed_sized_output, output_weights) + output_bias
    target = ky.Blank()
    unreg_loss = ky.L2Loss(output, target)
    l2regs = [ky.L2Loss(w, ky.Parameter(np.zeros(w.shape))) * ky.Parameter(l2_penalty)
              for w in weights._weights_list ]
    loss = ky.MatAdd(unreg_loss, *l2regs)

    def make_input_dict(smiles):
        array_rep = arrayrep_from_smiles(tuple(smiles))
        return {mol_nodes[k] : v for k, v in array_rep.iteritems()}

    def grad_fun(w, smiles, t):
        input_dict = make_input_dict(smiles)
        input_dict[weights] = w
        input_dict[target] = t
        return c_grad(loss, weights, input_dict)
    def loss_fun(w, smiles, t):
        input_dict = make_input_dict(smiles)
        input_dict[weights] = w
        input_dict[target] = t
        return c_value(loss, input_dict)
    def pred_fun(w, smiles):
        input_dict = make_input_dict(smiles)
        input_dict[weights] = w
        return c_value(output, input_dict)
    def output_layer_fun(w, smiles):
        input_dict = make_input_dict(smiles)
        input_dict[weights] = w
        return c_value(fixed_sized_output, input_dict)

    return loss_fun, grad_fun, pred_fun, output_layer_fun, weights



def matmult_neighbors(mol_nodes, other_ntypes, feature_sets, weights_gen, permutations=False):
    def neighbor_list(degree, other_ntype):
        return mol_nodes[(other_ntype + '_neighbors', degree)]
    result_by_degree = []
    for degree in [1, 2, 3, 4]:
        # dims of stacked_neighbors are [atoms, neighbors (as in atom-bond pairs), features]
        stacked_neighbors = ky.Concatenate(2,
            *[mk.NeighborStack(neighbor_list(degree, other_ntype), features, degree)
              for other_ntype, features in zip(other_ntypes, feature_sets)])
        if permutations:
            weightses = [weights_gen(degree, " neighbour " + str(n)) for n in range(degree)]
            neighbors = [ky.Take(stacked_neighbors, d, axis=1) for d in range(degree)]
            products = [[ky.MatMult(n, w) for w in weightses] for n in neighbors]
            candidates = [ky.MatAdd(*[products[i][j] for i, j in enumerate(p)])
                          for p in all_permutations(degree)]
            # dims of candidates are (atoms, features)
            result_by_degree.append(softened_max(candidates))
        else:
            result_by_degree.append(ky.MatSum(
                ky.TensorMult(stacked_neighbors, weights_gen(degree), axes=((2,), (0,))),
                axis=1, keepdims=False))

    # This is brittle! Relies on atoms being sorted by degree in the first place,
    # in Node.graph_from_smiles_tuple()
    return ky.Concatenate(0, *result_by_degree)


def build_universal_net(bond_vec_dim=1, num_hidden_features=[20, 50, 50],
                        permutations=False, l2_penalty=0.0):
    """Sets up a Kayak graph to compute convnets over all molecules in a minibatch together.
       The number of hidden layers is the length of num_hidden_features - 1."""
    mol_nodes = {'atom_features' : ky.Blank(),
                 'bond_features' : ky.Blank(),
                 'mol_atom_neighbors' : ky.Blank()}
    for degree in [1, 2, 3, 4]:
        mol_nodes[('atom_neighbors', degree)] = ky.Blank()
        mol_nodes[('bond_neighbors', degree)] = ky.Blank()

    weights = WeightsContainer()
    cur_atoms = ky.MatMult(mol_nodes['atom_features'],
        weights.new((N_atom_features, num_hidden_features[0]), name='atom2vec'))
    cur_bonds = ky.MatMult(mol_nodes['bond_features'],
        weights.new((N_bond_features, bond_vec_dim), name='bond2vec'))

    in_and_out_sizes = zip(num_hidden_features[:-1], num_hidden_features[1:])
    for layer, (N_prev, N_cur) in enumerate(in_and_out_sizes):
        def new_weights_func(degree, neighbourstr=""):
            return weights.new((N_prev + bond_vec_dim, N_cur),
                name="layer " + str(layer) + " degree " + str(degree) + neighbourstr + " filter" )
        layer_bias = weights.new((1, N_cur), name="layer " + str(layer) + " biases")
        self_activations = ky.MatMult(cur_atoms,
            weights.new((N_prev, N_cur), name="layer " + str(layer) + " self filter"))
        neighbour_activations = matmult_neighbors(mol_nodes, ('atom', 'bond'),
            (cur_atoms, cur_bonds), new_weights_func, permutations)
        cur_atoms = ky.TanH(layer_bias + self_activations + neighbour_activations)

    # Include both a softened-max and a sum node to pool all atom features together.
    mol_atom_neighbors = mol_nodes['mol_atom_neighbors']
    fixed_sized_softmax = mk.NeighborSoftenedMax(mol_atom_neighbors, cur_atoms)
    fixed_sized_sum = mk.NeighborSum(mol_atom_neighbors, cur_atoms)
    fixed_sized_output = ky.Concatenate(1, fixed_sized_softmax, fixed_sized_sum)
    output_bias = weights.new((1, ), "output bias")
    output_weights = weights.new((N_cur * 2, ), "output weights")
    output = ky.MatMult(fixed_sized_output, output_weights) + output_bias
    target = ky.Blank()
    unreg_loss = ky.L2Loss(output, target)
    l2regs = [ky.L2Loss(w, ky.Parameter(np.zeros(w.shape))) * ky.Parameter(l2_penalty)
              for w in weights._weights_list ]
    loss = ky.MatAdd(unreg_loss, *l2regs)

    def make_input_dict(smiles):
        array_rep = arrayrep_from_smiles(tuple(smiles))
        return {mol_nodes[k] : v for k, v in array_rep.iteritems()}

    def grad_fun(w, smiles, t):
        input_dict = make_input_dict(smiles)
        input_dict[weights] = w
        input_dict[target] = t
        return c_grad(loss, weights, input_dict)
    def loss_fun(w, smiles, t):
        input_dict = make_input_dict(smiles)
        input_dict[weights] = w
        input_dict[target] = t
        return c_value(loss, input_dict)
    def pred_fun(w, smiles):
        input_dict = make_input_dict(smiles)
        input_dict[weights] = w
        return c_value(output, input_dict)
    def output_layer_fun(w, smiles):
        input_dict = make_input_dict(smiles)
        input_dict[weights] = w
        return c_value(fixed_sized_output, input_dict)

    return loss_fun, grad_fun, pred_fun, output_layer_fun, weights

@memoize
def arrayrep_from_smiles(smiles):
    """Precompute everything we need from MolGraph so that we can free the memory asap."""
    molgraph = graph_from_smiles_tuple(smiles)
    arrayrep = {'atom_features' : molgraph.feature_array('atom'),
                'bond_features' : molgraph.feature_array('bond'),
                'mol_atom_neighbors' : molgraph.neighbor_list('molecule', 'atom')}

    for degree in [1, 2, 3, 4]:
        arrayrep[('atom_neighbors', degree)] = \
                molgraph.neighbor_list(('atom', degree), 'atom')
        arrayrep[('bond_neighbors', degree)] = \
                molgraph.neighbor_list(('atom', degree), 'bond')

    return arrayrep
