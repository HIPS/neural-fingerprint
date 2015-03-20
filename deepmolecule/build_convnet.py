import itertools as it
import kayak as ky
import kayak_ops as mk
import numpy as np
from features import N_atom_features, N_bond_features
from util import WeightsContainer, c_value, c_grad, memoize, WeightsParser
from mol_graph import graph_from_smiles_tuple
from autograd import grad

def all_permutations(N):
    return [permutation for permutation in it.permutations(range(N))]

def neighbor_stack(idxs, features, num_neighbors):
    """dims of result are (atoms, neighbors, features)"""
    #result_rows = np.zeros((len(idxs), num_neighbors, features.shape[1]))
    #for i, idx_list in enumerate(idxs):
    #    result_rows[i, :, :] = features[idx_list, :]
    #return result_rows
    results = [np.expand_dims(features[idx_list, :], axis=0) for idx_list in idxs]
    return np.concatenate(results, axis=0)

def neighbor_cat(idxs, features):
    result_rows = []
    for idx_list in idxs:
        result_rows.append(np.concatenate(
            [features[i, :] for i in idx_list]))
    return np.array(result_rows)

def neighbor_softened_max(idxs, features):

    def neighbour_softened_max(X):
        exp_X = np.exp(X)
        return np.sum(exp_X * X, axis=0) / np.sum(exp_X, axis=0)

    result_rows = []
    for idx_list in idxs:
        result_rows.append(np.expand_dims(neighbour_softened_max(features[idx_list, :]), axis=0))
    return np.concatenate(result_rows, axis=0)

def neighbor_sum(idxs, features):
    result_rows = []
    for idx_list in idxs:
        result_rows.append(np.expand_dims(np.sum(features[idx_list, :], axis=0), axis=0))
    return np.concatenate(result_rows, axis=0)

def safe_tensordot(A, B, axes):
    """Allows dimensions of length zero"""
    Adims, Bdims = list(A.shape), list(B.shape)
    if np.any([d is 0 for d in Adims + Bdims]):
        Anewdims = [d for i, d in enumerate(Adims) if i not in axes[0]]
        Bnewdims = [d for i, d in enumerate(Bdims) if i not in axes[1]]
        return np.zeros(Anewdims + Bnewdims)
    else:
        return np.tensordot(A, B, axes)

def logsumexp(X, axis=None):
    maxes = np.max(X, axis=axis, keepdims=True)
    return np.log(np.sum(np.exp(X - maxes), axis=axis, keepdims=True)) + maxes

#def logsumexp(X, axis):
#    max_X = np.max(X)
#    return max_X + np.log(np.sum(np.exp(X - max_X), axis=axis, keepdims=True))

def softened_max_ag(Xlist):
    X = np.concatenate([p[None, :] for p in Xlist], axis=0)
    # X is now permutations x atoms x features.
    # The first feature softly scales how much each
    # permutation is used to weight the final activations.
    scale_feature = X[:, :, 0:1]
    return np.sum(X * np.exp(scale_feature - logsumexp(scale_feature, axis=0)), axis=0)

def softened_max(X_list):
    X_cat = ky.ListToArray(*X_list)
    X_feature_0 = ky.Take(X_cat, slice(0,1), axis=2)
    return ky.MatSum(ky.MatElemMult(X_cat, ky.SoftMax(X_feature_0, axis=0)),
                     axis=0, keepdims=False)

def matmult_neighbors_ag(mol_nodes, other_ntypes, feature_sets, get_weights_fun, permutations=False):
    def neighbor_list(degree, other_ntype):
        return mol_nodes[(other_ntype + '_neighbors', degree)]
    result_by_degree = []
    for degree in [1, 2, 3, 4]:
        # dims of stacked_neighbors are [atoms, neighbors (as in atom-bond pairs), features]
        stacked_neighbors = np.concatenate(
            [neighbor_stack(neighbor_list(degree, other_ntype), features, degree)
              for other_ntype, features in zip(other_ntypes, feature_sets)], axis=2)
        if permutations:
            weightses = [get_weights_fun(degree, " neighbour " + str(n)) for n in range(degree)]
            neighbors = [stacked_neighbors[:, d, :] for d in range(degree)]
            products = [[np.dot(n, w) for w in weightses] for n in neighbors]
            candidates = [sum([products[i][j] for i, j in enumerate(p)])
                          for p in all_permutations(degree)]
            # dims of each candidate are (atoms, features)
            result_by_degree.append(softened_max_ag(candidates))
        else:
            result_by_degree.append(np.sum(
                safe_tensordot(stacked_neighbors, get_weights_fun(degree), axes=((2,), (0,))),
                axis=1, keepdims=False))

    # This is brittle! Relies on atoms being sorted by degree in the first place,
    # in Node.graph_from_smiles_tuple()
    return np.concatenate(result_by_degree, axis=0)


def build_universal_net_ag(bond_vec_dim=1, num_hidden_features=[20, 50, 50],
                        permutations=False, l2_penalty=0.0):
    """Sets up functions to compute convnets over all molecules in a minibatch together.
       The number of hidden layers is the length of num_hidden_features - 1."""

    parser = WeightsParser()
    parser.add_weights('atom2vec', (N_atom_features, num_hidden_features[0]))
    parser.add_weights('bond2vec', (N_bond_features, bond_vec_dim))

    in_and_out_sizes = zip(num_hidden_features[:-1], num_hidden_features[1:])
    for layer, (N_prev, N_cur) in enumerate(in_and_out_sizes):
        parser.add_weights("layer " + str(layer) + " biases", (1, N_cur))
        parser.add_weights("layer " + str(layer) + " self filter", (N_prev, N_cur))

        def new_weights_func(degree, neighbourstr=""):
            name = "layer " + str(layer) + " degree " + str(degree) + neighbourstr + " filter"
            parser.add_weights(name, (N_prev + bond_vec_dim, N_cur))
        for degree in [1, 2, 3, 4]:
            if permutations:
                for n in range(degree):
                    new_weights_func(degree, " neighbour " + str(n))
            else:
                new_weights_func(degree)

    parser.add_weights('output bias', (1, ))
    parser.add_weights('output weights', (num_hidden_features[-1] * 2, ))


    def output_layer_fun(weights, smiles):
        """Computes layer-wise convolution, and returns a fixed-size output."""
        mol_nodes = arrayrep_from_smiles(tuple(smiles))

        cur_atoms = np.dot(mol_nodes['atom_features'], parser.get(weights, 'atom2vec'))
        cur_bonds = np.dot(mol_nodes['bond_features'], parser.get(weights, 'bond2vec'))

        for layer in xrange(len(num_hidden_features) - 1):
            def get_weights_func(degree, neighbourstr=""):
                return parser.get(weights, "layer " + str(layer) + " degree "
                                  + str(degree) + neighbourstr + " filter" )
            layer_bias = parser.get(weights, "layer " + str(layer) + " biases")
            layer_self_weights = parser.get(weights, "layer " + str(layer) + " self filter")
            self_activations = np.dot(cur_atoms, layer_self_weights)
            neighbour_activations = matmult_neighbors_ag(mol_nodes, ('atom', 'bond'),
                (cur_atoms, cur_bonds), get_weights_func, permutations)
            cur_atoms = np.tanh(layer_bias + self_activations + neighbour_activations)

        # Include both a softened-max and a sum node to pool all atom features together.
        mol_atom_neighbors = mol_nodes['mol_atom_neighbors']
        fixed_sized_softmax = neighbor_softened_max(mol_atom_neighbors, cur_atoms)
        fixed_sized_sum = neighbor_sum(mol_atom_neighbors, cur_atoms)
        return np.concatenate((fixed_sized_softmax, fixed_sized_sum), axis=1)

    def pred_fun(weights, smiles):
        output_weights = parser.get(weights, 'output weights')
        output_bias = parser.get(weights, 'output bias')
        hiden_units = output_layer_fun(weights, smiles)
        return np.dot(hiden_units, output_weights) + output_bias

    def loss_fun(weights, smiles, targets):
        preds = pred_fun(weights, smiles)
        acc_loss = np.sum((preds - targets)**2)
        l2reg = l2_penalty * np.sum(np.sum(weights**2))
        return acc_loss + l2reg

    return loss_fun, grad(loss_fun), pred_fun, output_layer_fun, parser



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
