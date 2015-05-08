import itertools as it
import autograd.numpy as np
from features import num_atom_features, num_bond_features
from util import memoize, WeightsParser
from mol_graph import graph_from_smiles_tuple
from build_vanilla_net import build_fingerprint_deep_net

from autograd.scipy.misc import logsumexp

def fast_array_from_list(xs):
    return np.concatenate([np.expand_dims(x, axis=0) for x in xs], axis=0)

def apply_and_stack(idxs, features, op):
    return fast_array_from_list([op(features[idx_list]) for idx_list in idxs])

def softened_max(X, axis=0):
    """Takes the row-wise max, but gently."""
    exp_X = np.exp(X)
    return np.sum(exp_X * X, axis) / np.sum(exp_X, axis)

def weighted_softened_max(X, axis=0):
    # X is (permutations x atoms x features).
    # The first feature softly scales how much each
    # permutation is used to weight the final activations.
    scale_feature = X[:, :, 0:1]
    # logsumexp with keepdims requires scipy >= 0.15.
    return np.sum(X * np.exp(scale_feature - logsumexp(scale_feature, axis, keepdims=True)), axis)

def summing_combine(neighbor_features, weights, degree):
    # (Atoms x neighbors x features), (features x hiddens) -> (atoms x hiddens)
    # activations = np.einsum("anf,fh->ah", stacked_neighbors, get_weights(degree))
    return np.dot(np.sum(neighbor_features, axis=1), weights)

def permute_combine(neighbor_features, weights, degree):
    weightses = [weights[n, :, :] for n in range(degree)]
    neighbors = [neighbor_features[:, d, :] for d in range(degree)]
    products = [[np.dot(n, w) for w in weightses] for n in neighbors]
    # (Atoms x neighbors1 x features), (neighbors2 x features x hiddens)
    # -> (atoms x neighbors x hiddens)
    #products = np.einsum("anf,mfh->nmah", neighbor_features, weights)
    candidates = [sum([products[i][j] for i, j in enumerate(p)])
                  for p in it.permutations(range(degree))]
    # dims candidates is (permutations x atoms x features)
    return weighted_softened_max(fast_array_from_list(candidates))

def matmult_neighbors(mol_nodes, other_ntypes, feature_sets, get_weights, combine_func):
    def neighbor_list(degree, other_ntype):
        return mol_nodes[(other_ntype + '_neighbors', degree)]
    activations_by_degree = []
    for degree in [1, 2, 3, 4]:
        # TODO: make this (plus the stacking) into an autograd primitive
        neighbor_features = [features[neighbor_list(degree, other_ntype)]
                             for other_ntype, features in zip(other_ntypes, feature_sets)]
        if any([len(feat) > 0 for feat in neighbor_features]):
            # dims of stacked_neighbors are [atoms, neighbors, features]
            stacked_neighbors = np.concatenate(neighbor_features, axis=2)
            activations = combine_func(stacked_neighbors, get_weights(degree), degree)
            activations_by_degree.append(activations)
    # This is brittle! Relies on atoms being sorted by degree in the first place,
    # in Node.graph_from_smiles_tuple()
    return np.concatenate(activations_by_degree, axis=0)

def weights_name(layer, degree, neighbor=None):
    if neighbor:
        neighborstr = " neighbor " + str(neighbor)
    else:
        neighborstr = ""
    return "layer " + str(layer) + " degree " + str(degree) + neighborstr + " filter"

def build_convnet_fingerprint_fun(bond_vec_dim=10, num_hidden_features=[20, 50, 50],
                  permutations=True, pool_funcs=['softened_max']):
    """Sets up functions to compute convnets over all molecules in a minibatch together.
       The number of hidden layers is the length of num_hidden_features - 1."""

    if permutations:
        composition_func=permute_combine
    else:
        composition_func=summing_combine

    # Specify weight shapes.
    parser = WeightsParser()
    parser.add_weights('atom2vec', (num_atom_features(), num_hidden_features[0]))
    parser.add_weights('bond2vec', (num_bond_features(), bond_vec_dim))

    in_and_out_sizes = zip(num_hidden_features[:-1], num_hidden_features[1:])
    for layer, (N_prev, N_cur) in enumerate(in_and_out_sizes):
        parser.add_weights("layer " + str(layer) + " biases", (1, N_cur))
        parser.add_weights("layer " + str(layer) + " self filter", (N_prev, N_cur))
        base_shape = (N_prev + bond_vec_dim, N_cur)
        for degree in [1, 2, 3, 4]:
            if composition_func == permute_combine:
                shape = (degree,) + base_shape
            else:
                shape = base_shape
            parser.add_weights(weights_name(layer, degree), shape)

    def hash_to_index_then_or(features):
        """The final layer of ECFP takes the integer representation at each atom and maps it
        to a '1' in the final representation.
        We can do the same thing by just mapping the least significant bits of a combination
        of all the input features.
        This function is not differentiable, which is fine."""
        combined_features = np.sum(features, axis=0)   # Do we need more than one feature?
        int_rep = np.mod(combined_features * 1000.0, num_hidden_features[-1]).astype(np.int)
        binary_features = np.zeros(features.shape[1])
        binary_features[int_rep] = 1
        return binary_features


    def output_layer_fun(weights, smiles):
        """Computes layer-wise convolution, and returns a fixed-size output."""
        mol_nodes = arrayrep_from_smiles(tuple(smiles))

        atom_features = np.dot(mol_nodes['atom_features'], parser.get(weights, 'atom2vec'))
        bond_features = np.dot(mol_nodes['bond_features'], parser.get(weights, 'bond2vec'))

        for layer in xrange(len(num_hidden_features) - 1):
            #atom_features = combine_func(atom_features, bond_features, mol_nodes)
            def get_weights_func(degree):
                return parser.get(weights, weights_name(layer, degree))
            layer_bias = parser.get(weights, "layer " + str(layer) + " biases")
            layer_self_weights = parser.get(weights, "layer " + str(layer) + " self filter")
            self_activations = np.dot(atom_features, layer_self_weights)
            neighbour_activations = matmult_neighbors(mol_nodes, ('atom', 'bond'),
                (atom_features, bond_features), get_weights_func, composition_func)
            atom_features = np.tanh(layer_bias + self_activations + neighbour_activations)

        # Pool all atom features together.
        atom_idxs = mol_nodes['atom_list']
        pooled_features = []
        if 'softened_max' in pool_funcs:
            pooled_features.append(apply_and_stack(atom_idxs, atom_features,
                                                   softened_max))
        if 'mean' in pool_funcs:
            pooled_features.append(apply_and_stack(atom_idxs, atom_features,
                                                   lambda x: np.mean(x, axis=0)))
        if 'sum' in pool_funcs:
            pooled_features.append(apply_and_stack(atom_idxs, atom_features,
                                                   lambda x: np.sum(x, axis=0)))
        if 'index' in pool_funcs:
            # Same spirit as last layer of ECFP.
            # Map each atom's features to an integer in [0, fp_length].
            pooled_features.append(apply_and_stack(atom_idxs, atom_features,
                                                   hash_to_index_then_or))
        return np.concatenate(pooled_features, axis=1)

    return output_layer_fun, parser


@memoize
def arrayrep_from_smiles(smiles):
    """Precompute everything we need from MolGraph so that we can free the memory asap."""
    molgraph = graph_from_smiles_tuple(smiles)
    arrayrep = {'atom_features' : molgraph.feature_array('atom'),
                'bond_features' : molgraph.feature_array('bond'),
                'atom_list'     : molgraph.neighbor_list('molecule', 'atom')}

    for degree in [1, 2, 3, 4]:
        # Since we know the number of neighbors, we can cast to arrays here
        # instead of using lists of lists.
        arrayrep[('atom_neighbors', degree)] = \
                np.array(molgraph.neighbor_list(('atom', degree), 'atom'), dtype=int)
        arrayrep[('bond_neighbors', degree)] = \
                np.array(molgraph.neighbor_list(('atom', degree), 'bond'), dtype=int)

    return arrayrep

def build_conv_deep_net(layer_sizes, conv_params):
    # Returns (loss_fun(fp_weights, nn_weights, smiles, targets), pred_fun, net_parser, conv_parser)
    conv_fp_func, conv_parser = build_convnet_fingerprint_fun(**conv_params)
    return build_fingerprint_deep_net(layer_sizes, conv_fp_func) + (conv_parser,)
