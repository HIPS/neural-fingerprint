from copy import copy

import autograd.numpy as np
import autograd.numpy.random as npr
from features import num_atom_features, num_bond_features
from util import memoize, WeightsParser
from mol_graph import graph_from_smiles_tuple
from build_vanilla_net import build_fingerprint_deep_net

def fast_array_from_list(xs):
    return np.concatenate([np.expand_dims(x, axis=0) for x in xs], axis=0)

def apply_and_stack(idxs, features, op):
    return fast_array_from_list([op(features[idx_list]) for idx_list in idxs])

def softened_max(X, axis=0):
    """Takes the row-wise max, but gently."""
    exp_X = np.exp(X)
    return np.sum(exp_X * X, axis) / np.sum(exp_X, axis)

def softmax(X, axis=0):
    exp_X = np.exp(X)
    return exp_X / np.sum(exp_X, axis=axis, keepdims=True)

def matmult_neighbors(array_rep, other_ntypes, feature_sets, get_weights):
    def neighbor_list(degree, other_ntype):
        return array_rep[(other_ntype + '_neighbors', degree)]
    activations_by_degree = []
    for degree in [1, 2, 3, 4]:
        # TODO: make this (plus the stacking) into an autograd primitive
        neighbor_features = [features[neighbor_list(degree, other_ntype)]
                             for other_ntype, features in zip(other_ntypes, feature_sets)]
        if any([len(feat) > 0 for feat in neighbor_features]):
            # dims of stacked_neighbors are [atoms, neighbors, features]
            stacked_neighbors = np.concatenate(neighbor_features, axis=2)
            activations = np.tensordot(stacked_neighbors, get_weights(degree), axes=[(1,2),(0,1)])
            activations_by_degree.append(activations)
    # This is brittle! Relies on atoms being sorted by degree in the first place,
    # in Node.graph_from_smiles_tuple()
    return np.concatenate(activations_by_degree, axis=0)

def sigmoid(x):
    return 0.5 + 0.5 * np.tanh(x)

def weights_name(layer, degree):
    return "layer " + str(layer) + " degree " + str(degree) + " filter"

def build_convnet_fingerprint_fun(atom_vec_dim=20, bond_vec_dim=10,
                                  num_hidden_features=[100, 100], fp_length=512):
    """Sets up functions to compute convnets over all molecules in a minibatch together.
       The number of hidden layers is the length of num_hidden_features - 1."""

    # Specify weight shapes.
    parser = WeightsParser()
    parser.add_weights('atom2vec', (num_atom_features(), atom_vec_dim))
    parser.add_weights('bond2vec', (num_bond_features(), bond_vec_dim))
    all_layer_sizes = [atom_vec_dim] + num_hidden_features
    in_and_out_sizes = zip(all_layer_sizes[:-1], all_layer_sizes[1:])
    for layer, (N_prev, N_cur) in enumerate(in_and_out_sizes):
        parser.add_weights("layer " + str(layer) + " biases", (1, N_cur))
        parser.add_weights("layer " + str(layer) + " self filter", (N_prev, N_cur))
        base_shape = (N_prev + bond_vec_dim, N_cur)
        for degree in [1, 2, 3, 4]:
            parser.add_weights(weights_name(layer, degree), (degree,) + base_shape)

    parser.add_weights('final layer weights', (all_layer_sizes[-1], fp_length))
    parser.add_weights('final layer bias', (1,fp_length))

    def update_layer(weights, layer, atom_features, bond_features, array_rep, normalize=False):
        def get_weights_func(degree):
            return parser.get(weights, weights_name(layer, degree))
        layer_bias = parser.get(weights, "layer " + str(layer) + " biases")
        layer_self_weights = parser.get(weights, "layer " + str(layer) + " self filter")
        self_activations = np.dot(atom_features, layer_self_weights)
        neighbour_activations = matmult_neighbors(array_rep, ('atom', 'bond'),
            (atom_features, bond_features), get_weights_func)

        total_activations = neighbour_activations + self_activations
        if normalize:
            total_activations = batch_normalize(total_activations)

        return np.tanh(total_activations + layer_bias)

    def last_layer_features(weights, array_rep):
        atom_features = np.dot(array_rep['atom_features'], parser.get(weights, 'atom2vec'))
        bond_features = np.dot(array_rep['bond_features'], parser.get(weights, 'bond2vec'))

        for layer in xrange(len(num_hidden_features)):
            atom_features = update_layer(weights, layer, atom_features, bond_features, array_rep,
                                         normalize=True)
        return atom_features

    # Generate random weights for sorting.
    rs = npr.RandomState(0)
    sorting_weights = 0.01 * rs.randn(len(parser))

    def canonicalizer(array_rep):
        # Sorts lists of atoms into a canonical ordering.
        atom_features = np.dot(array_rep['atom_features'], parser.get(sorting_weights, 'atom2vec'))
        bond_features = np.dot(array_rep['bond_features'], parser.get(sorting_weights, 'bond2vec'))
        for layer in xrange(len(num_hidden_features)):
            array_rep = sort_array_rep(array_rep, atom_features, bond_features)
            atom_features = update_layer(sorting_weights, layer,
                                         atom_features, bond_features, array_rep,
                                         normalize=False)
        array_rep = sort_array_rep(array_rep, atom_features, bond_features)
        return array_rep

    def sort_array_rep(array_rep, atom_features, bond_features):
        def sort_atoms(atom_idxs, bond_idxs):
            sort_perm = np.argsort(atom_features[atom_idxs, 0] + bond_features[bond_idxs, 0])
            return atom_idxs[sort_perm], bond_idxs[sort_perm]

        sorted_array_rep = copy(array_rep)
        for degree in [2, 3, 4]:
            sorted_array_rep[('atom_neighbors', degree)] = \
                np.zeros(array_rep[('atom_neighbors', degree)].shape, dtype=int)
            sorted_array_rep[('bond_neighbors', degree)] = \
                np.zeros(array_rep[('atom_neighbors', degree)].shape, dtype=int)
            for i, (atom_idxs, bond_idxs) in enumerate(zip(array_rep[('atom_neighbors', degree)],
                                                           array_rep[('bond_neighbors', degree)])):
                sorted_atoms, sorted_bonds = sort_atoms(atom_idxs, bond_idxs)
                sorted_array_rep[('atom_neighbors', degree)][i, :] = sorted_atoms
                sorted_array_rep[('bond_neighbors', degree)][i, :] = sorted_bonds

        return sorted_array_rep

    def output_layer_fun(weights, smiles):
        """Computes layer-wise convolution, and returns a fixed-size output."""
        array_rep = array_rep_from_smiles(tuple(smiles))
        atom_features = last_layer_features(weights, array_rep)

        # Expand hidden layer to the final fingerprint size.
        final_weights = parser.get(weights, 'final layer weights')
        final_bias    = parser.get(weights, 'final layer bias')

        # One-hot coding
        final_activations = softmax(final_bias + np.dot(atom_features, final_weights), axis=1)
        atom_idxs = array_rep['atom_list']
        molecule_activations = apply_and_stack(atom_idxs, final_activations,
                                               lambda x : np.sum(x, axis=0))
        return np.tanh(molecule_activations)

    @memoize
    def array_rep_from_smiles(smiles):
        """Precompute everything we need from MolGraph so that we can free the memory asap."""
        molgraph = graph_from_smiles_tuple(smiles)
        arrayrep = {'atom_features' : molgraph.feature_array('atom'),
                    'bond_features' : molgraph.feature_array('bond'),
                    'atom_list'     : molgraph.neighbor_list('molecule', 'atom')}  # List of lists.

        for degree in [1, 2, 3, 4]:
            # Since we know the number of neighbors, we can cast to arrays here
            # instead of using lists of lists.
            arrayrep[('atom_neighbors', degree)] = \
                    np.array(molgraph.neighbor_list(('atom', degree), 'atom'), dtype=int)
            arrayrep[('bond_neighbors', degree)] = \
                    np.array(molgraph.neighbor_list(('atom', degree), 'bond'), dtype=int)

        return canonicalizer(arrayrep)

    return output_layer_fun, parser

def batch_normalize(activations):
    return activations / (0.5 * np.std(activations, axis=0, keepdims=True))

def build_conv_deep_net(layer_sizes, conv_params):
    # Returns (loss_fun(fp_weights, nn_weights, smiles, targets), pred_fun, net_parser, conv_parser)
    conv_fp_func, conv_parser = build_convnet_fingerprint_fun(**conv_params)
    return build_fingerprint_deep_net(layer_sizes, conv_fp_func) + (conv_parser,)
