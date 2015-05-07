import itertools as it
import autograd.numpy as np
from features import num_atom_features, num_bond_features
from util import memoize, WeightsParser
from mol_graph import graph_from_smiles_tuple
from autograd import grad
from build_vanilla_net import build_standard_net

from autograd.scipy.misc import logsumexp

def fast_array_from_list(xs):
    return np.concatenate([np.expand_dims(x, axis=0) for x in xs], axis=0)

def apply_and_stack(idxs, features, op):
    return fast_array_from_list([op(features[idx_list, :]) for idx_list in idxs])

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

def matmult_neighbors(mol_nodes, other_ntypes, feature_sets, get_weights, permutations):
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
            if permutations:
                weightses = [get_weights(degree)[n, :, :] for n in range(degree)]
                neighbors = [stacked_neighbors[:, d, :] for d in range(degree)]
                products = [[np.dot(n, w) for w in weightses] for n in neighbors]
                # (Atoms x neighbors1 x features), (neighbors2 x features x hiddens)
                # -> (atoms x neighbors x hiddens)
                #products = np.einsum("anf,mfh->nmah", stacked_neighbors, get_weights(degree))
                candidates = [sum([products[i][j] for i, j in enumerate(p)])
                              for p in it.permutations(range(degree))]
                # dims candidates is (permutations x atoms x features)
                activations_by_degree.append(weighted_softened_max(fast_array_from_list(candidates)))
            else:
                # (Atoms x neighbors x features), (features x hiddens) -> (atoms x hiddens)
                # activations = np.einsum("anf,fh->ah", stacked_neighbors, get_weights(degree))
                activations = np.dot(np.sum(stacked_neighbors, axis=1), get_weights(degree))
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

def build_convnet(bond_vec_dim=10, num_hidden_features=[20, 50, 50],
                  permutations=False, l2_penalty=0.0, pool_funcs=['softened_max']):
    """Sets up functions to compute convnets over all molecules in a minibatch together.
       The number of hidden layers is the length of num_hidden_features - 1."""
    parser = WeightsParser()
    parser.add_weights('atom2vec', (num_atom_features(), num_hidden_features[0]))
    parser.add_weights('bond2vec', (num_bond_features(), bond_vec_dim))

    in_and_out_sizes = zip(num_hidden_features[:-1], num_hidden_features[1:])
    for layer, (N_prev, N_cur) in enumerate(in_and_out_sizes):
        parser.add_weights("layer " + str(layer) + " biases", (1, N_cur))
        parser.add_weights("layer " + str(layer) + " self filter", (N_prev, N_cur))
        base_shape = (N_prev + bond_vec_dim, N_cur)
        for degree in [1, 2, 3, 4]:
            if permutations:
                shape = (degree,) + base_shape
            else:
                shape = base_shape
            parser.add_weights(weights_name(layer, degree), shape)

    parser.add_weights('output bias', (1, ))
    parser.add_weights('output weights', (num_hidden_features[-1] * 2, ))

    def output_layer_fun(weights, smiles):
        """Computes layer-wise convolution, and returns a fixed-size output."""
        mol_nodes = arrayrep_from_smiles(tuple(smiles))

        atom_features = np.dot(mol_nodes['atom_features'], parser.get(weights, 'atom2vec'))
        bond_features = np.dot(mol_nodes['bond_features'], parser.get(weights, 'bond2vec'))

        for layer in xrange(len(num_hidden_features) - 1):
            def get_weights_func(degree):#, neighbor=None):
                return parser.get(weights, weights_name(layer, degree))#, neighbor))
            layer_bias = parser.get(weights, "layer " + str(layer) + " biases")
            layer_self_weights = parser.get(weights, "layer " + str(layer) + " self filter")
            self_activations = np.dot(atom_features, layer_self_weights)
            neighbour_activations = matmult_neighbors(mol_nodes, ('atom', 'bond'),
                (atom_features, bond_features), get_weights_func, permutations)
            atom_features = np.tanh(layer_bias + self_activations + neighbour_activations)

        # Include both a softened-max and a sum node to pool all atom features together.
        atom_neighbor_idxs = mol_nodes['mol_atom_neighbors']
        pooled_features = []
        if 'softened_max' in pool_funcs:
            pooled_features.append(apply_and_stack(atom_neighbor_idxs, atom_features,
                                                   softened_max))
        if 'mean' in pool_funcs:
            pooled_features.append(apply_and_stack(atom_neighbor_idxs, atom_features,
                                                   lambda x: np.mean(x, axis=0)))
        if 'sum' in pool_funcs:
            pooled_features.append(apply_and_stack(atom_neighbor_idxs, atom_features,
                                                   lambda x: np.sum(x, axis=0)))
        return np.concatenate(pooled_features, axis=1)

    def prediction_fun(weights, smiles):
        output_weights = parser.get(weights, 'output weights')
        output_bias = parser.get(weights, 'output bias')
        hidden_units = output_layer_fun(weights, smiles)
        return np.dot(hidden_units, output_weights) + output_bias

    def loss_fun(weights, smiles, targets):
        preds = prediction_fun(weights, smiles)
        acc_loss = np.sum((preds - targets)**2)
        l2reg = l2_penalty * np.sum(weights**2)
        return acc_loss + l2reg

    return loss_fun, grad(loss_fun), prediction_fun, output_layer_fun, parser

@memoize
def arrayrep_from_smiles(smiles):
    """Precompute everything we need from MolGraph so that we can free the memory asap."""
    molgraph = graph_from_smiles_tuple(smiles)
    arrayrep = {'atom_features' : molgraph.feature_array('atom'),
                'bond_features' : molgraph.feature_array('bond'),
                'mol_atom_neighbors' : molgraph.neighbor_list('molecule', 'atom')}

    for degree in [1, 2, 3, 4]:
        # Since we know the number of neighbors, we can cast to arrays here
        # instead of using lists of lists.
        arrayrep[('atom_neighbors', degree)] = \
                np.array(molgraph.neighbor_list(('atom', degree), 'atom'), dtype=int)
        arrayrep[('bond_neighbors', degree)] = \
                np.array(molgraph.neighbor_list(('atom', degree), 'bond'), dtype=int)

    return arrayrep


def build_convnet_with_vanilla_ontop(bond_vec_dim=1, num_hidden_features=[20, 50, 50],
                        permutations=False, l2_penalty=0.0, vanilla_hidden=100):
    """A network with a convnet on the bottom, and vanilla fully-connected net on top."""

    _, _, _, conv_hiddens_fun, conv_parser = \
        build_convnet(bond_vec_dim, num_hidden_features, permutations)

    v_loss_fun, _, v_pred_fun, v_hiddens_fun, v_parser = \
        build_standard_net(num_inputs=num_hidden_features[-1], h1_size=vanilla_hidden)

    parser = WeightsParser()
    parser.add_weights('convnet weights', len(conv_parser))
    parser.add_weights('vanilla weights', len(v_parser))

    def hiddens_fun(weights, smiles):
        convnet_weights = parser.get(weights, 'convnet weights')
        vanilla_weights = parser.get(weights, 'vanilla weights')
        conv_output = conv_hiddens_fun(convnet_weights, smiles)
        return v_hiddens_fun(vanilla_weights, conv_output)

    def pred_fun(weights, smiles):
        convnet_weights = parser.get(weights, 'convnet weights')
        vanilla_weights = parser.get(weights, 'vanilla weights')
        conv_output = conv_hiddens_fun(convnet_weights, smiles)
        return v_pred_fun(vanilla_weights, conv_output)

    def loss_fun(weights, smiles, targets):
        preds = pred_fun(weights, smiles)
        acc_loss = np.sum((preds - targets)**2)
        l2reg = l2_penalty * np.sum(np.sum(weights**2))
        return acc_loss + l2reg

    return loss_fun, grad(loss_fun), pred_fun, hiddens_fun, parser
