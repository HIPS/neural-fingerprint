import itertools as it
import autograd.numpy as np
from features import N_atom_features, N_bond_features
from util import memoize, WeightsParser, safe_tensordot
from mol_graph import graph_from_smiles_tuple
from autograd import grad
from build_vanilla_net import build_vanilla_net

from scipy.misc import logsumexp

def apply_and_stack(idxs, features, op=lambda x: x):
    result_rows = []
    for idx_list in idxs:
        result_rows.append(np.expand_dims(op(features[idx_list, :]), axis=0))
    return np.concatenate(result_rows, axis=0)

def softened_max(X):
    """Takes the row-wise max, but gently."""
    exp_X = np.exp(X)
    return np.sum(exp_X * X, axis=0) / np.sum(exp_X, axis=0)

def weighted_softened_max(Xlist):
    X = np.concatenate([p[None, :] for p in Xlist], axis=0)
    # X is now permutations x atoms x features.
    # The first feature softly scales how much each
    # permutation is used to weight the final activations.
    scale_feature = X[:, :, 0:1]
    # The next line needs at least scipy 0.15, when keepdims was introduced to logsumexp.
    return np.sum(X * np.exp(scale_feature - logsumexp(scale_feature, axis=0, keepdims=True)), axis=0)

def matmult_neighbors(mol_nodes, other_ntypes, feature_sets, get_weights_fun, permutations=False):
    def neighbor_list(degree, other_ntype):
        return mol_nodes[(other_ntype + '_neighbors', degree)]
    result_by_degree = []
    for degree in [1, 2, 3, 4]:
        # dims of stacked_neighbors are [atoms, neighbors (as in atom-bond pairs), features]
        stacked_neighbors = np.concatenate(
            [apply_and_stack(neighbor_list(degree, other_ntype), features)
              for other_ntype, features in zip(other_ntypes, feature_sets)], axis=2)
        if permutations:
            weightses = [get_weights_fun(degree, " neighbour " + str(n)) for n in range(degree)]
            neighbors = [stacked_neighbors[:, d, :] for d in range(degree)]
            products = [[np.dot(n, w) for w in weightses] for n in neighbors]
            candidates = [sum([products[i][j] for i, j in enumerate(p)])
                          for p in it.permutations(range(degree))]
            # dims of each candidate are (atoms, features)
            result_by_degree.append(weighted_softened_max(candidates))
        else:
            result_by_degree.append(np.sum(
                safe_tensordot(stacked_neighbors, get_weights_fun(degree), axes=((2,), (0,))),
                axis=1, keepdims=False))

    # This is brittle! Relies on atoms being sorted by degree in the first place,
    # in Node.graph_from_smiles_tuple()
    return np.concatenate(result_by_degree, axis=0)


def build_convnet(bond_vec_dim=1, num_hidden_features=[20, 50, 50],
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
            neighbour_activations = matmult_neighbors(mol_nodes, ('atom', 'bond'),
                (cur_atoms, cur_bonds), get_weights_func, permutations)
            cur_atoms = np.tanh(layer_bias + self_activations + neighbour_activations)

        # Include both a softened-max and a sum node to pool all atom features together.
        mol_atom_neighbors = mol_nodes['mol_atom_neighbors']
        fixed_sized_softmax = apply_and_stack(mol_atom_neighbors, cur_atoms, softened_max)
        fixed_sized_sum = apply_and_stack(mol_atom_neighbors, cur_atoms, lambda x: np.sum(x, axis=0))
        return np.concatenate((fixed_sized_softmax, fixed_sized_sum), axis=1)

    def prediction_fun(weights, smiles):
        output_weights = parser.get(weights, 'output weights')
        output_bias = parser.get(weights, 'output bias')
        hiden_units = output_layer_fun(weights, smiles)
        return np.dot(hiden_units, output_weights) + output_bias

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
        arrayrep[('atom_neighbors', degree)] = \
                molgraph.neighbor_list(('atom', degree), 'atom')
        arrayrep[('bond_neighbors', degree)] = \
                molgraph.neighbor_list(('atom', degree), 'bond')

    return arrayrep


def build_convnet_with_vanilla_ontop(bond_vec_dim=1, num_hidden_features=[20, 50, 50],
                        permutations=False, l2_penalty=0.0, vanilla_hidden=100):
    """A network with a convnet on the bottom, and vanilla fully-connected net on top."""

    _, _, _, conv_hiddens_fun, conv_parser = \
        build_convnet(bond_vec_dim, num_hidden_features, permutations)

    v_loss_fun, _, v_pred_fun, v_hiddens_fun, v_parser = \
        build_vanilla_net(num_inputs=num_hidden_features[-1]*2, h1_size=vanilla_hidden)

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