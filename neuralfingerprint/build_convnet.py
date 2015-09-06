import autograd.numpy as np
from autograd.scipy.misc import logsumexp

from features import num_atom_features, num_bond_features
from util import memoize, WeightsParser
from mol_graph import graph_from_smiles_tuple, degrees
from build_vanilla_net import build_fingerprint_deep_net, relu, batch_normalize


def fast_array_from_list(xs):
    return np.concatenate([np.expand_dims(x, axis=0) for x in xs], axis=0)

def sum_and_stack(features, idxs_list_of_lists):
    return fast_array_from_list([np.sum(features[idx_list], axis=0) for idx_list in idxs_list_of_lists])

def softmax(X, axis=0):
    return np.exp(X - logsumexp(X, axis=axis, keepdims=True))

def matmult_neighbors(array_rep, atom_features, bond_features, get_weights):
    activations_by_degree = []
    for degree in degrees:
        atom_neighbors_list = array_rep[('atom_neighbors', degree)]
        bond_neighbors_list = array_rep[('bond_neighbors', degree)]
        if len(atom_neighbors_list) > 0:
            neighbor_features = [atom_features[atom_neighbors_list],
                                 bond_features[bond_neighbors_list]]
            # dims of stacked_neighbors are [atoms, neighbors, atom and bond features]
            stacked_neighbors = np.concatenate(neighbor_features, axis=2)
            summed_neighbors = np.sum(stacked_neighbors, axis=1)
            activations = np.dot(summed_neighbors, get_weights(degree))
            activations_by_degree.append(activations)
    # This operation relies on atoms being sorted by degree,
    # in Node.graph_from_smiles_tuple()
    return np.concatenate(activations_by_degree, axis=0)

def weights_name(layer, degree):
    return "layer " + str(layer) + " degree " + str(degree) + " filter"

def build_convnet_fingerprint_fun(num_hidden_features=[100, 100], fp_length=512,
                                  normalize=True, activation_function=relu,
                                  return_atom_activations=False):
    """Sets up functions to compute convnets over all molecules in a minibatch together."""

    # Specify weight shapes.
    parser = WeightsParser()
    all_layer_sizes = [num_atom_features()] + num_hidden_features
    for layer in range(len(all_layer_sizes)):
        parser.add_weights(('layer output weights', layer), (all_layer_sizes[layer], fp_length))
        parser.add_weights(('layer output bias', layer),    (1, fp_length))

    in_and_out_sizes = zip(all_layer_sizes[:-1], all_layer_sizes[1:])
    for layer, (N_prev, N_cur) in enumerate(in_and_out_sizes):
        parser.add_weights(("layer", layer, "biases"), (1, N_cur))
        parser.add_weights(("layer", layer, "self filter"), (N_prev, N_cur))
        for degree in degrees:
            parser.add_weights(weights_name(layer, degree), (N_prev + num_bond_features(), N_cur))

    def update_layer(weights, layer, atom_features, bond_features, array_rep, normalize=False):
        def get_weights_func(degree):
            return parser.get(weights, weights_name(layer, degree))
        layer_bias         = parser.get(weights, ("layer", layer, "biases"))
        layer_self_weights = parser.get(weights, ("layer", layer, "self filter"))
        self_activations = np.dot(atom_features, layer_self_weights)
        neighbour_activations = matmult_neighbors(
            array_rep, atom_features, bond_features, get_weights_func)

        total_activations = neighbour_activations + self_activations + layer_bias
        if normalize:
            total_activations = batch_normalize(total_activations)
        return activation_function(total_activations)

    def output_layer_fun_and_atom_activations(weights, smiles):
        """Computes layer-wise convolution, and returns a fixed-size output."""

        array_rep = array_rep_from_smiles(tuple(smiles))
        atom_features = array_rep['atom_features']
        bond_features = array_rep['bond_features']

        all_layer_fps = []
        atom_activations = []
        def write_to_fingerprint(atom_features, layer):
            cur_out_weights = parser.get(weights, ('layer output weights', layer))
            cur_out_bias    = parser.get(weights, ('layer output bias', layer))
            atom_outputs = softmax(cur_out_bias + np.dot(atom_features, cur_out_weights), axis=1)
            atom_activations.append(atom_outputs)
            # Sum over all atoms within a moleclue:
            layer_output = sum_and_stack(atom_outputs, array_rep['atom_list'])
            all_layer_fps.append(layer_output)

        num_layers = len(num_hidden_features)
        for layer in xrange(num_layers):
            write_to_fingerprint(atom_features, layer)
            atom_features = update_layer(weights, layer, atom_features, bond_features, array_rep,
                                         normalize=normalize)
        write_to_fingerprint(atom_features, num_layers)
        return sum(all_layer_fps), atom_activations, array_rep

    def output_layer_fun(weights, smiles):
        output, _, _ = output_layer_fun_and_atom_activations(weights, smiles)
        return output

    def compute_atom_activations(weights, smiles):
        _, atom_activations, array_rep = output_layer_fun_and_atom_activations(weights, smiles)
        return atom_activations, array_rep

    if return_atom_activations:
        return output_layer_fun, parser, compute_atom_activations
    else:
        return output_layer_fun, parser

@memoize
def array_rep_from_smiles(smiles):
    """Precompute everything we need from MolGraph so that we can free the memory asap."""
    molgraph = graph_from_smiles_tuple(smiles)
    arrayrep = {'atom_features' : molgraph.feature_array('atom'),
                'bond_features' : molgraph.feature_array('bond'),
                'atom_list'     : molgraph.neighbor_list('molecule', 'atom'), # List of lists.
                'rdkit_ix'      : molgraph.rdkit_ix_array()}  # For plotting only.
    for degree in degrees:
        arrayrep[('atom_neighbors', degree)] = \
            np.array(molgraph.neighbor_list(('atom', degree), 'atom'), dtype=int)
        arrayrep[('bond_neighbors', degree)] = \
            np.array(molgraph.neighbor_list(('atom', degree), 'bond'), dtype=int)
    return arrayrep

def build_conv_deep_net(conv_params, net_params, fp_l2_penalty=0.0):
    """Returns loss_fun(all_weights, smiles, targets), pred_fun, combined_parser."""
    conv_fp_func, conv_parser = build_convnet_fingerprint_fun(**conv_params)
    return build_fingerprint_deep_net(net_params, conv_fp_func, conv_parser, fp_l2_penalty)

