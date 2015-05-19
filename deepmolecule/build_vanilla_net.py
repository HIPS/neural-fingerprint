import autograd.numpy as np
from util import memoize, WeightsParser
from rdkit_utils import smiles_to_fps

def relu(X):
    "Rectified linear activation function."
    return X * (X > 0)

def build_standard_net(layer_sizes, L2_reg=0.0, activation_function=np.tanh):
    """Just a plain old neural net, nothing to do with molecules.
    layer sizes includes the input size."""
    layer_sizes = layer_sizes + [1]

    parser = WeightsParser()
    for i, shape in enumerate(zip(layer_sizes[:-1], layer_sizes[1:])):
        parser.add_weights(('weights', i), shape)
        parser.add_weights(('biases', i), (1, shape[1]))

    def predictions(W_vect, X):
        cur_units = X
        for layer in range(len(layer_sizes) - 1):
            cur_W = parser.get(W_vect, ('weights', layer))
            cur_B = parser.get(W_vect, ('biases', layer))
            cur_units = np.dot(cur_units, cur_W) + cur_B
            if layer < len(layer_sizes) - 2:
                cur_units = activation_function(cur_units)
        return cur_units[:, 0]

    def loss(w, X, targets):
        log_prior = -L2_reg * np.dot(w, w)
        preds = predictions(w, X)
        return np.sum((preds - targets)**2, axis=0) - log_prior

    return loss, predictions, parser


def build_fingerprint_deep_net(layer_sizes, fingerprint_func, fp_parser):
    """A 2-layer net whose inputs are fingerprints.
    fingerprint_func has signature (smiles, weights, params)"""
    net_loss_fun, net_pred_fun, net_parser = build_standard_net(layer_sizes)

    combined_parser = WeightsParser()
    combined_parser.add_weights('fingerprint weights', (len(fp_parser),))
    combined_parser.add_weights('net weights', (len(net_parser),))

    def unpack_weights(weights):
        fingerprint_weights = combined_parser.get(weights, 'fingerprint weights')
        net_weights         = combined_parser.get(weights, 'net weights')
        return fingerprint_weights, net_weights

    def loss_fun(weights, smiles, targets):
        fingerprint_weights, net_weights = unpack_weights(weights)
        fingerprints = fingerprint_func(fingerprint_weights, smiles)
        return net_loss_fun(net_weights, fingerprints, targets)
    def pred_fun(weights, smiles):
        fingerprint_weights, net_weights = unpack_weights(weights)
        fingerprints = fingerprint_func(fingerprint_weights, smiles)
        return net_pred_fun(net_weights, fingerprints)

    return loss_fun, pred_fun, combined_parser


def build_morgan_fingerprint_fun(fp_length, fp_radius):

    def fingerprints_from_smiles(weights, smiles):
        # Morgan fingerprints don't use weights.
        return fingerprints_from_smiles_tuple(tuple(smiles))

    @memoize # This wrapper function exists because tuples can be hashed, but arrays can't.
    def fingerprints_from_smiles_tuple(smiles_tuple):
        return smiles_to_fps(smiles_tuple, fp_length, fp_radius)

    return fingerprints_from_smiles

def build_morgan_deep_net(layer_sizes, fp_length=512, fp_radius=4):
    empty_parser = WeightsParser()
    morgan_fp_func = build_morgan_fingerprint_fun(fp_length, fp_radius)
    return build_fingerprint_deep_net(layer_sizes, morgan_fp_func, empty_parser)
