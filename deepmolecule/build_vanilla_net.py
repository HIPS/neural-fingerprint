import autograd.numpy as np
from util import memoize, WeightsParser
from rdkit_utils import smiles_to_fps

def relu(X):
    "Rectified linear activation function."
    return X * (X > 0)

# TODO: Make this an arbitraily deep net.
def build_standard_net(num_inputs, h1_size):
    """Just a plain old 2-layer net, nothing to do with molecules."""
    parser = WeightsParser()
    parser.add_weights('layer 1 weights', (num_inputs, h1_size))
    parser.add_weights('layer 1 biases', (1, h1_size))
    parser.add_weights('layer 2 weights', h1_size)
    parser.add_weights('layer 2 bias', 1)

    # All the functions we'll need to train and predict with this net.
    def hiddens(w, X):
        hw = parser.get(w, 'layer 1 weights')
        hb = parser.get(w, 'layer 1 biases')
        return relu(np.dot(X, hw) + hb)
    def pred_fun(w, X):
        ow = parser.get(w, 'layer 2 weights')
        ob = parser.get(w, 'layer 2 bias')
        hids = hiddens(w, X)
        return np.dot(hids, ow) + ob
    def loss_fun(w, X, targets):
        preds = pred_fun(w, X)
        return np.sum((preds - targets)**2)
    return loss_fun, pred_fun, hiddens, parser

def build_fingerprint_deep_net(layer_sizes, fingerprint_func):
    """A 2-layer net whose inputs are fingerprints.
    fingerprint_func has signature (smiles, weight, params)"""
    net_loss_fun, net_pred_fun, net_hiddens_fun, net_parser = \
        build_standard_net(num_inputs=layer_sizes[0], h1_size=layer_sizes[1])

    def loss_fun(fingerprint_weights, net_weights, smiles, targets):
        return net_loss_fun(net_weights, fingerprint_func(fingerprint_weights, smiles), targets)
    def pred_fun(fingerprint_weights, net_weights, smiles):
        return net_pred_fun(net_weights, fingerprint_func(fingerprint_weights, smiles))

    return loss_fun, pred_fun, net_parser

def build_morgan_fingerprint_fun(fp_length, fp_radius):

    def fingerprints_from_smiles(weights, smiles):
        return fingerprints_from_smiles_tuple(tuple(smiles))

    @memoize # This wrapper function exists because tuples can be hashed, but arrays can't.
    def fingerprints_from_smiles_tuple(smiles_tuple):
        return smiles_to_fps(smiles_tuple, fp_length, fp_radius)

    return fingerprints_from_smiles

def build_morgan_deep_net(layer_sizes, fp_length=512, fp_radius=4):
    morgan_fp_func = build_morgan_fingerprint_fun(fp_length, fp_radius)
    return build_fingerprint_deep_net(layer_sizes, morgan_fp_func)