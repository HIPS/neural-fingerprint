import numpy as np
from util import memoize, WeightsParser
from deepmolecule import smiles_to_fps
from autograd import grad

def relu(X):
    "Rectified linear activation function."
    return X * (X > 0)

def build_vanilla_net(num_inputs, h1_size):
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
    return loss_fun, grad(loss_fun), pred_fun, hiddens, parser

def build_morgan_deep_net(fp_length=512, fp_radius=4, h1_size=500):
    """A 2-layer net whose inputs are Morgan fingerprints."""
    v_loss_fun, v_grad_fun, v_pred_fun, v_hiddens_fun, parser = \
        build_vanilla_net(num_inputs=fp_length, h1_size=h1_size)

    def fingerprints_from_smiles(smiles):
        return fingerprints_from_smiles_tuple(tuple(smiles))

    @memoize # This wrapper function exists because tuples can be hashed, but arrays can't.
    def fingerprints_from_smiles_tuple(smiles_tuple):
        return smiles_to_fps(smiles_tuple, fp_length, fp_radius)

    def grad_fun(w, s, t): return v_grad_fun(w, fingerprints_from_smiles(s), t)
    def loss_fun(w, s, t): return v_loss_fun(w, fingerprints_from_smiles(s), t)
    def pred_fun(w, s):    return v_pred_fun(w, fingerprints_from_smiles(s))
    def hiddens_fun(w, s): return v_hiddens_fun(w, fingerprints_from_smiles(s))

    return loss_fun, grad_fun, pred_fun, hiddens_fun, parser

def build_morgan_flat_net(fp_length=512, fp_radius=4):
    """Wraps functions for computing Morgan fingerprints."""
    def grad_fun(w, s, t): return 0.0
    def loss_fun(w, s, t): return 0.0
    def pred_fun(w, s):    return 0.0
    def hiddens_fun(w, s): return smiles_to_fps(s, fp_length, fp_radius)

    return loss_fun, grad_fun, pred_fun, hiddens_fun, WeightsParser()