import kayak as ky
import numpy as np
import numpy.random as npr
from util import WeightsContainer, c_value, c_grad, memoize
from deepmolecule import smiles_to_fps

def build_vanilla_net(num_inputs, h1_size, h1_dropout):
    """Just a plain old 2-layer net, nothing to do with molecules."""
    weights = WeightsContainer()
    # Need to give fake data here so that the dropout node can make a mask.
    inputs =  ky.Inputs(np.zeros((1, num_inputs)))
    W1 = weights.new((num_inputs, h1_size), name='layer 1 weights')
    B1 = weights.new((1, h1_size), name='layer 1 biases')
    hidden = ky.HardReLU(ky.MatMult(inputs, W1) + B1)
    dropout = ky.Dropout(hidden, drop_prob=h1_dropout, rng=npr.RandomState(1))
    W2 = weights.new(h1_size, name='layer 2 weights')
    B2 = weights.new(1, name='layer 2 bias')
    output =  ky.MatMult(dropout, W2) + B2
    target =  ky.Blank()
    loss =  ky.L2Loss(output, target)

    # All the functions we'll need to train and predict with this net.
    def grad_fun(w, X, t):
        """X is a matrix of size num_training_examples by num_features."""
        inputs.value = X      # Necessary so that the dropout mask will be the right size.
        dropout.draw_new_mask()
        return c_grad(loss, weights, {weights : w, target : t})
    def loss_fun(w, X, t):
        return c_value(loss, {weights : w, inputs : X, target : t})
    def pred_fun(w, X):
        inputs.value = X      # Necessary so that the dropout mask will be the right size.
        dropout.reinstate_units()
        return c_value(output, {weights : w, inputs : X})
    def hidden_layer_fun(w, X):
        inputs.value = X      # Necessary so that the dropout mask will be the right size.
        return c_value(hidden, {weights : w, inputs : X})

    return loss_fun, grad_fun, pred_fun, hidden_layer_fun, weights


def build_morgan_deep_net(fp_length=512, fp_radius=4, h1_size=500, h1_dropout=0.1):
    """A 2-layer net whose inputs are Morgan fingerprints."""
    v_loss_fun, v_grad_fun, v_pred_fun, v_hiddens_fun, weights = \
        build_vanilla_net(num_inputs=fp_length, h1_size=h1_size, h1_dropout=h1_dropout)

    def features_from_smiles(smiles):
        return features_from_smiles_tuple(tuple(smiles))

    @memoize # This wrapper function exists because tuples can be hashed, but arrays can't.
    def features_from_smiles_tuple(smiles_tuple):
        return smiles_to_fps(smiles_tuple, fp_length, fp_radius)

    grad_fun = lambda w, s, t: v_grad_fun(w, features_from_smiles(s), t)
    loss_fun = lambda w, s, t: v_loss_fun(w, features_from_smiles(s), t)
    pred_fun = lambda w, s:    v_pred_fun(w, features_from_smiles(s))
    hiddens_fun = lambda w, s: v_hiddens_fun(w, features_from_smiles(s))

    return loss_fun, grad_fun, pred_fun, hiddens_fun, weights


def build_morgan_flat_net(fp_length=512, fp_radius=4):
    """Wraps functions for computing Morgan fingerprints."""
    grad_fun = lambda w, s, t: 0.0
    loss_fun = lambda w, s, t: 0.0
    pred_fun = lambda w, s:    0.0
    hiddens_fun = lambda w, s: smiles_to_fps(s, fp_length, fp_radius)

    return loss_fun, grad_fun, pred_fun, hiddens_fun, fp_length