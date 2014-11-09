import kayak as ky
import numpy as np
import numpy.random as npr
from util import WeightsContainer, c_value, c_grad

def build_vanilla_net(num_inputs, h1_size, h1_dropout):
    weights = WeightsContainer()
    # Need to give fake data here so that the dropout node can make a mask.
    inputs =  ky.Inputs(np.zeros((1, num_inputs)))
    W1 = weights.new((num_inputs, h1_size))
    B1 = weights.new((1, h1_size))
    hidden = ky.HardReLU(ky.MatMult(inputs, W1) + B1)
    dropout = ky.Dropout(hidden, drop_prob=h1_dropout, rng=npr.seed(1))
    W2 = weights.new(h1_size)
    B2 = weights.new(1)
    output =  ky.MatMult(dropout, W2) + B2
    target =  ky.Blank()
    loss =  ky.L2Loss(output, target)

    # All the functions we'll need to train and predict with this net.
    def grad_fun(w, i, t):
        dropout.draw_new_mask()
        return c_grad(loss, weights, {weights : w, inputs : i, target : t})
    def loss_fun(w, i, t):
        return c_value(loss, {weights : w, inputs : i, target : t})
    def pred_fun(w, i):
        inputs.value = i      # Necessary so that the dropout mask will be the right size.
        dropout.reinstate_units()
        return c_value(output, {weights : w, inputs : i})
    def hidden_layer_fun(w, i):
        inputs.value = i      # Necessary so that the dropout mask will be the right size.
        return c_value(hidden, {weights : w, inputs : i})

    return loss_fun, grad_fun, pred_fun, hidden_layer_fun, weights.N

