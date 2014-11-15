# Routines for building and training different types of networks.
#
# David Duvenaud
# Dougal Maclaurin
# Ryan P. Adams
#
# Sept 2014

import numpy as np
import numpy.random as npr

from deepmolecule import normalize_array, sgd_with_momentum

def train_nn(net_builder_fun, smiles, raw_targets, arch_params, train_params,
             validation_smiles=None, validation_targets=None, optimization_routine=sgd_with_momentum):
    npr.seed(1)
    targets, undo_norm = normalize_array(raw_targets)
    loss_fun, grad_fun, pred_fun, _, weights = net_builder_fun(**arch_params)
    print "Weight matrix shapes:"
    weights.print_shapes()
    print "Total number of weights in the network:", weights.N
    def callback(epoch, weights):
        if epoch % 10 == 0:
            train_preds = undo_norm(pred_fun(weights, smiles))
            cur_loss = loss_fun(weights, smiles, targets).flatten()[0]
            print "\nEpoch", epoch, "loss", cur_loss, "train RMSE", \
                np.sqrt(np.mean((train_preds - raw_targets)**2)),
            if validation_smiles is not None:
                validation_preds = undo_norm(pred_fun(weights, validation_smiles))
                print "Validation RMSE", epoch, ":", \
                    np.sqrt(np.mean((validation_preds - validation_targets)**2)),
        else:
            print ".",
    grad_fun_with_data = lambda idxs, w : grad_fun(w, smiles[idxs], targets[idxs])
    trained_weights = optimization_routine(grad_fun_with_data, len(targets), weights.N,
                                           callback, **train_params)
    return lambda new_smiles : undo_norm(pred_fun(trained_weights, new_smiles)), trained_weights


def random_net_linear_output(net_builder_fun, smiles, raw_targets, arch_params, train_params):
    """Creates a network with random weights, and trains a ridge regression model on top."""
    npr.seed(1)
    targets, undo_norm = normalize_array(raw_targets)
    _, _, _, output_layer_fun, N_weights = net_builder_fun(**arch_params)
    net_weights = npr.randn(N_weights) * train_params['param_scale']
    train_outputs = output_layer_fun(net_weights, smiles)
    linear_weights = np.linalg.solve(np.dot(train_outputs.T, train_outputs)
                                     + np.eye(train_outputs.shape[1]) * train_params['l2_reg'],
                                     np.dot(train_outputs.T, targets))
    return lambda new_smiles : undo_norm(np.dot(output_layer_fun(net_weights, new_smiles),
                                                linear_weights))

