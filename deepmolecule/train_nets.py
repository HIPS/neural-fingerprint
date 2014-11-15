# Routines for building and training different types of networks.
#
# David Duvenaud
# Dougal Maclaurin
# Ryan P. Adams
#
# Sept 2014

import sys, os
import numpy as np
import numpy.random as npr

from deepmolecule import normalize_array, sgd_with_momentum, rms_prop, print_performance
from deepmolecule import tictoc, load_data, build_universal_net, build_morgan_deep_net
from deepmolecule import plot_predictions, plot_maximizing_inputs, plot_weight_meanings


def train_nn(net_builder_fun, smiles, raw_targets, arch_params, train_params,
             validation_smiles=None, validation_targets=None,
             optimization_routine=rms_prop):
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
                np.sqrt(np.mean((train_preds - raw_targets) ** 2)),
            if validation_smiles is not None:
                validation_preds = undo_norm(pred_fun(weights, validation_smiles))
                print "Validation RMSE", epoch, ":", \
                    np.sqrt(np.mean((validation_preds - validation_targets) ** 2)),
        else:
            print ".",

    grad_fun_with_data = lambda idxs, w: grad_fun(w, smiles[idxs], targets[idxs])
    trained_weights = optimization_routine(grad_fun_with_data, len(targets), weights.N,
                                           callback, **train_params)
    return lambda new_smiles: undo_norm(pred_fun(trained_weights, new_smiles)), trained_weights


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
    return lambda new_smiles: undo_norm(np.dot(output_layer_fun(net_weights, new_smiles),
                                               linear_weights))


def load_and_train_nn(params_file, output_dir):
    with open(params_file, 'r') as f:
        allparams = eval(f.read())
    train_params = allparams['train_params']
    arch_params = allparams['arch_params']
    task_params = allparams['task_params']
    net_type = allparams['net_type']
    optimizer = allparams['optimizer']
    run_nn_with_params(train_params, arch_params, task_params, output_dir,
                       net_type, optimizer)


def run_nn_with_params(train_params, arch_params, task_params, output_dir,
                       net_type='conv', optimizer='rmsprop'):
    print "Training params: ", train_params
    print "Architecture params: ", arch_params
    print "Task params:", task_params
    print "Net type:", net_type
    print "Optimizer:", optimizer
    print "Output directory:", output_dir

    def get_output_file(filename):
        return os.path.join(output_dir, filename)

    print "\nLoading data..."
    traindata, valdata, testdata = load_data(task_params['data_file'],
                                             (task_params['N_train'], task_params['N_valid'], task_params['N_test']))
    train_inputs, train_targets = traindata['smiles'], traindata[task_params['target_name']]
    val_inputs, val_targets = valdata['smiles'], valdata[task_params['target_name']]
    test_inputs, test_targets = testdata['smiles'], testdata[task_params['target_name']]

    def save_net(weights, arch_params, filename):
        np.savez_compressed(file=get_output_file(filename), weights=weights,
                            arch_params=arch_params)

    print "-" * 80
    print "Mean predictor"
    y_train_mean = np.mean(train_targets)
    print_performance(lambda x: y_train_mean, train_inputs, train_targets,
                      test_inputs, test_targets)

    print "-" * 80
    print "Training neural net:"
    if net_type is "conv":
        net_training_function = build_universal_net
    elif net_type is "morgan":
        net_training_function = build_morgan_deep_net
    else:
        raise Exception("No such type of neural network.")

    if optimizer is "rmsprop":
        optimization_routine = rms_prop
    elif net_type is "sgd":
        optimization_routine = sgd_with_momentum
    else:
        raise Exception("No such optimization routine.")

    with tictoc():
        predictor, weights = train_nn(net_training_function, train_inputs, train_targets,
                                      arch_params, train_params, val_inputs, val_targets,
                                      optimization_routine=optimization_routine)
        print "\n"
    print_performance(predictor, train_inputs, train_targets,
                      test_inputs, test_targets, task_params['target_name'],
                      get_output_file('predictions-mass'))
    save_net(weights, arch_params, 'net-weights')

    plot_predictions(get_output_file('predictions-mass.npz'),
                     os.path.join(output_dir, 'prediction-mass-plots'))
    plot_maximizing_inputs(build_universal_net, get_output_file('net-weights.npz'),
                           os.path.join(output_dir, 'features-mass'))
    plot_weight_meanings(get_output_file('net-weights.npz'),
                         os.path.join(output_dir, 'prediction-mass-plots'),
                         'true-vs-atomvecs')


if __name__ == '__main__':
    sys.exit(load_and_train_nn(*sys.argv[1:]))