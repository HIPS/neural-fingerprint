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
import resource

from deepmolecule import normalize_array, sgd_with_momentum, rms_prop, conj_grad
from deepmolecule import tictoc, load_data, build_universal_net, build_morgan_deep_net
from deepmolecule import plot_predictions, plot_maximizing_inputs, plot_weight_meanings
from deepmolecule import plot_weights, plot_weights_container, plot_learning_curve

#import matplotlib
#matplotlib.use('Agg')   # Cluster-friendly backend.
#import matplotlib.pyplot as plt

def train_nn(net_builder_fun, smiles, raw_targets, arch_params, train_params,
             validation_smiles=None, validation_targets=None,
             optimization_routine=rms_prop):
    npr.seed(1)
    targets, undo_norm = normalize_array(raw_targets)
    loss_fun, grad_fun, pred_fun, _, weights_container = net_builder_fun(**arch_params)
    print "Weight matrix shapes:"
    weights_container.print_shapes()
    print "Total number of weights in the network:", weights_container.N

    #plt.ion()
    #fig = plt.figure(figsize=(12,10))
    training_curve = []
    def callback(epoch, weights):

        #fig = plt.figure(figsize=(12,10))
        #plot_weights_container(weights_container, fig)
        #plt.draw()
        #plt.pause(0.05)
        #plt.title(str(epoch))

        if epoch % 10 == 0:
            print 'Memory usage: %s (MB)' % (resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1000000.0)
            train_preds = undo_norm(pred_fun(weights, smiles))
            cur_loss = loss_fun(weights, smiles, targets).flatten()[0]
            training_curve.append(cur_loss)
            print "\nEpoch", epoch, "loss", cur_loss, "train RMSE", \
                np.sqrt(np.mean((train_preds - raw_targets) ** 2)),
            if validation_smiles is not None:
                validation_preds = undo_norm(pred_fun(weights, validation_smiles))
                print "Validation RMSE", epoch, ":", \
                    np.sqrt(np.mean((validation_preds - validation_targets) ** 2)),
        else:
            print ".",
    #plt.close()

    def grad_fun_with_data(training_idxs, weights):
        return grad_fun(weights, smiles[training_idxs], targets[training_idxs])
    trained_weights = optimization_routine(grad_fun_with_data, len(targets), weights_container.N,
                                           callback, **train_params)

    def predict_func(new_smiles):
        """Returns to the original units that the raw targets were in."""
        return undo_norm(pred_fun(trained_weights, new_smiles))
    return predict_func, trained_weights, training_curve


def random_net_linear_output(net_builder_fun, smiles, raw_targets, arch_params, train_params):
    """Creates a network with random weights, and trains a ridge regression model on top."""
    npr.seed(1)
    targets, undo_norm = normalize_array(raw_targets)
    _, _, _, output_layer_fun, weights = net_builder_fun(**arch_params)
    net_weights = npr.randn(weights.N) * train_params['param_scale']
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

    def print_performance(pred_func, filename=None):
        train_preds = pred_func(train_inputs)
        val_preds = pred_func(val_inputs)
        test_preds = pred_func(test_inputs)
        print "\nPerformance (RMSE) on " + task_params['target_name'] + ":"
        print "Train:", np.sqrt(np.mean((train_preds - train_targets)**2))
        print "Validation:", np.sqrt(np.mean((val_preds - val_targets)**2))
        print "Test: ", np.sqrt(np.mean((test_preds - test_targets)**2))
        print "-" * 80

        if filename:
            np.savez_compressed(file=filename, target_name=task_params['target_name'],
                                train_preds=train_preds, train_targets=train_targets,
                                val_preds=train_preds, val_targets=train_targets,
                                test_preds=test_preds, test_targets=test_targets)

    print "-" * 80
    print "Mean predictor"
    y_train_mean = np.mean(train_targets)
    print_performance(lambda x: y_train_mean)

    print "-" * 80
    print "Training neural net:"
    if net_type is "conv":
        net_training_function = build_universal_net
    elif net_type is "morgan":
        net_training_function = build_morgan_deep_net
    elif net_type is "morgan-linear":
        net_training_function = build_morgan_linear
    elif net_type is "conv-linear":
        net_training_function = random_net_linear_output
    else:
        raise Exception("No such type of neural network.")

    if optimizer is "rmsprop":
        optimization_routine = rms_prop
    elif optimizer is "sgd":
        optimization_routine = sgd_with_momentum
    elif optimizer is "conj_grad":
        optimization_routine = conj_grad
    else:
        raise Exception("No such optimization routine.")

    with tictoc():
        predictor, weights, learning_curve = train_nn(net_training_function,
            train_inputs, train_targets, arch_params, train_params, val_inputs,
            val_targets, optimization_routine=optimization_routine)
        print "\n"
    print_performance(predictor, get_output_file('predictions'))
    np.savez_compressed(file=get_output_file('net-weights'),
                        weights=weights, arch_params=arch_params)
    np.savez_compressed(file=get_output_file('learning-curve'), learning_curve=learning_curve)

    plot_predictions(get_output_file('predictions.npz'),
                     os.path.join(output_dir, 'plots'))
    plot_maximizing_inputs(net_training_function, get_output_file('net-weights.npz'),
                           os.path.join(output_dir, 'features'))
    plot_weights(net_training_function, get_output_file('net-weights.npz'),
                 os.path.join(output_dir, 'plots'))
    if net_type is "conv":
        plot_weight_meanings(net_training_function, get_output_file('net-weights.npz'),
                             os.path.join(output_dir, 'plots'), 'true-vs-atomvecs')
    plot_learning_curve(get_output_file('learning-curve.npz'), output_dir)


if __name__ == '__main__':
    sys.exit(load_and_train_nn(*sys.argv[1:]))