# Compares the predictive accuracy of using
# standard fingerprints versus custom convolutional nets.
#
# Dougal Maclaurin
# David Duvenaud
# Ryan P. Adams
#
# Sept 2014

import sys, os
import numpy as np
import numpy.random as npr

from deepmolecule import tictoc, normalize_array, sgd_with_momentum, get_data_file, load_data
from deepmolecule import build_morgan_deep_net, build_morgan_flat_net
from deepmolecule import build_universal_net, output_dir, get_output_file
from deepmolecule import plot_predictions, plot_maximizing_inputs

def train_nn(net_builder_fun, smiles, raw_targets, arch_params, train_params,
             validation_smiles=None, validation_targets=None):
    npr.seed(1)
    targets, undo_norm = normalize_array(raw_targets)
    loss_fun, grad_fun, pred_fun, _, N_weights = net_builder_fun(**arch_params)
    def callback(epoch, weights):
        if epoch % 10 == 0:
            train_preds = undo_norm(pred_fun(weights, smiles))
            print "\nTraining RMSE after epoch", epoch, ":",\
                np.sqrt(np.mean((train_preds - raw_targets)**2)),
            if validation_smiles is not None:
                validation_preds = undo_norm(pred_fun(weights, validation_smiles))
                print "Validation RMSE", epoch, ":",\
                    np.sqrt(np.mean((validation_preds - validation_targets)**2)),
        else:
            print ".",
    grad_fun_with_data = lambda idxs, w : grad_fun(w, smiles[idxs], targets[idxs])
    trained_weights = sgd_with_momentum(grad_fun_with_data, len(targets), N_weights,
                                        callback, **train_params)
    return lambda new_smiles : undo_norm(pred_fun(trained_weights, new_smiles)), trained_weights


def run_conv_net(params_file, output_dir):

    with open(params_file, 'r') as f:
        allparams = eval(f.read())
    morgan_train_params = allparams['morgan_train_params']
    morgan_arch_params = allparams['morgan_arch_params']
    task_params = allparams['task_params']

    print "Morgan net training params: ", morgan_train_params
    print "Morgan net architecture params: ", morgan_arch_params
    print "Task params", task_params
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
        test_preds = pred_func(test_inputs)
        print "\nPerformance (RMSE) on " + task_params['target_name'] + ":"
        print "Train:", np.sqrt(np.mean((train_preds - train_targets)**2))
        print "Test: ", np.sqrt(np.mean((test_preds - test_targets)**2))
        print "-" * 80
        if filename:
            np.savez_compressed(file=get_output_file(filename),
                                train_preds=train_preds, train_targets=train_targets,
                                test_preds=test_preds, test_targets=test_targets,
                                target_name=task_params['target_name'])

    def save_net(weights, arch_params, filename):
        np.savez_compressed(file=get_output_file(filename), weights=weights,
                            arch_params=arch_params)

    print "-" * 80
    print "Mean predictor"
    y_train_mean = np.mean(train_targets)
    print_performance(lambda x : y_train_mean)

    print "Training custom neural net"
    with tictoc():
        predictor, weights = train_nn(build_morgan_deep_net, train_inputs, train_targets,
                             morgan_arch_params, morgan_train_params, val_inputs, val_targets)
        print "\n"
    print_performance(predictor, 'morgan-predictions')
    #plot_predictions(get_output_file('convnet-predictions.npz'),
    #                 os.path.join(output_dir(), 'convnet-prediction-plots'))
    save_net(weights, morgan_arch_params, 'morgan-net-weights')
    #plot_maximizing_inputs(build_universal_net, get_output_file('conv-net-weights.npz'),
    #                       os.path.join(output_dir(), 'convnet-features'))

if __name__ == '__main__':
    sys.exit(run_conv_net(*sys.argv[1:]))
