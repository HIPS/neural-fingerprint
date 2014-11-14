# This experiment is designed to show that convnets can learn to predict molecular weight, while
# Morgan fingerprints can't.
#
# David Duvenaud
# Dougal Maclaurin
# Ryan P. Adams
#
# Sept 2014

import sys, os
import numpy as np

from deepmolecule import tictoc,  get_data_file, load_data
from deepmolecule import build_morgan_deep_net, build_morgan_flat_net
from deepmolecule import build_universal_net, output_dir, get_output_file
from deepmolecule import plot_predictions, plot_maximizing_inputs
from deepmolecule import random_net_linear_output, train_nn

def main():
    # Parameters for convolutional net.
    conv_train_params = {'num_epochs'  : 50,
                         'batch_size'  : 200,
                         'learn_rate'  : 1e-3,
                         'momentum'    : 0.98,
                         'param_scale' : 0.1}
    conv_arch_params = {'num_hidden_features' : [10, 10, 10],
                        'permutations' : False}

    # Parameters for standard net build on Morgan fingerprints.
    morgan_train_params = {'num_epochs'  : 500,
                           'batch_size'  : 2,
                           'learn_rate'  : 1e-3,
                           'momentum'    : 0.98,
                           'param_scale' : 0.1}
    morgan_deep_arch_params = {'h1_size'    : 10,
                               'h1_dropout' : 0.01,
                               'fp_length'  : 512,
                               'fp_radius'  : 4}
    morgan_flat_arch_params = {'fp_length'  : 512,
                               'fp_radius'  : 4}

    linear_train_params = {'param_scale' : 0.1,
                           'l2_reg'      : 0.1}

    task_params = {'N_train'     : 20,
                   'N_valid'     : 10,
                   'N_test'      : 30,
                   'target_name' : 'Molecular Weight',
                   #'target_name' : 'Log Rate',
                   'data_file'   : get_data_file('2014-11-03-all-tddft/processed.csv')}
    #target_name = 'Log Rate'
    #'Polar Surface Area'
    #'Number of Rings'
    #'Number of H-Bond Donors'
    #'Number of Rotatable Bonds'
    #'Minimum Degree'

    print "Linear net training params: ", linear_train_params
    print "Morgan net training params: ", morgan_train_params
    print "Morgan flat net architecture params: ", morgan_flat_arch_params
    print "Morgan deep net architecture params: ", morgan_deep_arch_params
    print "Conv net training params: ", conv_train_params
    print "Conv net architecture params: ", conv_arch_params
    print "Task params", task_params
    print "Output directory:", output_dir()

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

    print "Fingerprints with linear weights"
    predictor = random_net_linear_output(build_morgan_flat_net, train_inputs, train_targets,
                                         morgan_flat_arch_params, linear_train_params)
    print_performance(predictor)

    print "Fingerprints with random net on top and linear weights"
    predictor = random_net_linear_output(build_morgan_deep_net, train_inputs, train_targets,
                                         morgan_deep_arch_params, linear_train_params)
    print_performance(predictor)

    print "Random convolutional net with linear weights"
    predictor = random_net_linear_output(build_universal_net, train_inputs, train_targets,
                                         conv_arch_params, linear_train_params)
    print_performance(predictor)

    print "Training vanilla neural net"
    with tictoc():
        predictor, weights = train_nn(build_morgan_deep_net, train_inputs, train_targets,
                             morgan_deep_arch_params, morgan_train_params, val_inputs, val_targets)
        print "\n"
    print_performance(predictor, 'vanilla-predictions')
    plot_predictions(get_output_file('vanilla-predictions.npz'),
                     os.path.join(output_dir(), 'vanilla-prediction-plots'))
    save_net(weights, morgan_deep_arch_params, 'morgan-net-weights')
    plot_maximizing_inputs(build_morgan_deep_net, get_output_file('morgan-net-weights.npz'),
                           os.path.join(output_dir(), 'morgan-features'))

    print "Training custom neural net : array representation"
    with tictoc():
        predictor, weights = train_nn(build_universal_net, train_inputs, train_targets,
                             conv_arch_params, conv_train_params, val_inputs, val_targets)
        print "\n"
    print_performance(predictor, 'convnet-predictions')
    plot_predictions(get_output_file('convnet-predictions.npz'),
                     os.path.join(output_dir(), 'convnet-prediction-plots'))
    save_net(weights, conv_arch_params, 'conv-net-weights')
    plot_maximizing_inputs(build_universal_net, get_output_file('conv-net-weights.npz'),
                           os.path.join(output_dir(), 'convnet-features'))

if __name__ == '__main__':
    sys.exit(main())
