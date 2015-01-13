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
from deepmolecule import plot_predictions, plot_maximizing_inputs, random_net_linear_output


def main():
    # Parameters for convolutional net.
    conv_arch_params = {'num_hidden_features' : [50, 50, 512],
                        'permutations' : False}

    morgan_flat_arch_params = {'fp_length'  : 512,
                               'fp_radius'  : 3}

    linear_train_params = {'param_scale' : 0.1,
                           'l2_reg'      : 0.1}

    task_params = {'N_train'     : 200,
                   'N_valid'     : 100,
                   'N_test'      : 100,
                   #'target_name' : 'Molecular Weight',
                   'target_name' : 'Log Rate',
                   'data_file'   : get_data_file('2014-11-03-all-tddft/processed.csv')}
    #target_name = 'Log Rate'
    #'Polar Surface Area'
    #'Number of Rings'
    #'Number of H-Bond Donors'
    #'Number of Rotatable Bonds'
    #'Minimum Degree'

    print "Linear net training params: ", linear_train_params
    print "Morgan flat net architecture params: ", morgan_flat_arch_params
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

    print "-" * 80
    print "Mean predictor"
    y_train_mean = np.mean(train_targets)
    print_performance(lambda x : y_train_mean)

    print "Random convolutional net with linear weights"
    predictor = random_net_linear_output(build_universal_net, train_inputs, train_targets,
                                         conv_arch_params, linear_train_params)
    print_performance(predictor)

    print "Morgan Fingerprints with linear weights"
    predictor = random_net_linear_output(build_morgan_flat_net, train_inputs, train_targets,
                                         morgan_flat_arch_params, linear_train_params)
    print_performance(predictor)




if __name__ == '__main__':
    sys.exit(main())
