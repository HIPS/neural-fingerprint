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

from deepmolecule import tictoc, get_data_file, load_data
from deepmolecule import build_universal_net, output_dir, get_output_file
from deepmolecule import plot_predictions, plot_maximizing_inputs, plot_weight_meanings
from deepmolecule import train_nn, rms_prop, print_performance

def main():
    # Parameters for convolutional net.
    conv_train_params = {'num_epochs'  : 50,
                         'batch_size'  : 10,
                         'learn_rate'  : 1e-3,
                         'momentum'    : 0.98,
                         'param_scale' : 0.1,
                         'gamma': 0.9}
    conv_arch_params = {'num_hidden_features' : [1, 1],
                        'bond_vec_dim' : 3,
                        'permutations' : False,
                        'l2_penalty': 0.0001}

    task_params = {'N_train'     : 2000,
                   'N_valid'     : 10,
                   'N_test'      : 30,
                   'target_name' : 'Molecular Weight',
                   'data_file'   : get_data_file('2014-11-03-all-tddft/processed.csv')}

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

    def save_net(weights, arch_params, filename):
        np.savez_compressed(file=get_output_file(filename), weights=weights,
                            arch_params=arch_params)

    print "-" * 80
    print "Mean predictor"
    y_train_mean = np.mean(train_targets)
    print_performance(lambda x : y_train_mean, train_inputs, train_targets,
                      test_inputs, test_targets)

    print "-" * 80
    print "Training custom neural net: RMSProp"
    with tictoc():
        predictor, weights = train_nn(build_universal_net, train_inputs, train_targets,
                             conv_arch_params, conv_train_params, val_inputs, val_targets,
                             optimization_routine=rms_prop)
        print "\n"
    print_performance(predictor, train_inputs, train_targets,
                      test_inputs, test_targets, task_params['target_name'],
                      get_output_file('convnet-predictions-mass'))

    save_net(weights, conv_arch_params, 'conv-net-weights')

    plot_predictions(get_output_file('convnet-predictions-mass.npz'),
                     os.path.join(output_dir(), 'convnet-prediction-mass-plots'))
    plot_maximizing_inputs(build_universal_net, get_output_file('conv-net-weights.npz'),
                           os.path.join(output_dir(), 'convnet-features-mass'))
    plot_weight_meanings(get_output_file('conv-net-weights.npz'),
                                          os.path.join(output_dir(), 'convnet-prediction-mass-plots'),
                                          'true-vs-atomvecs')


if __name__ == '__main__':
    sys.exit(main())
