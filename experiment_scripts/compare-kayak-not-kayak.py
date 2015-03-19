# A simple script to check whether two implementations are equivalent.

import sys
import numpy as np
import numpy.random as npr

from deepmolecule import get_data_file, load_data
from deepmolecule import build_morgan_deep_net, build_morgan_deep_net_ag
from deepmolecule import build_universal_net, output_dir, get_output_file
from deepmolecule import random_net_linear_output


def main():
    # Parameters for convolutional net.
    conv_arch_params = {'num_hidden_features' : [50, 50, 512],
                        'permutations' : False}

    morgan_flat_arch_params = {'fp_length'  : 512,
                               'fp_radius'  : 3,
                               'h1_dropout' : 0.5}

    morgan_flat_arch_params_ag = {'fp_length'  : 512,
                               'fp_radius'  : 3}

    linear_train_params = {'param_scale' : 0.1,
                           'l2_reg'      : 0.1}

    task_params = {'N_train'     : 200,
                   'target_name' : 'Log Rate',
                   'data_file'   : get_data_file('2014-11-03-all-tddft/processed.csv')}

    print "Linear net training params: ", linear_train_params
    print "Morgan flat net architecture params: ", morgan_flat_arch_params
    print "Conv net architecture params: ", conv_arch_params
    print "Task params", task_params
    print "Output directory:", output_dir()

    print "\nLoading data..."
    (traindata, ) = load_data(task_params['data_file'], (task_params['N_train'],))
    train_inputs, train_targets = traindata['smiles'], traindata[task_params['target_name']]

    print "\nMorgan Fingerprints with neural net on top"
    #predictor1 = random_net_linear_output(build_morgan_deep_net, train_inputs, train_targets,
    #                                     morgan_flat_arch_params, linear_train_params)

    print "\nMorgan Fingerprints with neural net on top - autograd"
    #predictor2 = random_net_linear_output(build_morgan_deep_net_ag, train_inputs, train_targets,
    #                                     morgan_flat_arch_params_ag, linear_train_params)

    loss_fun1, grad_fun1, pred_fun1, hiddens_fun1, parser1 = build_morgan_deep_net(**morgan_flat_arch_params)
    loss_fun2, grad_fun2, pred_fun2, hiddens_fun2, parser2 = build_morgan_deep_net_ag(**morgan_flat_arch_params_ag)

    weights = npr.randn(parser1.N)

    print "Difference in outputs: ", np.sum(np.abs(pred_fun1(weights, train_inputs) - pred_fun2(weights, train_inputs)))
    print "Difference in loss: ", np.sum(np.abs(loss_fun1(weights, train_inputs, train_targets) - loss_fun2(weights, train_inputs, train_targets)))


    #print "Random convolutional net with linear weights"
    #predictor = random_net_linear_output(build_universal_net, train_inputs, train_targets,
    #                                     conv_arch_params, linear_train_params)
    #print_performance(predictor)


if __name__ == '__main__':
    sys.exit(main())
