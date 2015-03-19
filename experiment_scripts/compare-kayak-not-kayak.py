# A simple script to check whether two implementations are equivalent.

import sys, os
import numpy as np
import numpy.random as npr

from deepmolecule import get_data_file, load_data
from deepmolecule import build_universal_net, build_universal_net_ag

def main():

    os.environ['DATA_DIR'] = "/Users/dkd/repos/DeepMoleculesData/data"

    # Parameters for convolutional net.
    conv_arch_params = {'num_hidden_features' : [5, 5, 12],
                        'permutations' : True}

    task_params = {'N_train'     : 200,
                   'target_name' : 'Log Rate',
                   'data_file'   : get_data_file('2014-11-03-all-tddft/processed.csv')}

    print "Conv net architecture params: ", conv_arch_params
    print "Task params", task_params

    print "\nLoading data..."
    (traindata, ) = load_data(task_params['data_file'], (task_params['N_train'],))
    train_inputs, train_targets = traindata['smiles'], traindata[task_params['target_name']]

    loss_fun1, grad_fun1, pred_fun1, hiddens_fun1, parser1 = build_universal_net(**conv_arch_params)
    loss_fun2, grad_fun2, pred_fun2, hiddens_fun2, parser2 = build_universal_net_ag(**conv_arch_params)

    print parser1._names_list
    print parser2.idxs_and_shapes.keys()

    npr.seed(0)
    weights = npr.randn(parser1.N)
    #weights = 0.5 * np.ones(parser1.N)

    print "Difference in outputs: ", np.sum(np.abs(pred_fun1(weights, train_inputs) - pred_fun2(weights, train_inputs)))
    print "Difference in loss: ", np.sum(np.abs(loss_fun1(weights, train_inputs, train_targets) - loss_fun2(weights, train_inputs, train_targets)))


    #print "Random convolutional net with linear weights"
    #predictor = random_net_linear_output(build_universal_net, train_inputs, train_targets,
    #                                     conv_arch_params, linear_train_params)
    #print_performance(predictor)


if __name__ == '__main__':
    sys.exit(main())
