# A simple script to check whether two implementations are equivalent.
#
# To profile, run
# python -m cProfile -s time experiment_scripts/convnet_profiling_test.py

import sys, os
import numpy.random as npr

from deepmolecule import get_data_file, load_data
from deepmolecule import build_universal_net

def main():

    os.environ['DATA_DIR'] = "/Users/dkd/repos/DeepMoleculesData/data"

    # Parameters for convolutional net.
    conv_arch_params = {'num_hidden_features' : [50, 50, 100],
                        'permutations' : True}

    task_params = {'N_train'     : 200,
                   'target_name' : 'Log Rate',
                   'data_file'   : get_data_file('2014-11-03-all-tddft/processed.csv')}

    print "Conv net architecture params: ", conv_arch_params
    print "Task params", task_params

    print "\nLoading data..."
    (traindata, ) = load_data(task_params['data_file'], (task_params['N_train'],))
    train_inputs, train_targets = traindata['smiles'], traindata[task_params['target_name']]

    loss_fun, grad_fun, pred_fun, hiddens_fun, parser = build_universal_net(**conv_arch_params)
    npr.seed(0)
    weights = npr.randn(parser.N)
    print "Grad: ", grad_fun(weights, train_inputs, train_targets)


if __name__ == '__main__':
    sys.exit(main())

