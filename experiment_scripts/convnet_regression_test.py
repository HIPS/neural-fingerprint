# A simple script to check whether two implementations are equivalent.

import sys, os
import numpy.random as npr

from deepmolecule import get_data_file, load_data
from deepmolecule import build_convnet_fingerprint_fun

from deepmolecule import tictoc


def main():

    os.environ['DATA_DIR'] = os.path.expanduser("~/repos/DeepMoleculesData/data")

    # Parameters for convolutional net.
    conv_arch_params = {'num_hidden_features' : [50, 50, 120],
                        'fp_length':512}

    task_params = {'N_train'     : 200,
                   'target_name' : 'Log Rate',
                   'data_file'   : get_data_file('2014-11-03-all-tddft/processed.csv')}

    print "Conv net architecture params: ", conv_arch_params
    print "Task params", task_params

    print "\nLoading data..."
    (traindata, ) = load_data(task_params['data_file'], (task_params['N_train'],))
    train_inputs, train_targets = traindata['smiles'], traindata[task_params['target_name']]

    fp_fun, parser = build_convnet_fingerprint_fun(**conv_arch_params)
    npr.seed(0)
    weights = npr.randn(len(parser))

    with tictoc():
        print "Outputs: ", fp_fun(weights, train_inputs)


if __name__ == '__main__':
    sys.exit(main())

"""Should produce:
Outputs:  [[ 0.36883396  0.23668908  0.05273891 ...,  0.03304657  0.04665458
   0.03438516]
 [ 0.17770927  0.08918203  0.03132146 ...,  0.0102424   0.74531305
   0.01682093]
 [ 0.14587046  0.08922296  0.05089039 ...,  0.00936303  0.00747896
   0.37331685]
 ...,
 [ 0.04576854  0.06240084  0.02145457 ...,  0.04729211  0.14872239
   0.00758755]
 [ 0.50115373  0.12300134  0.04225563 ...,  0.02029587  0.02006073
   0.03598508]
 [ 0.48323492  0.45061002  0.03230126 ...,  0.01356204  0.01025982
   0.03237921]]
"""
