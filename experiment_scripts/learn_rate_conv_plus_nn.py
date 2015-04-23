# This experiment is designed to test a conv net with a neural net on top.
#
# David Duvenaud
# Dougal Maclaurin
# Ryan P. Adams
#
# April 22nd, 2015

import sys, os
os.environ['DATA_DIR'] = "/Users/dkd/repos/DeepMoleculesData/data"
os.environ['OUTPUT_DIR'] = "/tmp"

from deepmolecule import get_data_file, run_nn_with_params, output_dir

def main():
    conj_grad_params = {'num_epochs'  : 5,
                         'param_scale' : 0.1}

    # Parameters for convolutional net.
    arch_params = {'num_hidden_features' : [20, 20, 20],
                        'bond_vec_dim' : 10,
                        'permutations' : False,
                        'vanilla_hidden' : 100,
                        'l2_penalty': 0.001}

    task_params = {'N_train'     : 200,
                   'N_valid'     : 100,
                   'N_test'      : 300,
                   'target_name' : 'Log Rate',
                   'data_file'   : get_data_file('2014-11-03-all-tddft/processed.csv')}

    run_nn_with_params(train_params=conj_grad_params, optimizer="rmsprop",
                       arch_params=arch_params,task_params=task_params,
                       output_dir=output_dir(), net_type='conv-plus-nn')

if __name__ == '__main__':
    sys.exit(main())
