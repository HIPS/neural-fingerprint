# Runs a big experiment.
import os
import time

from deepmolecule import get_data_file, run_jobs, call_on_odyssey
from deepmolecule import output_dir, get_output_file
from deepmolecule import plot_predictions, plot_maximizing_inputs

def job_generator():
    # Parameters for convolutional net.
    conv_train_params = {'num_epochs'  : 5,
                         'batch_size'  : 200,
                         'learn_rate'  : 1e-3,
                         'momentum'    : 0.98,
                         'param_scale' : 0.1}
    conv_arch_params = {'num_hidden_features' : [10, 10, 10],
                        'permutations' : False}
    task_params = {'N_train'        : 20,
                   'N_valid'        : 10,
                   'N_test'         : 10,
                   'target_name' : 'Log Rate',
                   'data_file'   : get_data_file('2014-11-03-all-tddft/processed.csv')}

    for l_ix, learn_rate in enumerate((1e-2, 1e-3, 1e-4, 1e-5)):
        conv_train_params['learn_rate'] = learn_rate
        for h_ix, num_hid in enumerate((1,  20,   50)):
            conv_arch_params['num_hidden_features'] = [num_hid] * 3
            job_name = 'rates_and_hids_' + str(l_ix) + '_' + str(h_ix)
            yield job_name, {'conv_train_params': conv_train_params,
                             'conv_arch_params' : conv_arch_params,
                             'task_params' : task_params}

def collate_jobs():
    pass
    # git pull...
    #
    # for (train_params, arch_params, dir_name) in job_generator:
    #   if dir_name exists:
    #       results( train_params arch_params) = load_data_from_dir( dir_name )
    #
    #    #plot_predictions(get_output_file('convnet-predictions.npz'),
        #                 os.path.join(output_dir(), 'convnet-prediction-plots'))
        #plot_maximizing_inputs(build_universal_net, get_output_file('conv-net-weights.npz'),
        #                       os.path.join(output_dir(), 'convnet-features'))

experiment_name = "test-job"
experiment_dir = time.strftime("%Y-%m-%d-") + experiment_name
dir_prefix = os.path.join(output_dir(), experiment_dir)

if __name__ == "__main__":

    run_jobs(job_generator, 'run_convnet.py', dir_prefix)
    #collate_jobs()
