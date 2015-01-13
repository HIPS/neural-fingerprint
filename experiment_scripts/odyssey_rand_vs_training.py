# Designed to demonstrate that convnets are a generalization of morgan fingerprints.

import os, time
from deepmolecule import get_data_file, run_jobs, output_dir

task_params = {'N_train': 20000,
               'N_valid': 10000,
               'N_test': 10000,
               'target_name': 'Log Rate',
               'data_file': get_data_file('2014-11-03-all-tddft/processed.csv')}

train_params = {'num_epochs': 50,
                'batch_size': 20,
                'learn_rate': 1e-3,
                'momentum': 0.98,
                'param_scale': 0.1,
                'gamma': 0.9}

def conv_job_generator():
    # Parameters for convolutional net.
    arch_params = {'num_hidden_features': [50, 50, 50],
                   'permutations': True}
    for l_ix, learn_rate in enumerate((1e-2, 1e-3, 1e-4, 1e-5)):
        train_params['learn_rate'] = learn_rate
        for h_ix, num_hid in enumerate((1, 20, 50, 100)):
            arch_params['num_hidden_features'] = [num_hid] * 3
            job_name = 'crh_' + str(l_ix) + '_' + str(h_ix)
            yield job_name, {'train_params': train_params,
                             'arch_params': arch_params,
                             'task_params': task_params,
                             'net_type': 'conv',
                             'optimizer': 'rmsprop'}

def morgan_job_generator():
    # Parameters for standard net build on Morgan fingerprints.
    arch_params = {'h1_size': 10,
                   'h1_dropout': 0.01,
                   'fp_length': 512,
                   'fp_radius': 4}
    for l_ix, learn_rate in enumerate((1e-2, 1e-3, 1e-4, 1e-5)):
        train_params['learn_rate'] = learn_rate
        for h_ix, num_hid in enumerate((50, 100, 500)):
            arch_params['h1_size'] = num_hid
            job_name = 'mrh_' + str(l_ix) + '_' + str(h_ix)
            yield job_name, {'train_params': train_params,
                             'arch_params': arch_params,
                             'task_params': task_params,
                             'net_type': 'morgan',
                             'optimizer': 'rmsprop'}

def collate_jobs():
    pass
    # git pull...
    #
    # for (train_params, arch_params, dir_name) in job_generator:
    # if dir_name exists:
    #       results( train_params arch_params) = load_data_from_dir( dir_name )
    #
    #    #plot_predictions(get_output_file('convnet-predictions.npz'),
    #                 os.path.join(output_dir(), 'convnet-prediction-plots'))
    #plot_maximizing_inputs(build_universal_net, get_output_file('conv-net-weights.npz'),
    #                       os.path.join(output_dir(), 'convnet-features'))

if __name__ == "__main__":
    experiment_name = "compare-rate-accuracy-conv"
    experiment_dir = time.strftime("%Y-%m-%d-") + experiment_name
    dir_prefix = os.path.join(output_dir(), experiment_dir)
    run_jobs(conv_job_generator, '../deepmolecule/train_nets.py', dir_prefix)

    experiment_name = "compare-rate-accuracy-morgan"
    experiment_dir = time.strftime("%Y-%m-%d-") + experiment_name
    dir_prefix = os.path.join(output_dir(), experiment_dir)
    run_jobs(morgan_job_generator, '../deepmolecule/train_nets.py', dir_prefix)
    #collate_jobs()
