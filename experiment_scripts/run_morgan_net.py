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

from deepmolecule import tictoc, load_data
from deepmolecule import build_morgan_deep_net


def run_conv_net(params_file, output_dir):

    with open(params_file, 'r') as f:
        allparams = eval(f.read())
    morgan_train_params = allparams['morgan_train_params']
    morgan_arch_params = allparams['morgan_arch_params']
    task_params = allparams['task_params']

    print "Morgan net training params: ", morgan_train_params
    print "Morgan net architecture params: ", morgan_arch_params
    print "Task params", task_params
    print "Output directory:", output_dir

    def get_output_file(filename):
        return os.path.join(output_dir, filename)

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

    print "Training custom neural net"
    with tictoc():
        predictor, weights = train_nn(build_morgan_deep_net, train_inputs, train_targets,
                             morgan_arch_params, morgan_train_params, val_inputs, val_targets)
        print "\n"
    print_performance(predictor, 'morgan-predictions')
    #plot_predictions(get_output_file('convnet-predictions.npz'),
    #                 os.path.join(output_dir(), 'convnet-prediction-plots'))
    save_net(weights, morgan_arch_params, 'morgan-net-weights')
    #plot_maximizing_inputs(build_universal_net, get_output_file('conv-net-weights.npz'),
    #                       os.path.join(output_dir(), 'convnet-features'))

if __name__ == '__main__':
    sys.exit(run_conv_net(*sys.argv[1:]))
