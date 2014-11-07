# A quick comparison script to compare the predictive accuracy of using standard fingerprints versus custom convnets.
#
# Dougal Maclaurin
# David Duvenaud
# Ryan P. Adams
#
# Sept 25th, 2014

import sys
import numpy as np
import numpy.random as npr
import kayak

from deepmolecule import tictoc, normalize_array, sgd_with_momentum, batch_idx_generator, get_data_file, load_data
from deepmolecule import initialize_weights, BuildNetFromSmiles, build_universal_net, smiles_to_fps

num_epochs = 50

def train_2layer_nn(smiles, targets):
    batch_size   = 256
    learn_rate   = 0.001
    momentum     = 0.98
    h1_dropout   = 0.01
    h1_size      = 500
    param_scale = 0.1
    fp_length = 512
    fp_radius = 4

    features = smiles_to_fps(smiles, fp_length, fp_radius)
    normed_targets, undo_norm = normalize_array(targets)
    batcher = kayak.Batcher(batch_size, features.shape[0])
    X =  kayak.Inputs(features, batcher)
    T =  kayak.Targets(normed_targets, batcher)
    W1 = kayak.Parameter(param_scale * npr.randn(features.shape[1], h1_size))
    B1 = kayak.Parameter(param_scale * npr.randn(1, h1_size))
    H1 = kayak.Dropout(kayak.HardReLU(kayak.ElemAdd(kayak.MatMult(X, W1), B1)), h1_dropout, batcher=batcher)
    W2 = kayak.Parameter(param_scale * npr.randn(h1_size))
    B2 = kayak.Parameter(param_scale * npr.randn(1))
    Y =  kayak.ElemAdd(kayak.MatMult(H1, W2), B2)
    L =  kayak.L2Loss(Y, T)
    params = [W1, W2, B1, B2]
    step_dirs = [np.zeros(p.shape) for p in params]
    for epoch in xrange(num_epochs):
        total_err = 0.0
        for batch in batcher:
            total_err += np.sum((undo_norm(Y.value) - undo_norm(T.value))**2)
            grads = [L.grad(p) for p in params]
            step_dirs = [momentum * step_dir - (1.0 - momentum) * grad
                         for step_dir, grad in zip(step_dirs, grads)]
            for p, step_dir in zip(params, step_dirs):
                p.value += learn_rate * step_dir
        if epoch % 10 == 0:
            print "RMSE after epoch", epoch, ":", np.sqrt(total_err / len(normed_targets))
        else:
            print ".",

    def make_predictions(newsmiles):
        X.data = smiles_to_fps(newsmiles, fp_length, fp_radius)
        batcher.test_mode()
        return undo_norm(Y.value)

    return make_predictions

def train_universal_custom_nn(smiles, raw_targets, arch_params, train_params):
    npr.seed(1)
    targets, undo_norm = normalize_array(raw_targets)
    loss_fun, grad_fun, pred_fun, _, N_weights = build_universal_net(**arch_params)
    def callback(epoch, weights):
        if epoch % 10 == 0:
            train_preds = undo_norm(pred_fun(weights, smiles))
            print "After epoch", epoch, "loss is", np.sqrt(np.mean((train_preds - raw_targets)**2))
        else:
            print ".",
    grad_fun_with_data = lambda idxs, w : grad_fun(w, smiles[idxs], targets[idxs])
    trained_weights = sgd_with_momentum(grad_fun_with_data, len(targets), N_weights,
                                        callback, **train_params)

    return lambda new_smiles : undo_norm(pred_fun(trained_weights, new_smiles))

def random_net_linear_output(smiles, raw_targets, arch_params, train_params):
    npr.seed(1)
    targets, undo_norm = normalize_array(raw_targets)
    loss_fun, grad_fun, pred_fun, output_layer_fun, N_weights = build_universal_net(**arch_params)
    net_weights = npr.randn(N_weights) * train_params['param_scale']
    train_outputs = output_layer_fun(net_weights, smiles)
    linear_weights = np.linalg.solve( np.dot(train_outputs.T, train_outputs)
                                      + np.eye(train_outputs.shape[1]) * train_params['lambda'],
                                      np.dot(train_outputs.T, targets))
    return lambda new_smiles : undo_norm(np.dot(output_layer_fun(net_weights, new_smiles), linear_weights))

def fingerprints_linear_output(smiles, raw_targets, arch_params, train_params):
    fp_length = 512
    fp_radius = 4
    features = smiles_to_fps(smiles, fp_length, fp_radius)

    targets, undo_norm = normalize_array(raw_targets)
    linear_weights = np.linalg.solve( np.dot(features.T, features)
                                      + np.eye(fp_length) * train_params['lambda'],
                                      np.dot(features.T, targets))
    return lambda new_smiles : undo_norm(np.dot(smiles_to_fps(new_smiles, fp_length, fp_radius),
                                                linear_weights))

def main():
    # Parameters for both custom nets
    train_params = {'num_epochs'  : num_epochs,
                    'batch_size'  : 200,
                    'learn_rate'  : 1e-3,
                    'momentum'    : 0.9,
                    'param_scale' : 0.1,
                    'gamma'       : 0.9,
                    'lambda'      : 0.1}

    arch_params = {'num_hidden_features' : [50, 50],
                   'permutations' : False}

    task_params = {'N_train' : 10000,
                   'N_test' : 10000,
                   'target_name'  : 'Molecular Weight',
                   'data_file' : get_data_file('2014-11-03-all-tddft/processed.csv')}

    #target_name = 'Log Rate'
    #'Polar Surface Area'
    #'Number of Rings'
    #'Number of H-Bond Donors'
    #'Number of Rotatable Bonds'
    #'Minimum Degree'

    print train_params
    print arch_params
    print task_params

    print "Loading data..."
    traindata, testdata = load_data(task_params['data_file'], (task_params['N_train'], task_params['N_test']))
    train_inputs, train_targets = traindata['smiles'], traindata[task_params['target_name']]
    test_inputs, test_targets = testdata['smiles'], testdata[task_params['target_name']]

    print "-" * 80
    def print_performance(pred_func):
        train_preds = pred_func(train_inputs)
        test_preds = pred_func(test_inputs)
        #print "Performance (mean abs error):"
        #print "Train:", np.mean(np.abs(train_preds - train_targets))
        #print "Test: ", np.mean(np.abs(test_preds - test_targets))
        print "Performance (RMSE):"
        print "Train:", np.sqrt(np.mean((train_preds - train_targets)**2))
        print "Test: ", np.sqrt(np.mean((test_preds - test_targets)**2))
        print "-" * 80

    print "Mean predictor"
    y_train_mean = np.mean(train_targets)
    print_performance(lambda x : y_train_mean)

    print "Fingerprints with linear weights"
    predictor = fingerprints_linear_output(train_inputs, train_targets, arch_params, train_params)
    print_performance(predictor)

    print "Random net with linear weights"
    predictor = random_net_linear_output(train_inputs, train_targets, arch_params, train_params)
    print_performance(predictor)

    print "Training vanilla neural net"
    predictor = train_2layer_nn(train_inputs, train_targets)
    print_performance(predictor)

    print "Training custom neural net : array representation"
    with tictoc():
        predictor = train_universal_custom_nn(train_inputs, train_targets, arch_params, train_params)
    print_performance(predictor)


if __name__ == '__main__':
    sys.exit(main())
