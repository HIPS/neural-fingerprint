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

from load_data import load_molecules
from build_kayak_net_nodewise import initialize_weights, BuildNetFromSmiles
from build_kayak_net_arrayrep import build_universal_net
from util import tictoc, normalize_array, c_value, c_grad
from optimization_routines import sgd_with_momentum, make_batcher, batch_idx_generator

def train_2layer_nn(features, targets):
    batch_size   = 256
    learn_rate   = 0.001
    momentum     = 0.98
    h1_dropout   = 0.01
    h1_size      = 500
    dropout_prob = 0.1
    l1_weight    = 0.1
    l2_weight    = 0.1
    param_scale = 0.1
    normed_targets, undo_norm = normalize_array(targets)
    batcher = kayak.Batcher(batch_size, features.shape[0])
    X =  kayak.Inputs(features, batcher)
    T =  kayak.Targets(normed_targets, batcher)
    W1 = kayak.Parameter(param_scale * npr.randn(features.shape[1], h1_size))
    B1 = kayak.Parameter(param_scale * npr.randn(1, h1_size))
    H1 = kayak.Dropout(kayak.HardReLU(kayak.ElemAdd(kayak.MatMult(X, W1), B1)), h1_dropout, batcher=batcher)
    W2 = kayak.Parameter(param_scale * npr.randn(h1_size, 1))
    B2 = kayak.Parameter(param_scale * npr.randn(1, 1))
    Y =  kayak.ElemAdd(kayak.MatMult(H1, W2), B2)
    L =  kayak.L2Loss(Y, T)
    params = [W1, W2, B1, B2]
    step_dirs = [np.zeros(p.shape) for p in params]
    for epoch in xrange(num_epochs):
        total_err = 0.0
        for batch in batcher:
            total_err += np.sum(np.abs(undo_norm(Y.value) - undo_norm(T.value)))
            grads = [L.grad(p) for p in params]
            step_dirs = [momentum * step_dir - (1.0 - momentum) * grad
                         for step_dir, grad in zip(step_dirs, grads)]
            for p, step_dir in zip(params, step_dirs):
                p.value += learn_rate * step_dir
        print "Abs error after epoch", epoch, ":", total_err / len(normed_targets)

    def make_predictions(newvals):
        X.data = newvals
        batcher.test_mode()
        return undo_norm(Y.value)

    return make_predictions

def train_custom_nn(smiles, targets, arch_params, train_params):
    num_epochs = train_params['num_epochs']
    batch_size = train_params['batch_size']
    learn_rate = train_params['learn_rate']
    momentum = train_params['momentum']
    param_scale = train_params['param_scale']
    num_hidden_features = arch_params['num_hidden_features']

    npr.seed(1)
    # Training parameters:
    normed_targets, undo_norm = normalize_array(targets)
    # Learn the weights
    np_weights = initialize_weights(num_hidden_features, param_scale)
    step_dirs = {k: np.zeros(w.shape) for k, w in np_weights.iteritems()}
    batches = batch_idx_generator(batch_size, len(targets))
    for epoch in xrange(num_epochs):
        for batch in batches:
            grad = {k : 0.0 for k in np_weights}
            for smile, target in zip(smiles[batch], normed_targets[batch]):
                loss, k_weights, output = BuildNetFromSmiles(smile, np_weights, target)
                for key, cur_k_weights in k_weights.iteritems():
                    grad[key] += loss.grad(cur_k_weights)

            for key, cur_k_weights in k_weights.iteritems():
                step_dirs[key] = momentum * step_dirs[key] - (1.0 - momentum) * grad[key]
                np_weights[key] = np_weights[key] + learn_rate * step_dirs[key]

        total_loss = 0.0
        for smile, target in zip(smiles, normed_targets):
            loss, _, _ = BuildNetFromSmiles(smile, np_weights, target)
            total_loss += loss.value
            
        print "After epoch", epoch, "loss is", total_loss

    def make_predictions(smiles):
        predictions = []
        for smile in smiles:
            _, _, output = BuildNetFromSmiles(smile, np_weights, None)
            predictions.append(undo_norm(output.value[0]))
        return np.array(predictions)

    return make_predictions

def train_universal_custom_nn(smiles, raw_targets, arch_params, train_params):
    npr.seed(1)
    targets, undo_norm = normalize_array(raw_targets)
    loss_fun, grad_fun, pred_fun, N_weights = build_universal_net(**arch_params)
    def callback(epoch, weights):
        print "After epoch", epoch, "loss is", loss_fun(weights, smiles, targets)
    grad_fun_with_data = lambda idxs, w : grad_fun(w, smiles[idxs], targets[idxs])
    trained_weights = sgd_with_momentum(grad_fun_with_data, len(targets), N_weights,
                                        callback, **train_params)
    return lambda new_smiles : undo_norm(pred_fun(trained_weights, new_smiles))

def main():
    # datadir = '/Users/dkd/Dropbox/Molecule_ML/data/Samsung_September_8_2014/'
    datadir = '/home/dougal/Dropbox/Shared/Molecule_ML/data/Samsung_September_8_2014/'

    # trainfile = datadir + 'davids-validation-split/train_split.csv'
    # testfile  = datadir + 'davids-validation-split/test_split.csv'
    trainfile = datadir + 'davids-validation-split/tiny.csv'
    testfile  = datadir + 'davids-validation-split/tiny.csv'
    # trainfile = datadir + 'davids-validation-split/1k_set.csv'
    # testfile  = datadir + 'davids-validation-split/1k_set.csv'

    # Parameters for both custom nets
    train_params = {'num_epochs'  : 5,
                    'batch_size'  : 50,
                    'learn_rate'  : 1e-3,
                    'momentum'    : 0.9,
                    'param_scale' : 0.1}
    arch_params = {'num_hidden_features' : [50, 50],
                   'permutations' : False}

    print "Loading data..."
    traindata = load_molecules(trainfile, transform = np.log)
    testdata = load_molecules(testfile, transform = np.log)
    print "-" * 80
    def print_performance(pred_func, input_field):
        train_preds = pred_func(traindata[input_field])
        test_preds = pred_func(testdata[input_field])
        print "Performance (mean abs error):"
        print "Train:", np.mean(np.abs(train_preds - traindata['y']))
        print "Test: ", np.mean(np.abs(test_preds - testdata['y']))
        print "-" * 80

    print "Mean predictor"
    y_train_mean = np.mean(traindata['y'])
    print_performance(lambda x : y_train_mean, 'smiles')

    print "Training custom neural net : array representation"
    with tictoc():
        predictor = train_universal_custom_nn(
            traindata['smiles'], traindata['y'], arch_params, train_params)
    print_performance(predictor, 'smiles')

    print "Training custom neural net : linked node representation"
    with tictoc():
        predictor = train_custom_nn(
            traindata['smiles'], traindata['y'], arch_params, train_params)
    print_performance(predictor, 'smiles')

    # print "Training vanilla neural net"
    # predictor = train_2layer_nn(traindata['fingerprints'], traindata['y'])
    # print_performance(predictor, 'fingerprints')

if __name__ == '__main__':
    sys.exit(main())
