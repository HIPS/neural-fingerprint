# A quick comparison script to compare the predictive accuracy of using standard fingerprints versus custom convnets.
#
# Dougal Maclaurin
# David Duvenaud
# Ryan P. Adams
#
# Sept 25th, 2014

import sys
import time
from contextlib import contextmanager

import numpy as np
import numpy.random as npr
sys.path.append('../../Kayak/')
import kayak

from MolGraph import *
from features import *
from load_data import load_molecules
from build_kayak_net import *

def train_2layer_nn(features, targets):
    # Hyperparameters
    batch_size   = 256
    num_epochs   = 10
    learn_rate   = 0.001
    momentum     = 0.98
    h1_dropout   = 0.01
    h1_size      = 500
    dropout_prob = 0.1
    l1_weight    = 0.1
    l2_weight    = 0.1
    param_scale = 0.1

    # Normalize the outputs.
    targ_mean, targ_std = np.mean(targets), np.std(targets)

    # Build the kayak graph
    batcher = kayak.Batcher(batch_size, features.shape[0])
    X =  kayak.Inputs(features, batcher)
    T =  kayak.Targets((targets - targ_mean) / targ_std, batcher)
    W1 = kayak.Parameter(param_scale * npr.randn(features.shape[1], h1_size))
    B1 = kayak.Parameter(param_scale * npr.randn(1, h1_size))
    H1 = kayak.Dropout(kayak.HardReLU(kayak.ElemAdd(kayak.MatMult(X, W1), B1)),
                       h1_dropout, batcher=batcher)
    W2 = kayak.Parameter(param_scale * npr.randn(h1_size, 1))
    B2 = kayak.Parameter(param_scale * npr.randn(1, 1))
    Y =  kayak.ElemAdd(kayak.MatMult(H1, W2), B2)
    L =  kayak.L2Loss(Y, T)

    # Train the parameters
    params = [W1, W2, B1, B2]
    step_dirs = [np.zeros(p.shape) for p in params]
    for epoch in xrange(num_epochs):
        total_err = 0.0
        for batch in batcher:
            total_err += np.sum(np.abs(Y.value - T.value)) * targ_std
            grads = [L.grad(p) for p in params]
            step_dirs = [momentum * step_dir - (1.0 - momentum) * grad
                         for step_dir, grad in zip(step_dirs, grads)]
            for p, step_dir in zip(params, step_dirs):
                p.value += learn_rate * step_dir

        print epoch, total_err / len(targets)

    def make_predictions(newvals):
        X.data = newvals
        batcher.test_mode()
        return Y.value * targ_std + targ_mean

    return make_predictions

def train_custom_nn(smiles, targets, num_hidden_features = [100, 100]):
    npr.seed(1)
    # Training parameters:
    learn_rate = 1e-4
    momentum = 0.99
    param_scale = 0.1
    num_epochs = 10
    # Normalize the outputs.
    targ_mean, targ_std = np.mean(targets), np.std(targets)
    normed_targets = (targets - targ_mean) / targ_std
    # Learn the weights
    np_weights = initialize_weights(num_hidden_features, param_scale)
    step_dirs = {k: np.zeros(w.shape) for k, w in np_weights.iteritems()}
    for epoch in xrange(num_epochs):
        total_err = 0.0
        for smile, target in zip(smiles, normed_targets):
            loss, k_weights, output = BuildNetFromSmiles(smile, np_weights, target)
            total_err += abs(output.value - target) * targ_std

            for key, cur_k_weights in k_weights.iteritems():
                grad = loss.grad(cur_k_weights)
                step_dirs[key] = momentum * step_dirs[key] - (1.0 - momentum) * grad
                np_weights[key] = np_weights[key] + learn_rate * step_dirs[key]

        print "Current abs error after epoch", epoch, ":", total_err / len(normed_targets)
    print "Finished training"

    def make_predictions(smiles):
        predictions = []
        for smile in smiles:
            _, _, output = BuildNetFromSmiles(smile, np_weights, None)
            predictions.append(output.value * targ_std + targ_mean)
        return predictions

    return make_predictions

def train_universal_custom_nn(smiles, targets):
    npr.seed(1)
    # Parameters
    num_hidden_features = [100, 100]
    learn_rate = 1e-4
    momentum = 0.99
    param_scale = 0.1
    num_epochs = 10
    # Normalize the outputs
    targ_mean, targ_std = np.mean(targets), np.std(targets)
    normed_targets = (targets - targ_mean) / targ_std
    # Build the net
    mol, target, loss, output, k_weights = \
        build_universal_net(num_hidden_features, param_scale)
    # Now train it
    step_dirs = [np.zeros(w.shape) for w in k_weights]
    mol_graphs = [arrayrep_from_smiles(smile) for smile in smiles]
    for epoch in xrange(num_epochs):
        total_err = 0.0
        for graph, target_val in zip(mol_graphs, normed_targets):
            load_new_input(mol, graph)
            target.data = target_val
            total_err += abs(output.value - target_val) * targ_std

            step_dirs = [momentum * d - (1.0 - momentum) * loss.grad(w)
                         for d, w in zip(step_dirs, k_weights)]
            for w, d in zip(k_weights, step_dirs):
                w.value += learn_rate * d

        print "Current abs error after epoch", ":", total_err / len(normed_targets)
    print "Finished training"

    def make_predictions(smiles):
        predictions = []
        for smile in smiles:
            load_new_input(mol, arrayrep_from_smiles(smile))
            predictions.append(output.value * targ_std + targ_mean)
        return predictions

    return make_predictions

@contextmanager
def tictoc():
    print "--- Start clock ---"
    t1 = time.time()
    yield
    dt = time.time() - t1
    print "--- Stop clock: %s seconds elapsed ---" % dt

def main():
    # datadir = '/Users/dkd/Dropbox/Molecule_ML/data/Samsung_September_8_2014/'
    datadir = '/home/dougal/Dropbox/Shared/Molecule_ML/data/Samsung_September_8_2014/'

    # trainfile = datadir + 'davids-validation-split/train_split.csv'
    # testfile  = datadir + 'davids-validation-split/test_split.csv'
    trainfile = datadir + 'davids-validation-split/tiny.csv'
    testfile  = datadir + 'davids-validation-split/tiny.csv'
    # trainfile = datadir + 'davids-validation-split/1k_set.csv'
    # testfile  = datadir + 'davids-validation-split/1k_set.csv'

    def print_performance(pred_func, input_field):
        train_preds = pred_func(traindata[input_field])
        test_preds = pred_func(testdata[input_field])
        print "Performance (mean abs error):"
        print "Train:", np.mean(np.abs(train_preds-traindata['y']))
        print "Test: ", np.mean(np.abs(test_preds-testdata['y']))
        print "-" * 80

    print "Loading data..."
    traindata = load_molecules(trainfile, transform = np.log)
    testdata = load_molecules(testfile, transform = np.log)
    print "-" * 80

    print "Mean predictor"
    y_train_mean = np.mean(traindata['y'])
    print_performance(lambda x : y_train_mean, 'smiles')

    print "Training custom neural net : array representation"
    with tictoc():
        predictor = train_universal_custom_nn(traindata['smiles'], traindata['y'])
    print_performance(predictor, 'smiles')

    print "Training custom neural net : linked node representation"
    with tictoc():
        predictor = train_custom_nn(traindata['smiles'], traindata['y'])
    print_performance(predictor, 'smiles')

    print "Training vanilla neural net"
    predictor = train_2layer_nn(traindata['fingerprints'], traindata['y'])
    print_performance(predictor, 'fingerprints')

if __name__ == '__main__':
    sys.exit(main())
