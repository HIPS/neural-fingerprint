# A quick comparison script to compare the predictive accuracy of using standard fingerprints versus custom convnets.
#
# Dougal Maclaurin
# David Duvenaud
# Ryan P. Adams
#
# Sept 25th, 2014

import sys
import time

import numpy as np
import numpy.random as npr
from rdkit.Chem import AllChem, MolFromSmiles
sys.path.append('../../Kayak/')
import kayak

from MolGraph import *
from features import *
from load_data import load_molecules
from build_kayak_net import *

num_folds    = 2
batch_size   = 256
num_epochs   = 5
learn_rate   = 0.001
momentum     = 0.95
h1_dropout   = 0.1
h1_size      = 500

dropout_prob = 0.1
l1_weight    = 1.0
l2_weight    = 1.0

def train_2layer_nn(features, targets):
    # Normalize the outputs.
    targ_mean = np.mean(targets)
    targ_std  = np.std(targets)

    batcher = kayak.Batcher(batch_size, features.shape[0])

    X = kayak.Inputs(features, batcher)
    T = kayak.Targets((targets[:,None] - targ_mean) / targ_std, batcher)

    W1 = kayak.Parameter( 0.1*npr.randn( features.shape[1], h1_size ))
    B1 = kayak.Parameter( 0.1*npr.randn( 1, h1_size ) )
    H1 = kayak.Dropout(kayak.HardReLU(kayak.ElemAdd(kayak.MatMult(X, W1), B1)), h1_dropout, batcher=batcher)

    W2 = kayak.Parameter( 0.1*npr.randn( h1_size, 1 ) )
    B2 = kayak.Parameter( 0.1*npr.randn(1,1))

    Y = kayak.ElemAdd(kayak.MatMult(H1, W2), B2)

    L = kayak.MatSum(kayak.L2Loss(Y, T))

    mom_grad_W1 = np.zeros(W1.shape)
    mom_grad_W2 = np.zeros(W2.shape)

    for epoch in xrange(num_epochs):
        total_loss = 0.0
        total_err  = 0.0
        total_data = 0
        
        for batch in batcher:
            H1.draw_new_mask()

            total_loss += L.value
            total_err  += np.sum(np.abs(Y.value - T.value))
            total_data += T.shape[0]

            grad_W1 = L.grad(W1)
            grad_B1 = L.grad(B1)
            grad_W2 = L.grad(W2)
            grad_B2 = L.grad(B2)

            mom_grad_W1 = momentum*mom_grad_W1 + (1.0-momentum)*grad_W1
            mom_grad_W2 = momentum*mom_grad_W2 + (1.0-momentum)*grad_W2

            W1.value += -learn_rate * mom_grad_W1
            W2.value += -learn_rate * mom_grad_W2
            B1.value += -learn_rate * grad_B1
            B2.value += -learn_rate * grad_B2
        
        print epoch, total_err / total_data

    def make_predictions(newvals):
        X.data = newvals
        batcher.test_mode()
        return Y.value*targ_std + targ_mean

    return make_predictions

def train_custom_nn(smiles, targets, num_hidden_features = [100, 100]):
    # Figure out how many features and layers we have.
    num_layers = len(num_hidden_features)
    num_atom_features = atom_features()
    num_edge_features = bond_features()
    num_features = [num_atom_features] + num_hidden_features

    # Initialize the weights
    np_weights = {}
    for layer in range(num_layers):
        np_weights[('self', layer)] = 0.1*npr.randn(num_features[layer], num_features[layer + 1])
        np_weights[('other', layer)] = 0.1*npr.randn(num_features[layer], num_features[layer + 1])
        np_weights[('edge', layer)] = 0.1*npr.randn(num_edge_features, num_features[layer + 1])

    np_weights['out'] = 0.1*npr.randn(num_features[-1], 1)

    # Normalize the outputs.
    targ_mean = np.mean(targets)
    targ_std  = np.std(targets)
    normed_targets = (targets - targ_mean) / targ_std

    # Learn the weights
    learn_rate = 1e-7
    num_epochs = 5
    # TODO: implement RMSProp or whatever.
    print "\nTraining parameters",
    for epoch in xrange(num_epochs):
        total_loss = 0
        for smile, target in zip(smiles, normed_targets):
            mol = Chem.MolFromSmiles(smile)
            graph = BuildGraphFromMolecule(mol)
            loss, k_weights, _ = BuildNetFromGraph(graph, np_weights, target, num_layers)
            # Loop over kayak parameter vectors and evaluate the gradient w.r.t. each one.
            # Take a step on all parameters.
            for key, cur_k_weights in k_weights.iteritems():
                # Set weights value to None so that the weights arrays can be garbage-collected
                cur_k_weights.value = np_weights[key]
            total_loss += loss.value
            for key, cur_k_weights in k_weights.iteritems():
                np_weights[key] -= loss.grad(cur_k_weights) * learn_rate

        print "Current loss after epoch", epoch, ":", total_loss

    print "Finished training"

    def make_predictions(smiles):
        predictions = []
        for smile in smiles:
            mol = Chem.MolFromSmiles(smile)
            graph = BuildGraphFromMolecule(mol)
            _, _, output = BuildNetFromGraph(graph, np_weights, None, num_layers)
            predictions.append(output.value*targ_std + targ_mean)
        return predictions

    return make_predictions

def main():
    # datadir = '/Users/dkd/Dropbox/Molecule_ML/data/Samsung_September_8_2014/'
    datadir = '/home/dougal/Dropbox/Shared/Molecule_ML/data/Samsung_September_8_2014/'

    # trainfile = datadir + 'davids-validation-split/train_split.csv'
    # testfile = datadir + 'davids-validation-split/test_split.csv'
    trainfile = datadir + 'davids-validation-split/tiny.csv'
    testfile = datadir + 'davids-validation-split/tiny.csv'
    # trainfile = datadir + 'davids-validation-split/1k_set.csv'
    # testfile = datadir + 'davids-validation-split/1k_set.csv'

    print "Loading training data..."
    traindata = load_molecules(trainfile, transform = np.log)

    print "Loading test data..."
    testdata = load_molecules(testfile, transform = np.log)

    # Custom Neural Net
    pred_func_custom = train_custom_nn(traindata['smiles'], traindata['y'])
    train_preds = pred_func_custom( traindata['smiles'] )
    test_preds = pred_func_custom( testdata['smiles'] )
    print "Custom net test performance (train/test mean abs error):"
    print np.mean(np.abs(train_preds-traindata['y'])), np.mean(np.abs(test_preds-testdata['y']))

    # Vanilla Neural Net
    pred_func_vanilla = train_2layer_nn(traindata['fingerprints'], traindata['y'])
    train_preds = pred_func_vanilla( traindata['fingerprints'] )
    test_preds = pred_func_vanilla( testdata['fingerprints'] )
    print "Vanilla net test performance (train/test mean abs error): "
    print np.mean(np.abs(train_preds-traindata['y'])), np.mean(np.abs(test_preds-testdata['y']))

    print "Mean predictor performance abs error (train/test):"
    train_mean = np.mean(traindata['y'])
    print np.mean(np.abs(train_mean-traindata['y'])), np.mean(np.abs(train_mean-testdata['y']))

if __name__ == '__main__':
    sys.exit(main())
