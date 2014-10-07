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
import pandas as pd

from rdkit.Chem import AllChem, MolFromSmiles

sys.path.append('../../Kayak/')
import kayak

from MolGraph import *
from features import *


num_folds    = 2
batch_size   = 256
num_epochs   = 10
learn_rate   = 0.001
momentum     = 0.95
h1_dropout   = 0.1
h1_size      = 500

dropout_prob = 0.1
l1_weight    = 1.0
l2_weight    = 1.0


def compute_features(mol, size=512, radius=2):
    return AllChem.GetMorganFingerprintAsBitVect(mol, radius, nBits=size)

def train_2layer_nn(features, targets):
    
    # Normalize the outputs.
    targ_mean = np.mean(targets)
    targ_std  = np.std(targets)

    batcher = kayak.Batcher(batch_size, features.shape[0])

    X = kayak.Inputs(features, batcher)
    T = kayak.Targets((targets[:,None] - targ_mean) / targ_std, batcher)

    W1 = kayak.Parameter( 0.1*npr.randn( features.shape[1], h1_size ))
    B1 = kayak.Parameter( 0.1*npr.randn( 1, h1_size ) )
    H1 = kayak.Dropout(kayak.HardReLU(kayak.ElemAdd(kayak.MatMult(X, W1), B1)), h1_dropout)

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
        X.value = newvals
        H1.reinstate_units()
        return Y.value*targ_std + targ_mean

    return make_predictions



def stringlist2intarray(A):
    '''This function will convert from a list of strings "10010101" into in integer numpy array '''
    return np.array([list(s) for s in A], dtype=int)


def load_molecules(filename, test=False, predict_field='rate', transform=None, cutoff = None):

    data = pd.io.parsers.read_csv(filename, sep=',')  # Load the data.
    fingerprints = [smile_to_fp(s) for s in list(data['smiles'])]

    alldata = {}
    alldata['x'] = stringlist2intarray(fingerprints)  # N by D
    alldata['inchi_key'] = data['inchi_key']

    if test == False:
        alldata['y'] = np.array(transform(data[predict_field]))[:, None]  # Load targets

        if cutoff:
            above_cutoff = (alldata['y'] > cutoff) & np.isfinite(alldata['y'])
        else:
            above_cutoff = np.isfinite(alldata['y'])
        alldata = slicedict(alldata, above_cutoff)
        print "Datapoints thrown away:", np.sum(np.logical_not(above_cutoff))
        print "Datapoints kept:", np.sum(above_cutoff)

    return alldata


def slicedict(d, ixs):
    newd = {}
    for key in d:
        newd[key] = d[key][ixs.flatten()]
    return newd

def smile_to_fp(s):
    fplength = 512
    radius = 4
    m = Chem.MolFromSmiles(s)
    return (AllChem.GetMorganFingerprintAsBitVect(m, radius,
                                                  nBits=fplength)).ToBitString()


def main():

    datadir = '/Users/dkd/Dropbox/Molecule_ML/data/Samsung_September_8_2014/'

    trainfile = datadir + 'davids-validation-split/train_split.csv'
    # trainfile = datadir + 'davids-validation-split/train_split_plus_methyls.csv'
    testfile = datadir + 'davids-validation-split/test_split.csv'

    #filename = '~/Desktop/tddft_hyb_b3l_agatha_harsh_process_postprocess_lifetime.csv'
    # filename = '~/Dropbox/Collaborations/MolecularML.shared/data/ML_exploit_1k/tddft_hyb_b3l_lifetime.csv'
    #data     = util.load_csv(filename)
    #features = np.array([compute_features(MolFromSmiles(mol['smiles'])) for mol in data])
    #targets  = np.log(np.array([float(mol['rate']) for mol in data]))

    # Eliminate NaN values.
    #good_indices = np.isfinite(targets)
    #targets      = targets[good_indices]
    #features     = features[good_indices,...]

    print "Loading training data..."
    traindata = load_molecules(trainfile, transform = np.log)

    # Vanilla neural net predictions, without any transform.
    #    pred_func = nn_train(traindata['x'], traindata['y'][:,0], params=params)
    features = traindata['x']
    targets = traindata['y'][:,0]

    CV = kayak.CrossValidator(num_folds, features, targets)

    for ii, fold in enumerate(CV):
        print "Fold %d" % (ii+1)

        train_features, train_targets = fold.train()
        valid_features, valid_targets = fold.valid()

        pred_func1 = train_2layer_nn(train_features, train_targets)

        train_preds = pred_func1( train_features )
        valid_preds = pred_func1( valid_features )
        print np.mean(np.abs(train_preds-train_targets)), np.mean(np.abs(valid_preds-valid_targets))
        

if __name__ == '__main__':
    sys.exit(main())
