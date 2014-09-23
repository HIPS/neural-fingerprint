import sys
import numpy as np
import numpy.random as npr
import util

from rdkit.Chem import AllChem, MolFromSmiles

sys.path.append('../Kayak/')
import kayak

num_folds    = 5
batch_size   = 256
num_epochs   = 1000
learn_rate   = 0.0001
momentum     = 0.9
dropout_prob = 0.1
l1_weight    = 1.0
l2_weight    = 1.0

def compute_features(mol, size=512, radius=2):
    return AllChem.GetMorganFingerprintAsBitVect(mol, radius, nBits=size)

def train(features, targets):

    batcher = kayak.Batcher(batch_size, features.shape[0])

    X = kayak.Inputs(features, batcher)
    T = kayak.Targets(targets, batcher)

    W = kayak.Parameter( 0.1*npr.randn( features.shape[1] ))
    B = kayak.Parameter( 0.0 )

    Y = kayak.ElemAdd(kayak.MatMult(X, W), B)

    L = kayak.MatSum(kayak.L2Loss(Y, T))

    mom_grad_W = np.zeros(W.shape())

    for epoch in xrange(num_epochs):

        total_loss = 0.0
        
        for batch in batcher:

            total_loss += L.value(True)

            grad_W = L.grad(W)
            grad_B = L.grad(B)

            mom_grad_W = momentum*mom_grad_W + (1.0-momentum)*grad_W

            W.add( -learn_rate * mom_grad_W )
            B.add( -learn_rate * grad_B )
        
        print epoch, total_loss
    
    return lambda x: Y.value(True, inputs={ X: x })

def main():
    filename = '~/Dropbox/Collaborations/MolecularML.shared/data/ML_exploit_1k/tddft_hyb_b3l_lifetime.csv'
    data     = util.load_csv(filename)
    features = np.array([compute_features(MolFromSmiles(mol['smiles'])) for mol in data])
    targets  = np.log(np.array([float(mol['rate']) for mol in data]))

    # Eliminate NaN values.
    good_indices = np.isfinite(targets)
    targets      = targets[good_indices]
    features     = features[good_indices,...]

    CV = kayak.CrossValidator(num_folds, features, targets)

    for ii, fold in enumerate(CV):
        print "Fold %d" % (ii+1)

        train_features, train_targets = fold.train()
        valid_features, valid_targets = fold.valid()

        # Normalize the outputs.
        targ_mean     = np.mean(train_targets)
        targ_std      = np.std(train_targets)
        train_targets = (train_targets - targ_mean) / targ_std

        pred_func = train(train_features, train_targets)

        valid_preds = pred_func( valid_features ) * targ_std + targ_mean
        print np.mean(np.abs(valid_preds-valid_targets))
        

if __name__ == '__main__':
    sys.exit(main())
