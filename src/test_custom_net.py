"""A test file to see if Kayak can be used without modification in order to build a custom computation graph given
a graph corresponding to a molecule.

Dougal Maclaurin
David Duvenaud

Sept 22nd, 2014"""

import sys
import numpy as np
import numpy.random as npr
import util

from rdkit.Chem import AllChem, MolFromSmiles

sys.path.append('../Kayak/')
import kayak

num_folds    = 5
batch_size   = 256
num_epochs   = 50
learn_rate   = 0.001
momentum     = 0.95
h1_dropout   = 0.1
h1_size      = 500

dropout_prob = 0.1
l1_weight    = 1.0
l2_weight    = 1.0

num_features = 4
num_data = 1;

def main():

    X1 = kayak.Inputs( npr.randn(  num_data, num_features ) )
    X2 = kayak.Inputs( npr.randn(  num_data, num_features ) )

    print X1.value()
    print X2.value()

    T = kayak.Targets(npr.randn( num_data ))

    Wself = kayak.Parameter( npr.randn( num_features, num_features ))
    Wother = kayak.Parameter( npr.randn( num_features, num_features ))

    B = kayak.Parameter( npr.randn( 1, num_features ))
    H11 = kayak.HardReLU(kayak.ElemAdd(kayak.ElemAdd(kayak.MatMult(X1, Wself), B), kayak.ElemAdd(kayak.MatMult(X2, Wother), B)))
    H12 = kayak.HardReLU(kayak.ElemAdd(kayak.ElemAdd(kayak.MatMult(X2, Wself), B), kayak.ElemAdd(kayak.MatMult(X1, Wother), B)))
    H21 = kayak.HardReLU(kayak.ElemAdd(kayak.ElemAdd(kayak.MatMult(H11, Wself), B), kayak.ElemAdd(kayak.MatMult(H12, Wother), B)))
    H22 = kayak.HardReLU(kayak.ElemAdd(kayak.ElemAdd(kayak.MatMult(H12, Wself), B), kayak.ElemAdd(kayak.MatMult(H11, Wother), B)))

    WLast = kayak.Parameter( npr.randn( num_features ))
    BLast = kayak.Parameter( 0.1*npr.randn(1))
    Y = kayak.ElemAdd(kayak.ElemAdd(kayak.MatMult(H21, WLast), BLast), kayak.ElemAdd(kayak.MatMult(H22, WLast), BLast))

    L = kayak.MatSum(kayak.L2Loss(Y, T))
    grad_W1 = L.grad(Wself)
    print grad_W1

    first = L.value(True)
    delta = 0.00001
    delta_mat = np.zeros( (num_features, num_features))
    delta_mat[1,1] = delta
    Wself.add(delta_mat)

    second = L.value(True)
    print "Numerical derivative:", (second - first) / delta



if __name__ == '__main__':
    sys.exit(main())

    X = kayak.Inputs(features, batcher)
    T = kayak.Targets((targets-targ_mean) / targ_std, batcher)

    W1 = kayak.Parameter( 0.1*npr.randn( features.shape[1], h1_size ))
    B1 = kayak.Parameter( 0.1*npr.randn( 1, h1_size ) )
    H1 = kayak.Dropout(kayak.HardReLU(kayak.ElemAdd(kayak.MatMult(X, W1), B1)), h1_dropout)

    W2 = kayak.Parameter( 0.1*npr.randn( h1_size ) )
    B2 =     X = kayak.Inputs(features, batcher)
    T = kayak.Targets((targets-targ_mean) / targ_std, batcher)

    W1 = kayak.Parameter( 0.1*npr.randn( features.shape[1], h1_size ))
    B1 = kayak.Parameter( 0.1*npr.randn( 1, h1_size ) )
    H1 = kayak.Dropout(kayak.HardReLU(kayak.ElemAdd(kayak.MatMult(X, W1), B1)), h1_dropout)

    W2 = kayak.Parameter( 0.1*npr.randn( h1_size ) )
    B2 = kayak.Parameter( 0.1*npr.randn(1))

    Y = kayak.ElemAdd(kayak.MatMult(H1, W2), B2)

    L = kayak.MatSum(kayak.L2Loss(Y, T))

    Y = kayak.ElemAdd(kayak.MatMult(H1, W2), B2)

    L = kayak.MatSum(kayak.L2Loss(Y, T))