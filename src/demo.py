"""A test file to see if Kayak can be used without modification in order to build a custom computation graph given
a graph corresponding to a molecule.

Dougal Maclaurin
David Duvenaud

Sept 22nd, 2014"""

import sys
import numpy as np
import numpy.random as npr
# import util

from rdkit import Chem

sys.path.append('../Kayak/')
import kayak

from MolGraph import *
from features import *



def test_custom_net():

    num_features = 4
    num_data = 1;

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


def build_net_from_graph(graph):

    num_layers = 7
    num_dims   = 5

    for ii in xrange(NUM_TRIALS):
        probs = npr.rand()

        X = kayak.Inputs(npr.randn(25,num_dims))

        wts    = []
        layers = []
        for jj in xrange(num_layers):

            U = kayak.Constant(np.zeros((25,num_dims)))

            if npr.rand() < probs:
                W = kayak.Parameter(0.1*npr.randn(num_dims, num_dims))
                wts.append(W)
                U = kayak.MatAdd( U, kayak.SoftReLU(kayak.MatMult(X, W)) )

            for kk in xrange(jj):
                if npr.rand() < probs:
                    W = kayak.Parameter(0.1*npr.randn(num_dims, num_dims))
                    wts.append(W)
                    U = kayak.MatAdd( U, kayak.SoftReLU(kayak.MatMult(layers[kk], W)) )

            layers.append(U)

        out = kayak.MatSum(layers[-1])

        out.value(True)
        for jj, wt in enumerate(wts):
            diff = kayak.util.checkgrad(wt, out, 1e-4)
            print diff
            assert diff < 1e-4


def BuildGraphFromMolecule(mol):
    # Replicate the graph that RDKit produces.
    # Go on and extract features using RDKit also.

    graph = MolGraph()

    # Iterate over the atoms.
    rd_atoms = {}
    for atom in mol.GetAtoms():
        rd_atoms[atom.GetIdx()] = Vertex( data=atom_features(atom) )
        graph.add_vert( rd_atoms[atom.GetIdx()] )

    # Iterate over the bonds.
    for bond in mol.GetBonds():
        atom1 = bond.GetBeginAtom()
        atom2 = bond.GetEndAtom()

        graph.add_edge( Edge(rd_atoms[atom1.GetIdx()],
                             rd_atoms[atom2.GetIdx()],
                             data=bond_features(bond)) )

    return graph


def main():

    # Load in a molecule
    mol = Chem.MolFromSmiles('CC')

    # Build a graph from it
    graph = BuildGraphFromMolecule(mol)

    # Build a Kayak neural net from that molecule
    #net, weights = BuildNetFromGraph(graph)

    # Test network
    #print net.value(True)
    #print net.checkgrad()





if __name__ == '__main__':
    sys.exit(main())
