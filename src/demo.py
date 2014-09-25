"""A test file to see if Kayak can be used without modification in order to build a custom computation graph given
a graph corresponding to a molecule.

Dougal Maclaurin
David Duvenaud

Sept 22nd, 2014"""

import sys
import numpy as np
import numpy.random as npr
# import util
import time

from rdkit import Chem

sys.path.append('../../Kayak/')
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


def BuildNetFromGraph(graph, num_hidden_features = [5, 6]):

    # This first version just tries to emulate ECFP.
    # Different weights on each layer

    num_layers = len(num_hidden_features)
    num_atom_features = graph.verts[0].nodes[0].shape()[1]
    num_edge_features = graph.edges[0].nodes[0].shape()[1]

    # Every atom and edge is a separate Kayak Input.
    # These inputs already live in the graph.

    W_self = []
    W_other = []
    W_edge = []

    for layer in range(num_layers):

        num_prev_layer_features = graph.verts[0].nodes[layer].shape()[1]

        # Create a Kayak parameter for this layer
        W_self.append(kayak.Parameter(0.1*npr.randn(num_prev_layer_features, num_hidden_features[layer])))
        # possible refinement: separate weights for each connection, max-pooled over all permutations
        W_other.append(kayak.Parameter(0.1*npr.randn(num_prev_layer_features, num_hidden_features[layer])))
        W_edge.append(kayak.Parameter(0.1*npr.randn(num_edge_features, num_hidden_features[layer])))

        for v in graph.verts:
            # Create a Differentiable node N that depends on the corresponding node in the previous layer, its edges,
            # and its neighbours.
            mults = [kayak.MatMult(v.nodes[layer], W_self[layer])]
            for e in v.edges:
                mults.append(kayak.MatMult( e.nodes[0], W_edge[layer]))
            for n in v.get_neighbors()[0]:
                mults.append(kayak.MatMult( n.nodes[layer], W_other[layer]))

            # Add the next layer of computation to this node.
            v.nodes.append(kayak.SoftReLU(kayak.ElemAdd(*mults)))

    # Connect everything to the fixed-size layer using some sort of max
    penultimate_nodes = [v.nodes[-1] for v in graph.verts]
    concatenated = kayak.Concatenate( 0, *penultimate_nodes)
    output_layer = kayak.MatSum( concatenated, 0)

    # Perform a little more computation to get a single number.
    W_out = kayak.Parameter(0.1*npr.randn(num_hidden_features[-1], 1))
    output = kayak.MatMult(output_layer, W_out)

    target = kayak.Targets(np.array([[1.23]]));
    loss = kayak.L2Loss( output, target)

    weights = W_self + W_other + W_edge + [W_out]

    return loss, weights


def BuildGraphFromMolecule(mol):
    # Replicate the graph that RDKit produces.
    # Go on and extract features using RDKit also.

    graph = MolGraph()

    # Iterate over the atoms.
    rd_atoms = {}
    for atom in mol.GetAtoms():
        rd_atoms[atom.GetIdx()] = Vertex( nodes = [kayak.Inputs(atom_features(atom)[None,:])] )
        graph.add_vert( rd_atoms[atom.GetIdx()] )

    # Iterate over the bonds.
    for bond in mol.GetBonds():
        atom1 = bond.GetBeginAtom()
        atom2 = bond.GetEndAtom()

        graph.add_edge( Edge(rd_atoms[atom1.GetIdx()],
                             rd_atoms[atom2.GetIdx()],
                             nodes=[kayak.Inputs(bond_features(bond)[None, :])] ))

    return graph


def main():

    # Load in a molecule
    print "Parsing the molecule..."
    mol = Chem.MolFromSmiles('CN1C=NC2=C1C(=O)N(C(=O)N2C)C')   # Caffeine
    mol = Chem.MolFromSmiles('CCN(CC)C(=O)[C@H]1CN([C@@H]2Cc3c[nH]c4c3c(ccc4)C2=C1)C')  # LSD

    # Build a graph from it
    print "Building the graph of the molecule..."
    start = time.clock()
    graph = BuildGraphFromMolecule(mol)

    # Build a Kayak neural net from that molecule
    print "Building the custom neural net..."
    net, weights = BuildNetFromGraph(graph, num_hidden_features = [10, 10, 10, 10, 10, 10, 10])
    print "Time elapsed:", time.clock() - start

    print "Evaluating the network..."
    start = time.clock()
    net.value()
    print "Time elapsed:", time.clock() - start

    print "Evaluate the gradient of the network..."
    start = time.clock()
    net.grad(weights[0])
    print "Time elapsed:", time.clock() - start

    # Test network
    #print "Checking gradients..."
    #print kayak.util.checkgrad(weights[0], net, 1e-4)
    #for jj, wt in enumerate(weights):
    #    diff = kayak.util.checkgrad(wt, net, 1e-4)
    #    print diff
    #    assert diff < 1e-4

if __name__ == '__main__':
    sys.exit(main())
