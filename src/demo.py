"""A test file to see if Kayak can be used without modification in order to build a custom computation graph given
a graph corresponding to a molecule.

Dougal Maclaurin
David Duvenaud

Sept 22nd, 2014"""

import sys
import numpy as np
import numpy.random as npr
import time

from rdkit import Chem
from rdkit.Chem import Draw
import rdkit.Chem.AllChem as AllChem

sys.path.append('../../Kayak/')
import kayak

from MolGraph import *
from features import *

from util import *
from build_kayak_net import *

def main():

    print "Parsing the molecule..."
    #mol = Chem.MolFromSmiles('CN1C=NC2=C1C(=O)N(C(=O)N2C)C')   # Caffeine
    mol = Chem.MolFromSmiles('CCN(CC)C(=O)[C@H]1CN([C@@H]2Cc3c[nH]c4c3c(ccc4)C2=C1)C')  # LSD

    print "Building the graph of the molecule..."
    graph = BuildGraphFromMolecule(mol)

    print "Building the custom neural net..."
    num_hidden_features = [2, 2, 2]      # Figure out how many features we have.
    tempgraph = BuildGraphFromMolecule(Chem.MolFromSmiles('C=C'))   # Hacky? You decide.
    num_atom_features = tempgraph.verts[0].nodes[0].shape[1]
    num_edge_features = tempgraph.edges[0].nodes[0].shape[1]
    num_features = [num_atom_features] + num_hidden_features
    num_layers = len(num_hidden_features)

    # Initialize the weights
    np_weights = {}
    for layer in range(num_layers):
        np_weights[('self', layer)] = 0.1*npr.randn(num_features[layer], num_features[layer + 1])
        np_weights[('other', layer)] = 0.1*npr.randn(num_features[layer], num_features[layer + 1])
        np_weights[('edge', layer)] = 0.1*npr.randn(num_edge_features, num_features[layer + 1])

    np_weights['out'] = 0.1*npr.randn(num_features[-1], 1)

    net, k_weights, _ = BuildNetFromGraph(graph,np_weights, 1., num_layers)

    print "Evaluating the network..."
    net.value

    drawComputationGraph(graph, net)
    print "Done"

if __name__ == '__main__':
    sys.exit(main())
