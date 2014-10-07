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

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


def BuildNetFromGraph(graph, num_hidden_features = [5, 6]):

    # This first version just tries to emulate ECFP.
    # Different weights on each layer

    num_layers = len(num_hidden_features)
    num_atom_features = graph.verts[0].nodes[0].shape[1]
    num_edge_features = graph.edges[0].nodes[0].shape[1]

    # Every atom and edge is a separate Kayak Input.
    # These inputs already live in the graph.

    W_self = []
    W_other = []
    W_edge = []

    for layer in range(num_layers):

        num_prev_layer_features = graph.verts[0].nodes[layer].shape[1]

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

    AllChem.Compute2DCoords(mol)    # Only for visualization.


    # Iterate over the atoms.
    rd_atoms = {}
    for atom in mol.GetAtoms():
        rd_atoms[atom.GetIdx()] = Vertex( nodes = [kayak.Inputs(atom_features(atom)[None,:])] )
        new_vert = rd_atoms[atom.GetIdx()]
        pos = mol.GetConformer().GetAtomPosition(atom.GetIdx())
        new_vert.pos = (pos.x, pos.y)
        graph.add_vert( new_vert )

    # Iterate over the bonds.
    for bond in mol.GetBonds():
        atom1 = bond.GetBeginAtom()
        atom2 = bond.GetEndAtom()

        graph.add_edge( Edge(rd_atoms[atom1.GetIdx()],
                             rd_atoms[atom2.GetIdx()],
                             nodes=[kayak.Inputs(bond_features(bond)[None, :])] ))

    return graph


# Define the positions of a node.
def position(k_node, node_positions):
    """node is a kayak node."""
    if k_node in node_positions:
        return node_positions[k_node]
    else:
        parents_positions = [position(p, node_positions)
                             for p in k_node._parents
                             if not isinstance(p, (kayak.Parameter, kayak.Targets))]
        (xs, ys, zs) = zip(*parents_positions)
        new_x = np.mean(xs)
        new_y = np.mean(ys)
        new_z = np.max(zs) + 1
        node_positions[k_node] = (new_x, new_y, new_z)
        return position(k_node, node_positions)

def drawComputationGraph(mol_graph, target, mol):
    """Graph already contains a list of vertices, each one has a Kayak Input,
    which knows about its children.

    Graph is the mol graph.
    """
    num_layers = 3
    # Make a dict indexed by kayak nodes that has the location of each atom as values.
    # First fix the locations of the atoms and bonds
    node_positions = {}
    base_nodes = mol_graph.get_verts()
    for layer in range(num_layers):
        for k_n, mol_v in [(mol_v.nodes[layer], mol_v) for mol_v in base_nodes]:
            node_positions[k_n] = mol_v.pos + (layer * 6, )
    base_edges = mol_graph.get_edges()
    for mol_e in base_edges:
        mol_v1, mol_v2 = mol_e.get_verts()
        avg_x = (mol_v1.pos[0] + mol_v2.pos[0]) / 2.0
        avg_y = (mol_v1.pos[1] + mol_v2.pos[1]) / 2.0
        avg_z = 0
        node_positions[mol_e.nodes[0]] = (avg_x, avg_y, avg_z)

    # Now plot all edges of interest given these positions.
    ax = plt.figure().add_subplot(111, projection = '3d')
    for n1, n2 in find_all_edges_leading_to(target):
        if not isinstance(n1, (kayak.Parameter, kayak.Targets)) and not isinstance(n2, (kayak.Parameter, kayak.Targets)):
            pos1 = position(n1, node_positions)
            pos2 = position(n2, node_positions)
            ax.plot(*(zip(pos1, pos2)), color="Grey", lw=1)

    # Finally, plot the edges corresponding to the molecule itself
    for layer in range(num_layers):
        for e in mol_graph.get_edges():
            (v1, v2) = e.get_verts()
            pos1 = position(v1.nodes[layer], node_positions)
            pos2 = position(v2.nodes[layer], node_positions)
            ax.plot(*zip(pos1, pos2), color="Black", lw=3)

    plt.show()


def find_all_edges_leading_to(k_target):
    # TODO: Move into Kayak.
    found = {}
    find_all_edges_internal(k_target, found)
    return found

def find_all_edges_internal(k_node, found):
    # Works backwards to find all edges of parents leading to the current node.
    for p in k_node._parents:
        if (p, k_node) not in found:
            find_all_edges_internal(p, found)
        found[(p, k_node)] = None
    return


def main():

    # Load in a molecule
    print "Parsing the molecule..."
    #mol = Chem.MolFromSmiles('CC')   # Caffeine
    mol = Chem.MolFromSmiles('CN1C=NC2=C1C(=O)N(C(=O)N2C)C')   # Caffeine
    #mol = Chem.MolFromSmiles('CCN(CC)C(=O)[C@H]1CN([C@@H]2Cc3c[nH]c4c3c(ccc4)C2=C1)C')  # LSD

    # Build a graph from it
    print "Building the graph of the molecule..."
    start = time.clock()
    graph = BuildGraphFromMolecule(mol)

    # Build a Kayak neural net from that molecule
    print "Building the custom neural net..."
    net, weights = BuildNetFromGraph(graph, num_hidden_features = [2, 2, 2])
    print "Time elapsed:", time.clock() - start

    print "Evaluating the network..."
    start = time.clock()
    net.value
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

    # Visualize the resulting neural net
    #fig = Draw.MolToMPL(mol)
    #fig.show()
    #Draw.MolToFile(mol, 'molecule-image.png', size=(1000,1000))

    # m = Chem.MolFromSmiles('c1ccccc1')
    # AllChem.Compute2DCoords(m)
    # pos = m.GetConformer().GetAtomPosition(0)

    drawComputationGraph(graph, net, mol)
    print "Done"






if __name__ == '__main__':
    sys.exit(main())
