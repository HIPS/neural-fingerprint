import numpy as np
import numpy.random as npr
import  itertools as it
from rdkit.Chem import AllChem, MolFromSmiles
import kayak
from old_MolGraph import Vertex, Edge, MolGraph
from features import atom_features, bond_features, N_atom_features, N_bond_features

def initialize_weights(num_hidden_features, scale):
    num_layers = len(num_hidden_features)
    num_atom_features = N_atom_features
    num_edge_features = N_bond_features
    num_features = [num_atom_features] + num_hidden_features
    np_weights = {}
    for layer in range(num_layers):
        N_prev, N_next = num_features[layer], num_features[layer + 1]
        np_weights[('self', layer)]  = scale * npr.randn(N_prev, N_next)
        for degree in [1, 2, 3, 4]:
            np_weights[('neighbors', layer, degree)] = \
                scale * npr.randn(N_prev + num_edge_features, N_next)

    np_weights['out'] = scale * npr.randn(num_features[-1], 1)
    return np_weights

def BuildNetFromSmiles(smile, np_weights, target):
    mol = MolFromSmiles(smile)
    graph = BuildGraphFromMolecule(mol)
    return BuildNetFromGraph(graph, np_weights, target)

def BuildGraphFromMolecule(mol):
    graph = MolGraph()
    AllChem.Compute2DCoords(mol)    # Only for visualization.
    # Iterate over the atoms.
    rd_atoms = {}
    for atom in mol.GetAtoms():
        new_vert = Vertex()
        new_vert.nodes = [kayak.Inputs(atom_features(atom)[None,:])]
        pos = mol.GetConformer().GetAtomPosition(atom.GetIdx())
        new_vert.pos = (pos.x, pos.y)
        graph.add_vert( new_vert )
        rd_atoms[atom.GetIdx()] = new_vert

    # Iterate over the bonds.
    for bond in mol.GetBonds():
        atom1 = bond.GetBeginAtom()
        atom2 = bond.GetEndAtom()
        new_edge = Edge(rd_atoms[atom1.GetIdx()], rd_atoms[atom2.GetIdx()])
        new_edge.nodes = [kayak.Inputs(bond_features(bond)[None, :])]
        graph.add_edge(new_edge)

    return graph

def BuildNetFromGraph(graph, np_weights, target):
    # This first version just tries to emulate ECFP, with different weights on each layer
    k_weights = {key: kayak.Parameter(weights) for key, weights in np_weights.iteritems()}
    # Build concatenated sets of weights
    cat_weights = {}
    for layer in it.count():
        if ('self', layer) not in k_weights:
            num_layers = layer
            break

        for num_neighbors in [1, 2, 3, 4]:
            cat_weights[(layer, num_neighbors)] = kayak.Concatenate(0,
                k_weights[('self', layer)],
                *((k_weights[('neighbors', layer, num_neighbors)],) * num_neighbors))

    for layer in range(num_layers):
        # Every atom and edge is a separate Kayak Input. These inputs already live in the graph.
        for v in graph.verts:
            nodes_to_cat = [v.nodes[layer]]
            neighbors = zip(*v.get_neighbors()) # list of (node, edge) tuple
            num_neighbors = len(neighbors)
            for n, e in neighbors:
                nodes_to_cat.append(n.nodes[layer])
                nodes_to_cat.append(e.nodes[layer])
            cat_node = kayak.Concatenate(1, *nodes_to_cat)
            v.nodes.append(kayak.Logistic(kayak.MatMult(cat_node, cat_weights[(layer, num_neighbors)])))

        for e in graph.edges:
            e.nodes.append(kayak.Identity(e.nodes[layer]))

    # Connect everything to the fixed-size layer using some sort of max
    penultimate_nodes = [v.nodes[-1] for v in graph.verts]
    concatenated = kayak.Concatenate( 0, *penultimate_nodes)
    softmax_layer = kayak.SoftMax(concatenated, axis=0)
    output_layer = kayak.MatSum(kayak.MatElemMult(concatenated, softmax_layer), axis=0)

    # Perform a little more computation to get a single number.
    output = kayak.MatMult(output_layer, k_weights['out'])
    loss = kayak.L2Loss(output, kayak.Targets(target))
    return loss, k_weights, output
