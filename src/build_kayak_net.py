
import sys
import numpy as np
import numpy.random as npr
from rdkit.Chem import AllChem, MolFromSmiles
sys.path.append('../../Kayak/')
import kayak
from MolGraph import *
from features import *
from load_data import load_molecules


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

def BuildNetFromGraph(graph, np_weights, target, num_layers):
    # This first version just tries to emulate ECFP, with different weights on each layer

    # Dict comprehension to loop over layers and types.
    k_weights = {key: kayak.Parameter(weights) for key, weights in np_weights.iteritems()}
    # Build concatenated sets of weights
    cat_weights = {}
    for layer in range(num_layers):
        cur_cat = k_weights[('self', layer)]
        for num_neighbors in [1, 2, 3, 4]:
            cur_cat = kayak.Concatenate(0, cur_cat,
                                        k_weights[('other', layer)],
                                        k_weights[('edge', layer)])
            cat_weights[(layer, num_neighbors)] = cur_cat

    for layer in range(num_layers):
        # Every atom and edge is a separate Kayak Input. These inputs already live in the graph.
        for v in graph.verts:
            # Create a Differentiable node N that depends on the corresponding node in the previous layer, its edges,
            # and its neighbours.
            # First we'll concatenate all the input nodes:
            nodes_to_cat = [v.nodes[layer]]
            neighbors = zip(*v.get_neighbors()) # list of (node, edge) tuple
            num_neighbors = len(neighbors)
            for n, e in neighbors:
                nodes_to_cat.append(n.nodes[layer])
                nodes_to_cat.append(e.nodes[layer])
            cat_node = kayak.Concatenate(1, *nodes_to_cat)
            v.nodes.append(kayak.SoftReLU(kayak.MatMult(cat_node, cat_weights[(layer, num_neighbors)])))

        for e in graph.edges:
            e.nodes.append(kayak.Identity(e.nodes[layer]))

    # Connect everything to the fixed-size layer using some sort of max
    penultimate_nodes = [v.nodes[-1] for v in graph.verts]
    concatenated = kayak.Concatenate( 0, *penultimate_nodes)
    output_layer = kayak.MatSum( concatenated, 0)   # TODO: Turn sum into a softmax.

    # Perform a little more computation to get a single number.
    output = kayak.MatMult(output_layer, k_weights['out'])
    return kayak.L2Loss(output, kayak.Targets(target)), k_weights, output

