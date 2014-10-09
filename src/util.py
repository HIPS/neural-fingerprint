from MolGraph import *

import sys
sys.path.append('../../Kayak/')
import kayak

import numpy as np

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

def drawComputationGraph(mol_graph, target):
    """Graph already contains a list of vertices, each one has a Kayak Input,
    which knows about its children.

    Graph is the mol graph.
    """
    # Make a dict indexed by kayak nodes that has the location of each atom as values.
    # First fix the locations of the atoms and bonds
    node_positions = {}
    base_nodes = mol_graph.get_verts()
    num_layers = len(mol_graph.verts[0].nodes)
    for layer in range(num_layers):
        curr_z = layer * 3
        for k_n, mol_v in [(mol_v.nodes[layer], mol_v) for mol_v in base_nodes]:
            node_positions[k_n] = mol_v.pos + (curr_z, )
        base_edges = mol_graph.get_edges()
        for mol_e in base_edges:
            mol_v1, mol_v2 = mol_e.get_verts()
            avg_x = (mol_v1.pos[0] + mol_v2.pos[0]) / 2.0
            avg_y = (mol_v1.pos[1] + mol_v2.pos[1]) / 2.0
            node_positions[mol_e.nodes[layer]] = (avg_x, avg_y, curr_z)

    # Now plot all edges of interest given these positions.
    ax = plt.figure().add_subplot(111, projection = '3d')
    for n1, n2 in find_all_edges_leading_to(target):
        pos1 = position(n1, node_positions)
        pos2 = position(n2, node_positions)
        if pos1 and pos2:
            ax.plot(*(zip(pos1, pos2)), color="RoyalBlue", lw=1)

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


# Define the positions of a node.
def position(k_node, node_positions):
    """node is a kayak node."""
    if k_node in node_positions:
        return node_positions[k_node]
    elif not k_node._parents:
        return None
    else:
        parents_positions = [position(p, node_positions) for p in k_node._parents]
        parents_positions = [p for p in parents_positions if p is not None]
        if not parents_positions:
            return None
        xs, ys, zs = zip(*parents_positions)
        new_x = np.mean(xs)
        new_y = np.mean(ys)
        new_z = np.max(zs) + 1
        node_positions[k_node] = (new_x, new_y, new_z)
        return position(k_node, node_positions)