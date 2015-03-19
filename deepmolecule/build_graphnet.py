
import numpy as np

from funkyyak import grad
from features import N_atom_features, N_bond_features
from util import memoize, WeightsParser
from mol_graph import graph_from_smiles_tuple

from rdkit.Chem import MolFromSmiles
from features import atom_features, bond_features

from itertools import combinations_with_replacement

import numpy.random as npr

def build_graphnet(site_vec_dim=10, core_vec_dim=20, l2_penalty=0.0):
    """Sets up a recursive computation graph that
       shrinks the molecular graph into a fixed-size representation."""
    parser = WeightsParser()
    parser.add_weights('atom2vec', (N_atom_features, core_vec_dim))
    parser.add_weights('bond2vec', (N_bond_features, site_vec_dim))

    def atomgraph2vec(weights, atomgraph):
        """This function is a first pass that converts the 1-of-k encoding
           for both atoms and bond types into a vector."""
        atom_vectors = parser.get(weights, 'atom2vec')
        for atom in atomgraph.get_atoms():
            atom.core = np.dot(atom.core, atom_vectors)

        bond_vectors = parser.get(weights, 'bond2vec')
        for edge in atomgraph.get_edges:
            edge.site1 = np.dot(edge.site1, bond_vectors)
            edge.site2 = np.dot(edge.site1, bond_vectors)

    # The recursive net has 3 functions which need to be applied to shrink
    # the net while keeping (in principle) all information about the graph.
    # TODO: Add biases to each of these.
    parser.add_weights('combine cores', (2*core_vec_dim + 2*site_vec_dim, core_vec_dim))
    def combine_cores(weights, core_left, core_right, site_left, site_right):
        """Combines two nodes along an edge,
           returns a new vector representing the new node."""
        concat_units = np.concatenate([core_left, core_right, site_left, site_right])
        return nonlinearity(concat_units, parser.get(weights, 'combine cores'))

    parser.add_weights('update site', (2*core_vec_dim + 3*site_vec_dim, site_vec_dim))
    def update_site(weights, core_left, core_right, site_left, site_right, site_self):
        """Updates a site (connection parameters) to account for the fact that
           the node it was connected to was merged."""
        concat_units = np.concatenate([core_left, core_right, site_left, site_right, site_self])
        return nonlinearity(concat_units, parser.get(weights, 'update site'))

    parser.add_weights('remove loop', (core_vec_dim + 2*site_vec_dim, core_vec_dim))
    def remove_self_loop(weights,  core_self, site_left, site_right):
        """Removes a self-loop and updates the core vector."""
        concat_units = np.concatenate([core_self, site_left, site_right])
        return nonlinearity(concat_units, parser.get(weights, 'remove loop'))

    def combination_ranker(atom_and_site_pairs, weights):
        # TODO: make the ranking be learned instead of random.
        return npr.randn((len(atom_and_site_pairs), 1))

    def hidden_units(weights, smile):
        """Recursively combines nodes to get a fixed-size representation."""
        atomgraph = graph_from_smile(smile)
        atomgraph2vec(weights, atomgraph)

        def get_next_pair_to_merge():
            atom_pairs = combinations_with_replacement(atomgraph.get_atoms())
            # with replacement to allow for self-loops,
            # and for joining left-to-right instead of right-to-left.
            atom_and_site_pairs = \
                [(atom_pair, edge)
                 for atom_pair in atom_pairs
                 for edge in atomgraph.get_connecting_edges(atom_pair)]
            rankings = combination_ranker(atom_and_site_pairs)
            # TODO: only re-rank new pairs, to save time.
            return atom_and_site_pairs[np.argmax(rankings)]

        def merge_atoms(left_atom, right_atom, edge):
            """Updates atomgraph in place."""
            left_site  = edge.this_site(left_atom)
            right_site = edge.this_site(right_atom)
            atomgraph.remove_edge(edge)

            if left_atom is right_atom:  # Self-loop
                self_loop_weights = parser.get(weights, 'remove loop')
                left_atom.core = remove_self_loop(self_loop_weights, left_atom.core,
                                                  left_site, right_site)
            else:
                combine_weights = parser.get(weights, 'combine cores')
                new_core = combine_cores(combine_weights, left_atom.core, right_atom.core,
                                         left_site, right_site)
                new_atom = Atom(new_core)
                #combined_sites = left_atom.sites + right_atom.sites
                #combined_edges = union(left_atom.edges, right_atom.edges)
                update_weights = parser.get(weights, 'update site')
                # Update edges to connect to new atom.
                for edge in left_atom.edges:
                    new_site = update_site(update_weights, left_atom.core, right_atom.core,
                                           left_site, right_site, edge.this_site(left_atom))
                    edge.set_this_site(left_atom, new_site)
                    edge.set_other_atom(left_atom, new_atom)
                for edge in right_atom.edges:
                    # TODO: Put some thought into if the update arguments should be swapped.
                    # If not, perhaps this second bit of code can be combined with the above.
                    new_site = update_site(update_weights, left_atom.core, right_atom.core,
                                           left_site, right_site, edge.this_site(right_atom))
                    edge.set_this_site(right_atom, new_site)
                    edge.set_other_atom(right_atom, new_atom)

        while (len(atomgraph.get_atoms()) > 1    # Merge all atoms.
             & len(atomgraph.get_edges()) > 0):  # Close all self loops.
            merge_atoms(get_next_pair_to_merge())
        return atomgraph.get_atoms()[0].core

    def predictions(weights, smiles):
        """Go from the fixed-size representation to a prediction."""
        return hidden_units(weights, smiles) * parser.get(weights, 'output')

    def loss(weights, smiles, targets):
        log_lik = np.sum((predictions(weights, smiles) - targets)**2)
        log_prior = -l2_penalty * np.dot(weights, weights)
        return - log_prior - log_lik

    return loss, grad(loss), predictions, hidden_units, parser


class AtomGraph(object):
    def __init__(self):
        self.atoms = []
        self.edges = []

    def add_atom(self, atom):
        self.atoms.append(atom)

    def add_edge(self, edge):
        assert edge.get_atoms()[0] in self.atoms \
               and edge.get_atoms()[1] in self.atoms, \
            "Edge refers to non-existent atom."
        self.edges.append(edge)

    def remove_edge(self, edge):
        self.edges.remove(edge)

    def get_atoms(self):
        return self.atoms

    def get_edges(self):
        return self.edges

def get_connecting_edges(left_atom, right_atom):
    return intersect(left_atom.edges, right_atom.edges)

class Atom(object):
    def __init__(self, core):
        self.edges = ()
        self.core = core

    def get_degree(self):
        return len(self.edges)

    def add_edge(self, edge):
        # assert edge not in self.edges, \
        # "Edge is already associated with atom.  Self loops are not currently supported."
        self.edges = self.edges + (edge, )

    def get_neighbors(self):
        return [e.other_atom(self) for e in self.edges], self.edges

class Edge(object):
    def __init__(self, atom1, site1, atom2, site2):
        self.atom1 = atom1
        self.atom2 = atom2
        self.site1 = site1
        self.site2 = site2

        # Register with the atoms.
        self.atom1.add_edge(self)
        self.atom2.add_edge(self)

    def get_atoms(self):
        return (self.atom1, self.atom2)

    def other_atom(self, atom):
        if self.atom1 is atom:
            return self.atom2
        elif self.atom2 is atom:
            return self.atom1
        else:
            raise Exception("Edge does not connect this atom.")

    def set_other_atom(self, atom, new_atom):
        if self.atom1 is atom:
            self.atom2 = new_atom
        elif self.atom2 is atom:
            self.atom2 = new_atom
        else:
            raise Exception("Edge does not connect this atom.")

    def this_site(self, atom):
        if self.atom1 is atom:
            return self.site1
        elif self.atom2 is atom:
            return self.site2
        else:
            raise Exception("Edge does not connect this atom.")

    def set_this_site(self, atom, new_site):
        if self.atom1 is atom:
            self.site1 = new_site
        elif self.atom2 is atom:
            self.site2 = new_site
        else:
            raise Exception("Edge does not connect this atom.")

def graph_from_smile(smile):
    """Build a single graph from a single SMILES string."""
    graph = AtomGraph()
    mol = MolFromSmiles(smile)
    atoms_by_rd_idx = {}
    for atom in mol.GetAtoms():
        new_atom = Atom(atom_features(atom))
        graph.new_atom(new_atom)
        atoms_by_rd_idx[atom.GetIdx()] = new_atom

    for bond in mol.GetBonds():
        atom1 = atoms_by_rd_idx[bond.GetBeginAtom().GetIdx()]
        atom2 = atoms_by_rd_idx[bond.GetEndAtom().GetIdx()]
        features = bond_features(bond)
        graph.add_edge(Edge(atom1, features, atom2, features))
    return graph