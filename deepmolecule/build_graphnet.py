
from funkyyak import grad, numpy_wrapper as np
from features import N_atom_features, N_bond_features
from util import memoize
from mol_graph import graph_from_smiles_tuple

class WeightsParser(object):
    """A kind of dictionary of weights shapes,
       which can pick out named subsets from a long vector.
       Does not actually store any weights itself."""
    def __init__(self):
        self.idxs_and_shapes = {}
        self.N = 0

    def add_weights(self, name, shape):
        start = self.N
        self.N += np.prod(shape)
        self.idxs_and_shapes[name] = (slice(start, self.N), shape)

    def get(self, vect, name):
        """Takes in a vector and returns the subset indexed by name."""
        idxs, shape = self.idxs_and_shapes[name]
        return np.reshape(vect[idxs], shape)



def build_graphnet(site_vec_dim=10, core_vec_dim=20, l2_penalty=0.0):
    """Sets up a recursive computation graph that
       shrinks the molecular graph into a fixed-size representation."""
    # TODO: Add biases.

    parser = WeightsParser()

    parser.add_weights('atom2vec', (N_atom_features, core_vec_dim))
    parser.add_weights('bond2vec', (N_bond_features, site_vec_dim))

    def molgraph2vec(weights, molgraph):
        """This function is a first pass that converts the 1-of-k encoding
           for both atoms and bond types into a vector."""
        atom_vectors = parser.get(weights, 'atom2vec')
        for k,v in molgraph.nodes.iteritems():
            molgraph.nodes[k] = np.dot(v, atom_vectors)

        bond_vectors = parser.get(weights, 'bond2vec')
        for k,v in molgraph.bonds.iteritems():
            molgraph.bonds[k] = np.dot(v, bond_vectors)

    # The recursive net has 3 functions which need to be applied to shrink
    # the net while keeping (in principle) all information about the graph.

    def nonlinearity(units, weights):
        return np.tanh(np.dot(units, weights))

    parser.add_weights('combine cores', (2*core_vec_dim + 2*site_vec_dim, core_vec_dim))
    def combine_nodes(weights, core_left, core_right, site_left, site_right):
        """Combines two nodes along an edge,
           returns a new vector representing the new node."""
        concat_units = [core_left, core_right, site_left, site_right]
        return nonlinearity(concat_units, parser.get(weights, 'combine cores '))

    parser.add_weights('update site', (2*core_vec_dim + 3*site_vec_dim, site_vec_dim))
    def update_site(weights, core_left, core_right, site_left, site_right, site_self):
        """Updates a site (connection parameters) to account for the fact that
           the node it was connected to was merged."""
        concat_units = [core_left, core_right, site_left, site_right, site_self]
        return nonlinearity(concat_units, parser.get(weights, 'update site'))

    parser.add_weights('remove loop', (core_vec_dim + 2*site_vec_dim, core_vec_dim))
    def remove_self_loop(weights,  core_self, site_left, site_right):
        """Removes a self-loop and updates the core vector."""
        concat_units = [core_self, site_left, site_right]
        return nonlinearity(concat_units, parser.get(weights, 'remove loop'))

    def hidden_units(weights, smiles):

        """Recursively combine nodes to get a fixed-size representation."""
        molgraph = graph_from_smiles_tuple(smiles)


    def predictions(weights, smiles):
        """Go from the fixed-size representation to a prediction."""
        return hidden_units(weights, smiles) * parser.get(weights, 'output')

    def loss(weights, smiles, targets):
        log_lik = np.sum((predictions(weights, smiles) - targets)**2)
        log_prior = -l2_penalty * np.dot(weights, weights)
        return - log_prior - log_lik

    return loss, grad(loss), predictions, hidden_units, parser


@memoize
def parsetree_from_smiles(smiles):
    molgraph = graph_from_smiles_tuple(smiles)