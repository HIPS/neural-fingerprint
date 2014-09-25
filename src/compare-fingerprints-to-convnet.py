# A quick comparison script to compare the predictive accuracy of using standard fingerprints versus custom convnets.
#
# Dougal Maclaurin
# David Duvenaud
# Ryan P. Adams
#
# Sept 25th, 2014


import sys
import numpy as np
import numpy.random as npr

from rdkit.Chem import AllChem, MolFromSmiles

sys.path.append('../Kayak/')
import kayak

from MolGraph import *
from features import *


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


def compute_features(mol, size=512, radius=2):
    return AllChem.GetMorganFingerprintAsBitVect(mol, radius, nBits=size)

def train(features, targets):
    
    # Normalize the outputs.
    targ_mean = np.mean(targets)
    targ_std  = np.std(targets)

    batcher = kayak.Batcher(batch_size, features.shape[0])

    X = kayak.Inputs(features, batcher)
    T = kayak.Targets((targets-targ_mean) / targ_std, batcher)

    W1 = kayak.Parameter( 0.1*npr.randn( features.shape[1], h1_size ))
    B1 = kayak.Parameter( 0.1*npr.randn( 1, h1_size ) )
    H1 = kayak.Dropout(kayak.HardReLU(kayak.ElemAdd(kayak.MatMult(X, W1), B1)), h1_dropout)

    W2 = kayak.Parameter( 0.1*npr.randn( h1_size ) )
    B2 = kayak.Parameter( 0.1*npr.randn(1))

    Y = kayak.ElemAdd(kayak.MatMult(H1, W2), B2)

    L = kayak.MatSum(kayak.L2Loss(Y, T))

    mom_grad_W1 = np.zeros(W1.shape())
    mom_grad_W2 = np.zeros(W2.shape())

    for epoch in xrange(num_epochs):

        total_loss = 0.0
        total_err  = 0.0
        total_data = 0
        
        for batch in batcher:

            total_loss += L.value(True)
            total_err  += np.sum(np.abs(Y.value() - T.value()))
            total_data += T.shape()[0]

            grad_W1 = L.grad(W1)
            grad_B1 = L.grad(B1)
            grad_W2 = L.grad(W2)
            grad_B2 = L.grad(B2)

            mom_grad_W1 = momentum*mom_grad_W1 + (1.0-momentum)*grad_W1
            mom_grad_W2 = momentum*mom_grad_W2 + (1.0-momentum)*grad_W2

            W1.add( -learn_rate * mom_grad_W1 )
            W2.add( -learn_rate * mom_grad_W2 )
            B1.add( -learn_rate * grad_B1 )
            B2.add( -learn_rate * grad_B2 )
        
        print epoch, total_err / total_data
    
    return lambda x: Y.value(True, inputs={ X: x })*targ_std + targ_mean

def main():
    filename = '~/Desktop/tddft_hyb_b3l_agatha_harsh_process_postprocess_lifetime.csv'
    # filename = '~/Dropbox/Collaborations/MolecularML.shared/data/ML_exploit_1k/tddft_hyb_b3l_lifetime.csv'
    data     = util.load_csv(filename)
    features = np.array([compute_features(MolFromSmiles(mol['smiles'])) for mol in data])
    targets  = np.log(np.array([float(mol['rate']) for mol in data]))

    # Eliminate NaN values.
    good_indices = np.isfinite(targets)
    targets      = targets[good_indices]
    features     = features[good_indices,...]

    CV = kayak.CrossValidator(num_folds, features, targets)

    for ii, fold in enumerate(CV):
        print "Fold %d" % (ii+1)

        train_features, train_targets = fold.train()
        valid_features, valid_targets = fold.valid()

        pred_func = train(train_features, train_targets)

        train_preds = pred_func( train_features )
        valid_preds = pred_func( valid_features )
        print np.mean(np.abs(train_preds-train_targets)), np.mean(np.abs(valid_preds-valid_targets))
        

if __name__ == '__main__':
    sys.exit(main())
