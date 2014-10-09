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
import pandas as pd

from rdkit.Chem import AllChem, MolFromSmiles

sys.path.append('../../Kayak/')
import kayak

from MolGraph import *
from features import *


num_folds    = 2
batch_size   = 256
num_epochs   = 10
learn_rate   = 0.001
momentum     = 0.95
h1_dropout   = 0.1
h1_size      = 500

dropout_prob = 0.1
l1_weight    = 1.0
l2_weight    = 1.0


def compute_features(mol, size=512, radius=2):
    return AllChem.GetMorganFingerprintAsBitVect(mol, radius, nBits=size)

def train_2layer_nn(features, targets):
    
    # Normalize the outputs.
    targ_mean = np.mean(targets)
    targ_std  = np.std(targets)

    batcher = kayak.Batcher(batch_size, features.shape[0])

    X = kayak.Inputs(features, batcher)
    T = kayak.Targets((targets[:,None] - targ_mean) / targ_std, batcher)

    W1 = kayak.Parameter( 0.1*npr.randn( features.shape[1], h1_size ))
    B1 = kayak.Parameter( 0.1*npr.randn( 1, h1_size ) )
    H1 = kayak.Dropout(kayak.HardReLU(kayak.ElemAdd(kayak.MatMult(X, W1), B1)), h1_dropout, batcher=batcher)

    W2 = kayak.Parameter( 0.1*npr.randn( h1_size, 1 ) )
    B2 = kayak.Parameter( 0.1*npr.randn(1,1))

    Y = kayak.ElemAdd(kayak.MatMult(H1, W2), B2)

    L = kayak.MatSum(kayak.L2Loss(Y, T))

    mom_grad_W1 = np.zeros(W1.shape)
    mom_grad_W2 = np.zeros(W2.shape)

    for epoch in xrange(num_epochs):

        total_loss = 0.0
        total_err  = 0.0
        total_data = 0
        
        for batch in batcher:

            H1.draw_new_mask()

            total_loss += L.value
            total_err  += np.sum(np.abs(Y.value - T.value))
            total_data += T.shape[0]

            grad_W1 = L.grad(W1)
            grad_B1 = L.grad(B1)
            grad_W2 = L.grad(W2)
            grad_B2 = L.grad(B2)

            mom_grad_W1 = momentum*mom_grad_W1 + (1.0-momentum)*grad_W1
            mom_grad_W2 = momentum*mom_grad_W2 + (1.0-momentum)*grad_W2

            W1.value += -learn_rate * mom_grad_W1
            W2.value += -learn_rate * mom_grad_W2
            B1.value += -learn_rate * grad_B1
            B2.value += -learn_rate * grad_B2
        
        print epoch, total_err / total_data

    def make_predictions(newvals):
        X.data = newvals
        batcher.test_mode()
        return Y.value*targ_std + targ_mean

    return make_predictions



def stringlist2intarray(A):
    '''This function will convert from a list of strings "10010101" into in integer numpy array '''
    return np.array([list(s) for s in A], dtype=int)


def load_molecules(filename, test=False, predict_field='rate', transform=None, cutoff = None):

    data = pd.io.parsers.read_csv(filename, sep=',')  # Load the data.
    fingerprints = [smile_to_fp(s) for s in list(data['smiles'])]

    alldata = {}
    alldata['fingerprints'] = stringlist2intarray(fingerprints)  # N by D
    alldata['inchi_key'] = data['inchi_key']
    alldata['smiles'] = data['smiles']

    if test == False:
        alldata['y'] = np.array(transform(data[predict_field]))[:, None]  # Load targets

        if cutoff:
            above_cutoff = (alldata['y'] > cutoff) & np.isfinite(alldata['y'])
        else:
            above_cutoff = np.isfinite(alldata['y'])
        alldata = slicedict(alldata, above_cutoff)
        print "Datapoints thrown away:", np.sum(np.logical_not(above_cutoff))
        print "Datapoints kept:", np.sum(above_cutoff)

    return alldata


def slicedict(d, ixs):
    newd = {}
    for key in d:
        newd[key] = d[key][ixs.flatten()]
    return newd

def smile_to_fp(s):
    fplength = 512
    radius = 4
    m = Chem.MolFromSmiles(s)
    return (AllChem.GetMorganFingerprintAsBitVect(m, radius,
                                                  nBits=fplength)).ToBitString()



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

    for layer in range(num_layers):
        # Every atom and edge is a separate Kayak Input. These inputs already live in the graph.
        for v in graph.verts:
            # Create a Differentiable node N that depends on the corresponding node in the previous layer, its edges,
            # and its neighbours.
            mults = [kayak.MatMult(v.nodes[layer], k_weights[('self', layer)])]
            for n in v.get_neighbors()[0]:
                mults.append(kayak.MatMult( n.nodes[layer], k_weights[('other', layer)]))
            for e in v.edges:
                mults.append(kayak.MatMult( e.nodes[layer], k_weights[('edge', layer)]))

            # Add the next layer of computation to this node.
            v.nodes.append(kayak.SoftReLU(kayak.ElemAdd(*mults)))

        for e in graph.edges:
            e.nodes.append(kayak.Identity(e.nodes[layer]))

    # Connect everything to the fixed-size layer using some sort of max
    penultimate_nodes = [v.nodes[-1] for v in graph.verts]
    concatenated = kayak.Concatenate( 0, *penultimate_nodes)
    output_layer = kayak.MatSum( concatenated, 0)   # TODO: Turn sum into a softmax.

    # Perform a little more computation to get a single number.
    output = kayak.MatMult(output_layer, k_weights['out'])

    return kayak.L2Loss(output, kayak.Targets(target)), k_weights, output

def train_custom_nn(smiles, targets, num_hidden_features = [10, 10]):

    num_layers = len(num_hidden_features)

    # Figure out how many features we have.
    graph = BuildGraphFromMolecule(Chem.MolFromSmiles('C=C'))   # Hacky? You decide.
    num_atom_features = graph.verts[0].nodes[0].shape[1]
    num_edge_features = graph.edges[0].nodes[0].shape[1]
    num_features = [num_atom_features] + num_hidden_features

    # Create weights ot be passed in to all the networks we build.
    np_weights = {}
    for layer in range(num_layers):
        np_weights[('self', layer)] = 0.1*npr.randn(num_features[layer], num_features[layer + 1])
        np_weights[('other', layer)] = 0.1*npr.randn(num_features[layer], num_features[layer + 1])
        np_weights[('edge', layer)] = 0.1*npr.randn(num_edge_features, num_features[layer + 1])

    np_weights['out'] = 0.1*npr.randn(num_features[-1], 1)

    # Normalize the outputs.
    targ_mean = np.mean(targets)
    targ_std  = np.std(targets)

    # Build a list of custom neural nets, one for each molecule, all sharing the same set of weights.
    losses = []
    all_k_weights = []
    print "Building molecular nets",
    for smile, target in zip(smiles, targets):
        mol = Chem.MolFromSmiles(smile)
        graph = BuildGraphFromMolecule(mol)
        loss, k_weights, _ = BuildNetFromGraph(graph, np_weights, (target - targ_mean)/targ_std, num_layers)
        losses.append(loss)
        all_k_weights.append(k_weights)
        print ".",

    # Now actually learn.

    learn_rate = 1e-6

    # TODO: implement RMSProp or whatever.
    print "\nTraining parameters",
    num_epochs = 10
    for epoch in xrange(num_epochs):
        total_loss = 0
        for loss, k_weights in zip(losses, all_k_weights):
            # Loop over kayak parameter vectors and evaluate the gradient w.r.t. each one.
            # Take a step on all parameters.
            for key, cur_k_weights in k_weights.iteritems():
                # Set weights value to None so that the weights arrays can be garbage-collected
                cur_k_weights.value = np_weights[key]
            total_loss += loss.value
            for key, cur_k_weights in k_weights.iteritems():
                cur_grad = loss.grad(cur_k_weights)
                np_weights[key] -= cur_grad * learn_rate
            for key, cur_k_weights in k_weights.iteritems():
                # Set weights value to None so that the weights arrays can be garbage-collected
                cur_k_weights.value = None
        print "Current loss after epoch", epoch, ":", total_loss

    def make_predictions(smiles):
        for smile in smiles:
            mol = Chem.MolFromSmiles(smile)
            graph = BuildGraphFromMolecule(mol)
            _, _, output = BuildNetFromGraph(graph, np_weights, None, num_layers)
            return output.value*targ_std + targ_mean

    return make_predictions


def main():

    datadir = '/Users/dkd/Dropbox/Molecule_ML/data/Samsung_September_8_2014/'

    trainfile = datadir + 'davids-validation-split/train_split.csv'
    #trainfile = datadir + 'davids-validation-split/tiny.csv'
    testfile = datadir + 'davids-validation-split/test_split.csv'
    #testfile = datadir + 'davids-validation-split/tiny.csv'

    print "Loading training data..."
    traindata = load_molecules(trainfile, transform = np.log)

    print "Loading test data..."
    testdata = load_molecules(testfile, transform = np.log)

    # Custom Neural Net
    pred_func_custom = train_custom_nn(traindata['smiles'], traindata['y'])
    train_preds = pred_func_custom( traindata['smiles'] )
    test_preds = pred_func_custom( testdata['smiles'] )
    print "Custom net test performance: ", \
        np.mean(np.abs(train_preds-traindata['y'])), np.mean(np.abs(test_preds-testdata['y']))

    # Vanilla Neural Net
    pred_func_vanilla = train_2layer_nn(traindata['fingerprints'], traindata['y'])
    train_preds = pred_func_vanilla( traindata['fingerprints'] )
    test_preds = pred_func_vanilla( testdata['fingerprints'] )
    print "Vanilla net test performance: ", \
        np.mean(np.abs(train_preds-traindata['y'])), np.mean(np.abs(test_preds-testdata['y']))
        

if __name__ == '__main__':
    sys.exit(main())
