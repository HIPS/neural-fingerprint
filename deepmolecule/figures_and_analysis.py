import os
import itertools
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from rdkit import Chem
from rdkit.Chem import Draw

def plot_predictions(results_filename, outdir):
    """Generates prediction vs actual scatterplots."""

    def scatterplot(x, y, title, outfilename):
        fig = plt.figure()
        fig.add_subplot(1,1,1)
        plt.scatter(x, y)
        matplotlib.rcParams.update({'font.size': 16})
        plt.xlabel("predicted " + target_name)
        plt.ylabel("true " + target_name)
        plt.title(title)
        plt.savefig(os.path.join(outdir, outfilename + '.png'))
        plt.savefig(os.path.join(outdir, outfilename + '.eps'))
        #plt.draw()
        plt.close()

    preds = np.load(results_filename)
    target_name = str(preds['target_name'])
    if not os.path.exists(outdir):
        os.makedirs(outdir)
    scatterplot(preds['train_preds'], preds['train_targets'],
                "Training Accuracy on " + target_name, "training_" + target_name)
    scatterplot(preds['test_preds'], preds['test_targets'],
                "Test Accuracy on " + target_name, "testing_" + target_name)


def plot_maximizing_inputs(net_building_func, weights_file, outdir):
    """Plots the molecular fragment which maximizes each hidden unit of the network."""

    # Build the network
    saved_net = np.load(weights_file)
    trained_weights = saved_net['weights']
    arch_params = saved_net['arch_params'][()]
    _, _, _, hidden_layer, zero_weights = net_building_func(**arch_params)
    assert(len(trained_weights) == zero_weights.N)

    # Make a set of smiles to search over.
    def generate_smiles_list():
        fragments = ['C', 'N', 'O', 'c1ccccc1', 'F', "CC(C)C","NC(=O)C"]
        return [''.join(s) for s in itertools.combinations_with_replacement(fragments, 2)]
    smiles_list = np.array(generate_smiles_list())

    # Evaluate on the network and find the best smiles.
    input_scores = hidden_layer(trained_weights, smiles_list)
    max_smiles_ixs = np.argmax(input_scores, axis=0)
    min_smiles_ixs = np.argmin(input_scores, axis=0)

    # Now draw them and save the images.
    if not os.path.exists(outdir):
        os.makedirs(outdir)

    for n, ix in enumerate(max_smiles_ixs):
        mol = Chem.MolFromSmiles(smiles_list[ix])
        outfilename = os.path.join(outdir, 'hidden-unit-' + str(n) + '-maximizing.png')
        Draw.MolToFile(mol, outfilename, fitImage=True)

    for n, ix in enumerate(min_smiles_ixs):
        mol = Chem.MolFromSmiles(smiles_list[ix])
        outfilename = os.path.join(outdir, 'hidden-unit-' + str(n) + '-minimizing.png')
        Draw.MolToFile(mol, outfilename, fitImage=True)



def print_weight_meanings(weights_file, outdir, outfilename):
    saved_net = np.load(weights_file)
    weights = saved_net['weights']
    atoms = ['C', 'N', 'O', 'S', 'F', 'Si', 'P', 'Cl']
    masses = [12, 14,  16,   32,  19,  28,   31, 35.5]

    fig = plt.figure()
    ax = fig.add_subplot(1,1,1)
    ax.plot( masses, weights[:len(masses)], 'o')

    for ix, atom in enumerate(atoms):
        print "Atom: ", atom, " has weight", weights[ix]
        ax.text( masses[ix], weights[ix], atom)
    ax.set_xlabel("True mass")
    ax.set_ylabel("Weights")
    plt.savefig(os.path.join(outdir, outfilename + '.png'))
    plt.savefig(os.path.join(outdir, outfilename + '.eps'))
    plt.close()
