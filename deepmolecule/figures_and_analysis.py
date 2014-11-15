import os
import itertools
import numpy as np
import matplotlib
matplotlib.use('Agg')   # Cluster-friendly backend.
import matplotlib.pyplot as plt

from rdkit import Chem
from rdkit.Chem import Draw

from features import N_atom_features, N_bond_features

def print_performance(pred_func, train_inputs, train_targets, test_inputs,
                      test_targets, target_name="", filename=None):
    train_preds = pred_func(train_inputs)
    test_preds = pred_func(test_inputs)
    print "\nPerformance (RMSE) on " + target_name + ":"
    print "Train:", np.sqrt(np.mean((train_preds - train_targets)**2))
    print "Test: ", np.sqrt(np.mean((test_preds - test_targets)**2))
    print "-" * 80
    if filename:
        np.savez_compressed(file=filename,
                            train_preds=train_preds, train_targets=train_targets,
                            test_preds=test_preds, test_targets=test_targets,
                            target_name=target_name)


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


def plot_weight_meanings(weights_file, outdir, outfilename):
    saved_net = np.load(weights_file)
    weights = saved_net['weights']
    arch_params = saved_net['arch_params'][()]

    atoms = ['C', 'N', 'O', 'S', 'F', 'Si', 'P', 'Cl']
    masses = [12, 14,  16,   32,  19,  28,   31, 35.5]
    atom_weights = weights[:N_atom_features]   # TODO: Index these in a more robust way.

    fig = plt.figure()
    ax = fig.add_subplot(1,1,1)
    ax.plot( masses, atom_weights, 'o')

    for ix, atom in enumerate(atoms):
        print "Atom: ", atom, " has weight", atom_weights[ix]
        ax.text( masses[ix], atom_weights[ix], atom)
    ax.set_xlabel("True mass")
    ax.set_ylabel("Weights")
    plt.savefig(os.path.join(outdir, outfilename + '-atoms.png'))
    plt.savefig(os.path.join(outdir, outfilename + '-atoms.eps'))
    plt.close()

    if arch_params['bond_vec_dim'] > 0:
        bond_names = ['single', 'double', 'triple', 'aromatic', 'conjugated', 'in ring']
        bond_masses = [1.0, 2.0, 3.0, 1.5, 4.0, 1.5]
        bond_weights = weights[N_atom_features + 1:N_atom_features+1+N_bond_features]

        fig = plt.figure()
        ax = fig.add_subplot(1,1,1)
        ax.plot( bond_masses, bond_weights, 'o')
        for ix, bond in enumerate(bond_names):
            print "Bond: ", bond, " has weight", bond_weights[ix]
            ax.text( bond_masses[ix], bond_weights[ix], bond)
        ax.set_xlabel("True mass")
        ax.set_ylabel("Weights")
        plt.savefig(os.path.join(outdir, outfilename + '-bonds.png'))
        plt.savefig(os.path.join(outdir, outfilename + '-bonds.eps'))
        plt.close()


