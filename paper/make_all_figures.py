# Makes all the figures for the paper by grabbing results from existing experiments.

import sys, os

from deepmolecule import build_universal_net
from deepmolecule import plot_predictions, plot_maximizing_inputs, print_weight_meanings

def main():
    figures_dir = os.path.expanduser("~/repos/DeepMolecules/paper/figures")
    atom2vec_experiment_dir = os.path.expanduser("~/repos/DeepMoleculesData/results/2014-11-14-predict-mass-0")
    atom2vec_predictions_file = os.path.join(atom2vec_experiment_dir, 'convnet-predictions-mass.npz')
    atom2vec_weights_file = os.path.join(atom2vec_experiment_dir, 'conv-net-weights.npz')

    print "Plotting prediction accuracy..."
    plot_predictions(atom2vec_predictions_file, os.path.join(figures_dir, 'convnet-predict-mass-plots'))
    print "Plotting inputs which most maximize features..."
    plot_maximizing_inputs(build_universal_net, atom2vec_weights_file,
                           os.path.join(figures_dir, 'convnet-features-mass'))
    print "Plotting vector representations..."
    print_weight_meanings(atom2vec_weights_file,
                          os.path.join(figures_dir, 'atom2vec-mass-plots'), 'true-vs-vecs')

if __name__ == '__main__':
    sys.exit(main())
