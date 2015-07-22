
from util import tictoc, normalize_array, WeightsParser, build_batched_grad, add_dropout
from optimizers import sgd, rms_prop, adam, bfgs
from io_utils import get_output_file, get_data_file, load_data, load_data_slices, output_dir
from rdkit_utils import smiles_to_fps
from build_convnet import build_conv_deep_net, build_convnet_fingerprint_fun
from build_vanilla_net import build_morgan_deep_net, build_morgan_fingerprint_fun, \
    build_standard_net, binary_classification_nll, mean_squared_error
from mol_graph import degrees
#from figures_and_analysis import plot_predictions, plot_maximizing_inputs,\
#    plot_learning_curve, plot_weights_container, plot_weights
