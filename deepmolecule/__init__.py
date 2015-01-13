
from build_convnet import build_universal_net
from build_lstm import build_lstm_rnn, build_vanilla_rnn
from util import tictoc, normalize_array, c_value, c_grad, WeightsParser
from optimization_routines import sgd_with_momentum, rms_prop, make_batcher,\
    batch_idx_generator, conj_grad
from io_utils import get_output_file, get_data_file, load_data, output_dir
from rdkit_utils import smiles_to_fps
from build_vanilla_net import build_morgan_deep_net, build_morgan_flat_net
from figures_and_analysis import plot_predictions, plot_maximizing_inputs,\
    plot_weight_meanings, plot_learning_curve, plot_weights_container, plot_weights
from odyssey import run_jobs
from train_nets import run_nn_with_params, random_net_linear_output