
from build_kayak_net_nodewise import initialize_weights, BuildNetFromSmiles
from build_kayak_net_arrayrep import build_universal_net
from util import tictoc, normalize_array, c_value, c_grad
from optimization_routines import sgd_with_momentum, rms_prop, make_batcher, batch_idx_generator
from io_utils import get_output_file, get_data_file, load_data, output_dir
from rdkit_utils import smiles_to_fps
from build_vanilla_net import build_morgan_deep_net
