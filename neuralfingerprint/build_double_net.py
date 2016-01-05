from util import memoize, WeightsParser
from rdkit_utils import smiles_to_fps
from build_convnet import build_convnet_fingerprint_fun
from build_vanilla_net import build_fingerprint_deep_net

import autograd.numpy as np

def build_double_morgan_fingerprint_fun(fp_length=512, fp_radius=4):

    def fingerprints_from_smiles(weights, smiles_tuple):
        smiles1, smiles2 = zip(*smiles_tuple)
        # Morgan fingerprints don't use weights.
        fp_array_1 = fingerprints_from_smiles_tuple(tuple(smiles1))
        fp_array_2 = fingerprints_from_smiles_tuple(tuple(smiles2))
        return np.concatenate([fp_array_1, fp_array_2], axis=1)

    @memoize # This wrapper function exists because tuples can be hashed, but arrays can't.
    def fingerprints_from_smiles_tuple(smiles_tuple):
        return smiles_to_fps(smiles_tuple, fp_length, fp_radius)

    return fingerprints_from_smiles


def build_double_morgan_deep_net(fp_length, fp_depth, net_params):
    empty_parser = WeightsParser()
    morgan_fp_func = build_double_morgan_fingerprint_fun(fp_length, fp_depth)
    return build_fingerprint_deep_net(net_params, morgan_fp_func, empty_parser, 0)


def build_double_convnet_fingerprint_fun(**kwargs):

    fp_fun1, parser1 = build_convnet_fingerprint_fun(**kwargs)
    fp_fun2, parser2 = build_convnet_fingerprint_fun(**kwargs)

    def double_fingerprint_fun(weights, smiles_tuple):
        smiles1, smiles2 = zip(*smiles_tuple)
        fp1 = fp_fun1(weights, smiles1)
        fp2 = fp_fun2(weights, smiles2)
        return zip(fp1, fp2)

    combined_parser = WeightsParser()
    combined_parser.add_weights('weights1', len(parser1))
    combined_parser.add_weights('weights2', len(parser2))

    return double_fingerprint_fun, combined_parser


def build_double_conv_deep_net(conv_params, net_params, fp_l2_penalty=0.0):
    """Returns loss_fun(all_weights, smiles, targets), pred_fun, combined_parser."""
    conv_fp_func, conv_parser = build_double_convnet_fingerprint_fun(**conv_params)
    return build_fingerprint_deep_net(net_params, conv_fp_func, conv_parser, fp_l2_penalty)

