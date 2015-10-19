from util import memoize, WeightsParser
from rdkit_utils import smiles_to_fps
from build_convnet import build_convnet_fingerprint_fun

def build_double_morgan_fingerprint_fun(fp_length=512, fp_radius=4):

    def fingerprints_from_smiles(weights, smiles1, smiles2):
        # Morgan fingerprints don't use weights.
        fp_tuple_1 = fingerprints_from_smiles_tuple(tuple(smiles1))
        fp_tuple_2 = fingerprints_from_smiles_tuple(tuple(smiles2))
        return zip(fp_tuple_1, fp_tuple_2)

    @memoize # This wrapper function exists because tuples can be hashed, but arrays can't.
    def fingerprints_from_smiles_tuple(smiles_tuple):
        return smiles_to_fps(smiles_tuple, fp_length, fp_radius)

    return fingerprints_from_smiles


def build_double_convnet_fingerprint_fun(**kwargs):

    fp_fun1, parser1 = build_convnet_fingerprint_fun(**kwargs)
    fp_fun2, parser2 = build_convnet_fingerprint_fun(**kwargs)

    def double_fingerprint_fun(weights, smiles1, smiles2):
        fp1 = fp_fun1(weights, smiles1)
        fp2 = fp_fun2(weights, smiles2)
        return zip(fp1, fp2)

    combined_parser = WeightsParser()
    combined_parser.add_weights('weights1', len(parser1))
    combined_parser.add_weights('weights2', len(parser2))

    return double_fingerprint_fun, combined_parser
