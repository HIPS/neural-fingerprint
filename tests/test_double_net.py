"""Checks that the fingerprints are sensitive to chemically-relevant differences,
and invariant to arbitrary changes in atom numberings.

Examples from
http://www.daylight.com/meetings/summerschool98/course/dave/smiles-isomers.html
"""

import numpy as np
import numpy.random as npr
npr.seed(0)

from neuralfingerprint import build_double_convnet_fingerprint_fun, build_double_morgan_fingerprint_fun

def invariant_conv_fp_func():
    fp_func, parser = build_double_convnet_fingerprint_fun(
        num_hidden_features=[100, 100, 100], fp_length=64, normalize=False)
    weights = npr.randn(len(parser))
    return lambda smiles1, smiles2: fp_func(weights, (smiles1,), (smiles2,))

def invariant_morg_fp_func():
    fp_func = build_double_morgan_fingerprint_fun(fp_length=64, fp_radius=3)
    return lambda smiles1, smiles2: fp_func([], (smiles1,), (smiles2,))

def check_conv_fps_same(smiles1a, smiles1b, smiles2a, smiles2b):
    fp_func = invariant_conv_fp_func()
    conv_fp1 = fp_func(smiles1a, smiles1b)
    conv_fp2 = fp_func(smiles2a, smiles2b)
    assert np.allclose(conv_fp1, conv_fp2), "Diffs are:\n{0}.".format(conv_fp1 - conv_fp2)

def check_conv_fps_diff(smiles1a, smiles1b, smiles2a, smiles2b):
    fp_func = invariant_conv_fp_func()
    conv_fp1 = fp_func(smiles1a, smiles1b)
    conv_fp2 = fp_func(smiles2a, smiles2b)
    assert not np.allclose(conv_fp1, conv_fp2), "Diffs are:\n{0}.".format(conv_fp1 - conv_fp2)

def check_morg_fps_same(smiles1a, smiles1b, smiles2a, smiles2b):
    fp_func = invariant_morg_fp_func()
    conv_fp1 = fp_func(smiles1a, smiles1b)
    conv_fp2 = fp_func(smiles2a, smiles2b)
    assert np.allclose(conv_fp1, conv_fp2), "Diffs are:\n{0}.".format(conv_fp1 - conv_fp2)

def check_morg_fps_diff(smiles1a, smiles1b, smiles2a, smiles2b):
    fp_func = invariant_morg_fp_func()
    conv_fp1 = fp_func(smiles1a, smiles1b)
    conv_fp2 = fp_func(smiles2a, smiles2b)
    assert not np.allclose(conv_fp1, conv_fp2), "Diffs are:\n{0}.".format(conv_fp1 - conv_fp2)

def test_sanity():
    check_conv_fps_same('C=C', 'C=O', 'C=C', 'C=O')
    check_morg_fps_same('C=C', 'C=O', 'C=C', 'C=O')

    check_conv_fps_diff('C=C', 'C=O', 'C=O', 'C=O')
    check_morg_fps_diff('C=C', 'C=O', 'C=O', 'C=O')
