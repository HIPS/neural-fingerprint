"""Checks that the fingerprints are sensitive to chemically-relevant differences,
and invariant to arbitrary changes in atom numberings.

Examples from
http://www.daylight.com/meetings/summerschool98/course/dave/smiles-isomers.html
"""

import numpy as np
import numpy.random as npr
npr.seed(0)

from neuralfingerprint import build_convnet_fingerprint_fun, build_morgan_fingerprint_fun

def invariant_conv_fp_func():
    fp_func, parser = build_convnet_fingerprint_fun(
        num_hidden_features=[100, 100, 100], fp_length=64, normalize=False)
    weights = npr.randn(len(parser))
    return lambda smiles: fp_func(weights, (smiles,))

def invariant_morg_fp_func():
    fp_func = build_morgan_fingerprint_fun(fp_length=64, fp_radius=3)
    return lambda smiles: fp_func([], (smiles,))

def check_conv_fps_same(smiles1, smiles2):
    fp_func = invariant_conv_fp_func()
    conv_fp1 = fp_func(smiles1)
    conv_fp2 = fp_func(smiles2)
    assert np.allclose(conv_fp1, conv_fp2), "Diffs are:\n{0}.".format(conv_fp1 - conv_fp2)

def check_conv_fps_diff(smiles1, smiles2):
    fp_func = invariant_conv_fp_func()
    conv_fp1 = fp_func(smiles1)
    conv_fp2 = fp_func(smiles2)
    assert not np.allclose(conv_fp1, conv_fp2), "Diffs are:\n{0}.".format(conv_fp1 - conv_fp2)

def check_morg_fps_same(smiles1, smiles2):
    fp_func = invariant_morg_fp_func()
    conv_fp1 = fp_func(smiles1)
    conv_fp2 = fp_func(smiles2)
    assert np.allclose(conv_fp1, conv_fp2), "Diffs are:\n{0}.".format(conv_fp1 - conv_fp2)

def check_morg_fps_diff(smiles1, smiles2):
    fp_func = invariant_morg_fp_func()
    conv_fp1 = fp_func(smiles1)
    conv_fp2 = fp_func(smiles2)
    assert not np.allclose(conv_fp1, conv_fp2), "Diffs are:\n{0}.".format(conv_fp1 - conv_fp2)

def test_sanity():
    check_conv_fps_same('C=C', 'C=C')
    check_morg_fps_same('C=C', 'C=C')

    check_conv_fps_diff('C=C', 'C=O')
    check_morg_fps_diff('C=C', 'C=O')

def test_ring_closure():
    check_conv_fps_same('c12c(cccc1)cccc2', 'c1cc2ccccc2cc1')
    check_morg_fps_same('c12c(cccc1)cccc2', 'c1cc2ccccc2cc1')

def test_ring_closure_digits():
    check_conv_fps_same('c1ccccc1c1ccccc1', 'c1ccccc1c2ccccc2')
    check_morg_fps_same('c1ccccc1c1ccccc1', 'c1ccccc1c2ccccc2')

def test_ring_bond_location():
    check_conv_fps_same('C1=CCCCC1', 'C=1CCCCC1')
    check_morg_fps_same('C1=CCCCC1', 'C=1CCCCC1')

def test_ring_bond_location_with_attachment():
    check_conv_fps_diff('C1=CCCNCC1', 'C=1CCCNCC1')
    check_morg_fps_diff('C1=CCCNCC1', 'C=1CCCNCC1')

def test_branch_stacking():
    check_conv_fps_same('O=Cl(=O)(=O)[O-]', 'Cl(=O)(=O)(=O)[O-]')
    check_morg_fps_same('O=Cl(=O)(=O)[O-]', 'Cl(=O)(=O)(=O)[O-]')

def test_bond_config():
    check_conv_fps_same('F/C=C/F', 'F\C=C\F')
    check_morg_fps_same('F/C=C/F', 'F\C=C\F')

def test_chirality():
    check_conv_fps_same('N[C@@H](C)C(=O)O', 'N[C@H](C)C(=O)O')
    check_morg_fps_same('N[C@@H](C)C(=O)O', 'N[C@H](C)C(=O)O')

def test_aromaticity():
    check_conv_fps_same('FC1=CC=CC=C1O', 'FC1=C(O)C=CC=C1')
    check_morg_fps_same('FC1=CC=CC=C1O', 'FC1=C(O)C=CC=C1')
    check_conv_fps_same('Fc1ccccc1O', 'FC1=C(O)C=CC=C1')
    check_morg_fps_same('Fc1ccccc1O', 'FC1=C(O)C=CC=C1')

def test_azulene():
    check_conv_fps_same('c1cc2cccccc2c1', 'C1=CC2=CC=CC=CC2=C1')
    check_morg_fps_same('c1cc2cccccc2c1', 'C1=CC2=CC=CC=CC2=C1')

def test_tautomers():
    check_conv_fps_diff('O=c1[nH]cccc1', 'Oc1ncccc1')
    check_morg_fps_diff('O=c1[nH]cccc1', 'Oc1ncccc1')

def test_order_sensitivity():
    check_conv_fps_diff('OCNF', 'ONCF')
    check_morg_fps_diff('OCNF', 'ONCF')

    check_conv_fps_diff('OCNC', 'ONCC')
    check_morg_fps_diff('OCNC', 'ONCC')
