"""Checks gradients for deep networks with fingerprints on the lower layers."""
import numpy.random as npr
from deepmolecule import build_conv_deep_net, build_morgan_deep_net
from autograd.util import check_grads

smiles = ('C=C', 'c12c(cccc1)cccc2', 'C1=CCCCC1', 'FC1=CC=CC=C1O')
npr.seed(0)
targets = npr.randn(len(smiles))

fp_length = 3
vanila_layer_sizes = [fp_length]

def morg_fp_func():
    loss, _, parser = build_morgan_deep_net(vanila_layer_sizes, fp_length=fp_length,
                                            fp_radius=len(vanila_layer_sizes))
    return lambda weights: loss(weights, smiles, targets), parser

def test_morg_net_gradient():
    loss, parser = morg_fp_func()
    weights = npr.randn(len(parser))
    check_grads(loss, weights)

def conv_fp_func(conv_params):
    loss, _, parser = build_conv_deep_net(vanila_layer_sizes, conv_params)
    return lambda weights: loss(weights, smiles, targets), parser

def check_conv_grads(conv_params):
    loss, parser = conv_fp_func(conv_params)
    weights = npr.randn(len(parser)) * 0.1
    check_grads(loss, weights)

def test_conv_net_gradient_no_layers():
    conv_params = {'num_hidden_features' : [],
               'fp_length':fp_length,
               'symmetric':False,
               'normalize':True}
    check_conv_grads(conv_params)

def test_conv_net_gradient_one_layer():
    conv_params = {'num_hidden_features' : [3],
               'fp_length':fp_length,
               'symmetric':False,
               'normalize':True}
    check_conv_grads(conv_params)

def test_conv_net_gradient():
    conv_params = {'num_hidden_features' : [3, 4],
               'fp_length':fp_length,
               'symmetric':True,
               'normalize':True}
    check_conv_grads(conv_params)

def test_conv_net_gradient_nosymm():
    conv_params = {'num_hidden_features' : [2, 3],
               'fp_length':fp_length,
               'symmetric':False,
               'normalize':True}
    check_conv_grads(conv_params)


