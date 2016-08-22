import os, sys, subprocess, time, datetime
from neuralfingerprint import (build_morgan_deep_net,  build_conv_deep_net,
                               normalize_array, adam, build_batched_grad,
                               mean_squared_error, binary_classification_nll,
                               load_data_slices, build_mean_predictor)
import json
from autograd import grad
import autograd.numpy as np
import autograd.numpy.random as npr
log = sys.stderr.write

"""Reads params, train_data, test_data as a line_pickle from stdin. Here's an example:
params = dict(num_records = 20,
              model = dict(net_type = 'morgan',   # 'morgan' | 'conv'
                           fp_length = 512,
                           fp_depth = 5,
                           conv_width = 20,       # conv net only
                           h1_size = 300,
                           L2_reg = np.exp(-6),),
              train = dict(num_iters = 100,
                           batch_size = 100,
                           init_scale = np.exp(-4),
                           step_size = np.exp(-5),
                           seed = 0,)
              task = dict(name = 'delaney',
                          train_slices = [[0, 800]],
                          test_slices  = [[800, 1000]],))
"""

datasets_info = dict(
    delaney = dict(
        nll_func = mean_squared_error,
        target_name = 'measured log solubility in mols per litre',
        data_file = '2015-05-24-delaney/delaney-processed.csv'),
    toxin = dict(
        nll_func = binary_classification_nll,
        target_name = 'target',
        data_file = '2015-05-22-tox/sr-mmp.smiles-processed.csv'),
    malaria = dict(
        nll_func = mean_squared_error,
        target_name = 'activity',
        data_file = '2015-06-03-malaria/malaria-processed.csv'),
    cep = dict(
        nll_func = mean_squared_error,
        target_name = 'PCE',
        data_file = '2015-06-02-cep-pce/cep-processed.csv'))

def main(params):
    train_data, test_data, nll_func = load_task_data(**params['task'])
    log('Loaded {} train data points and {} test data points. Running'
        .format(len(train_data[0]), len(test_data[0])))
    net_objects = build_predictor(nll_func=nll_func, **params['model'])
    def compute_nll(predictor, inputs, targets):
        return nll_func(predictor(inputs), targets)
    num_iters, num_records = params['train']['num_iters'], params['num_records']
    record_idxs = set(map(int, np.linspace(num_iters - 1, 0, num_records)))
    training_curve = []
    def callback(predictor, i):
        if i in record_idxs:
            log(".")
            training_curve.append( (i, compute_nll(predictor, *train_data),
                                    compute_nll(predictor, *test_data )) )
    start_time = time.time()
    train_nn(net_objects, train_data[0], train_data[1], callback,
             normalize_outputs = (nll_func == mean_squared_error), **params['train'])
    stats = dict(minutes_duration = (time.time() - start_time) / 60.0,
                 timestamp        = str(datetime.datetime.now()),
                 host_name        = subprocess.check_output(['hostname'])[:-1],
                 training_curve   = training_curve)
    log("Done!\n")
    return params, stats

def build_predictor(net_type, fp_length, fp_depth, conv_width, h1_size, L2_reg, nll_func):
    if net_type == 'mean':
        return build_mean_predictor(nll_func)
    elif net_type == 'conv_plus_linear':
        vanilla_net_params = dict(layer_sizes = [fp_length],
                                  normalize=True, L2_reg = L2_reg, nll_func=nll_func)
        conv_params = dict(num_hidden_features = [conv_width] * fp_depth,
                           fp_length = fp_length)
        return build_conv_deep_net(conv_params, vanilla_net_params)
    elif net_type == 'morgan_plus_linear':
        vanilla_net_params = dict(layer_sizes = [fp_length],
                                  normalize=True, L2_reg = L2_reg, nll_func=nll_func)
        return build_morgan_deep_net(fp_length, fp_depth, vanilla_net_params)
    elif net_type == 'conv_plus_net':
        vanilla_net_params = dict(layer_sizes = [fp_length, h1_size],
                                  normalize=True, L2_reg = L2_reg, nll_func=nll_func)
        conv_params = dict(num_hidden_features = [conv_width] * fp_depth,
                           fp_length = fp_length)
        return build_conv_deep_net(conv_params, vanilla_net_params)
    elif net_type == 'morgan_plus_net':
        vanilla_net_params = dict(layer_sizes = [fp_length, h1_size],
                                  normalize=True, L2_reg = L2_reg, nll_func=nll_func)
        return build_morgan_deep_net(fp_length, fp_depth, vanilla_net_params)
    else:
        raise Exception("Unknown network type.")

def train_nn(net_objects, smiles, raw_targets, callback, normalize_outputs,
             seed, init_scale, batch_size, num_iters, **opt_params):
    loss_fun, pred_fun, net_parser = net_objects
    init_weights = init_scale * npr.RandomState(seed).randn(len(net_parser))
    if normalize_outputs:
        targets, undo_norm = normalize_array(raw_targets)
    else:
        targets, undo_norm = raw_targets, lambda x : x
    def make_predict_func(new_weights):
        return lambda new_smiles : undo_norm(pred_fun(new_weights, new_smiles))

    def opt_callback(weights, i):
        callback(make_predict_func(weights), i)

    grad_fun = build_batched_grad(grad(loss_fun), batch_size, smiles, targets)
    trained_weights = adam(grad_fun, init_weights, callback=opt_callback,
                           num_iters=num_iters, **opt_params)
    return trained_weights

def load_task_data(name, train_slices, test_slices):
    dataset_info = datasets_info[name]
    data_dir = os.path.join(os.path.dirname(__file__), '../data/')
    full_data_path = os.path.join(data_dir, dataset_info['data_file'])

    train_data, test_data = load_data_slices(
        full_data_path,
        [[slice(*bounds) for bounds in train_slices],
         [slice(*bounds) for bounds in test_slices ]],
        input_name='smiles',
        target_name=dataset_info['target_name'])

    return train_data, test_data, dataset_info['nll_func']

if __name__ == '__main__':
    # Takes in serialized hyperparameters, and outputs serialized training and test statistics.
    json.dump(main(json.load(sys.stdin)), sys.stdout, indent=4, sort_keys=True)
