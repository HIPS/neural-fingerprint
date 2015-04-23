# Compares the predictive accuracy of using
# standard fingerprints versus custom convolutional nets.
#
# Dougal Maclaurin
# David Duvenaud
# Ryan P. Adams
#
# Sept 2014

import sys, os
import numpy as np
import numpy.random as npr

from deepmolecule import tictoc, normalize_array, sgd_with_momentum, get_data_file, load_data
from deepmolecule import build_morgan_deep_net, build_morgan_flat_net
from deepmolecule import build_convnet, output_dir, get_output_file
from deepmolecule import plot_predictions, plot_maximizing_inputs

# for old net
import kayak
from deepmolecule import smiles_to_fps

datadir = '/Users/dkd/Dropbox/Molecule_ML/data/Samsung_September_8_2014/davids-validation-split/'
testfile = datadir + 'test_split.csv'
trainfile = datadir + 'train_split_plus_methyls.csv'

def train_nn(net_builder_fun, smiles, raw_targets, arch_params, train_params,
             validation_smiles=None, validation_targets=None):
    npr.seed(1)
    targets, undo_norm = normalize_array(raw_targets)
    loss_fun, grad_fun, pred_fun, _, N_weights = net_builder_fun(**arch_params)
    def callback(epoch, weights):
        if epoch % 10 == 0:
            train_preds = undo_norm(pred_fun(weights, smiles))
            print "\nTraining RMSE after epoch", epoch, ":",\
                np.sqrt(np.mean((train_preds - raw_targets)**2)),
            if validation_smiles is not None:
                validation_preds = undo_norm(pred_fun(weights, validation_smiles))
                print "Validation RMSE", epoch, ":",\
                    np.sqrt(np.mean((validation_preds - validation_targets)**2)),
        else:
            print ".",
    grad_fun_with_data = lambda idxs, w : grad_fun(w, smiles[idxs], targets[idxs])
    #print "loss:", loss_fun()
    #print grad_fun_with_data(slice(None), npr.randn(N_weights) * )
    trained_weights = sgd_with_momentum(grad_fun_with_data, len(targets), N_weights,
                                        callback, **train_params)
    return lambda new_smiles : undo_norm(pred_fun(trained_weights, new_smiles)), trained_weights




def train_2layer_nn(smiles, targets, arch_params, train_params):
    num_epochs   = train_params['num_epochs']
    batch_size   = train_params['batch_size']
    learn_rate   = train_params['learn_rate']
    momentum     = train_params['momentum']
    h1_dropout   = arch_params['h1_dropout']
    param_scale  = train_params['param_scale']
    h1_size      = arch_params['h1_size']
    fp_length    = arch_params['fp_length']
    fp_radius    = arch_params['fp_radius']

    npr.seed(1)
    print "Computing features..."
    features = smiles_to_fps(smiles, fp_length, fp_radius)
    print "Training"
    normed_targets, undo_norm = normalize_array(targets)
    batcher = kayak.Batcher(batch_size, features.shape[0])
    X =  kayak.Inputs(features, batcher)
    T =  kayak.Targets(normed_targets, batcher)
    W1 = kayak.Parameter(param_scale * npr.randn(features.shape[1], h1_size))
    B1 = kayak.Parameter(param_scale * npr.randn(1, h1_size))
    H1 = kayak.Dropout(kayak.HardReLU(kayak.MatMult(X, W1) + B1), h1_dropout, batcher=batcher, rng=npr.RandomState(1))
    W2 = kayak.Parameter(param_scale * npr.randn(h1_size))
    B2 = kayak.Parameter(param_scale * npr.randn(1))
    Y =  kayak.MatMult(H1, W2) + B2
    L =  kayak.L2Loss(Y, T)

    #print "loss:", L.value
    params = [W1, W2, B1, B2]
    #print "w1 grad", L.grad(W1)
    #print "w1 grad sum", np.sum(L.grad(W1), axis=1)
    #print "b1 grad", L.grad(B1)
    #print "w2 grad",L.grad(W2)
    #print "b2 grad", L.grad(B2)
    step_dirs = [np.zeros(p.shape) for p in params]
    for epoch in xrange(num_epochs):
        total_err = 0.0
        for batch in batcher:
            total_err += np.sum((undo_norm(Y.value) - undo_norm(T.value))**2)
            grads = [L.grad(p) for p in params]
            step_dirs = [momentum * step_dir - (1.0 - momentum) * grad
                         for step_dir, grad in zip(step_dirs, grads)]
            for p, step_dir in zip(params, step_dirs):
                p.value += learn_rate * step_dir
        if epoch % 10 == 0:
            print "\nRMSE after epoch", epoch, ":", np.sqrt(total_err / len(normed_targets)),
        else:
            print ".",

    def make_predictions(newsmiles):
        X.data = smiles_to_fps(newsmiles, fp_length, fp_radius)
        batcher.test_mode()
        return undo_norm(Y.value)

    return make_predictions



def old_nn_train(smiles, targets, arch_params, train_params):
    num_epochs   = train_params['num_epochs']
    batch_size   = train_params['batch_size']
    learn_rate   = train_params['learn_rate']
    momentum     = train_params['momentum']
    h1_dropout   = arch_params['h1_dropout']
    param_scale  = train_params['param_scale']
    h1_size      = arch_params['h1_size']
    fp_length    = arch_params['fp_length']
    fp_radius    = arch_params['fp_radius']

    features = smiles_to_fps(smiles, fp_length, fp_radius)

    npr.seed(1)

    # Normalize the outputs.
    targ_mean = np.mean(targets)
    targ_std  = np.std(targets)

    batcher = kayak.Batcher(batch_size, features.shape[0])

    X = kayak.Inputs(features, batcher)
    T = kayak.Targets((targets-targ_mean) / targ_std, batcher)

    W1 = kayak.Parameter( param_scale*npr.randn( features.shape[1], h1_size ))
    B1 = kayak.Parameter( param_scale*npr.randn( 1, h1_size ) )
    H1 = kayak.Dropout(kayak.HardReLU(kayak.MatMult(X, W1) + B1), h1_dropout,
                       rng=npr.RandomState(1), batcher=batcher)

    W2 = kayak.Parameter( param_scale*npr.randn( h1_size ) )
    B2 = kayak.Parameter( param_scale*npr.randn(1))

    Y = kayak.MatMult(H1, W2) + B2

    L = kayak.MatSum(kayak.L2Loss(Y, T))

    mom_grad_W1 = np.zeros(W1.shape)
    mom_grad_W2 = np.zeros(W2.shape)

    for epoch in xrange(num_epochs):

        total_loss = 0.0
        total_err  = 0.0
        total_data = 0

        for batch in batcher:

            total_loss += L.value
            total_err  += np.sum(np.abs(Y.value - T.value))
            total_data += T.shape[0]

            grad_W1 = L.grad(W1)
            grad_B1 = L.grad(B1)
            grad_W2 = L.grad(W2)
            grad_B2 = L.grad(B2)

            mom_grad_W1 = momentum*mom_grad_W1 + (1.0-momentum)*grad_W1
            mom_grad_W2 = momentum*mom_grad_W2 + (1.0-momentum)*grad_W2

            W1.value -= learn_rate * mom_grad_W1
            W2.value -= learn_rate * mom_grad_W2
            B1.value -= learn_rate * grad_B1
            B2.value -= learn_rate * grad_B2

        print epoch, total_err / total_data

    def compute_predictions(smiles):
        X.data = smiles_to_fps(smiles, fp_length, fp_radius)
        batcher.test_mode()
        return Y.value * targ_std + targ_mean

    return compute_predictions



def main():

    # Parameters for standard net build on Morgan fingerprints.

    morgan_train_params = {'num_epochs'  : 50,
                           'batch_size'  : 256,
                           'learn_rate'  : 0.001,
                           'momentum'    : 0.98,
                           'param_scale' : 0.1}

    #morgan_train_params = {'num_epochs'  : 50,
    #                       'batch_size'  : 200,
    #                       'learn_rate'  : 0.001604,
    #                       'momentum'    : 0.98371,
    #                       'param_scale' : 0.1}
    morgan_deep_arch_params = {'h1_size'    : 499,
                               'h1_dropout' : 0,
                               'fp_length'  : 512,
                               'fp_radius'  : 4}

    task_params = {'N_train'     : 27000,
                   'N_valid'     : 1000,
                   'N_test'      : 4000,
                   #'target_name' : 'Molecular Weight',
                   'target_name' : 'rate',
                   #'data_file'   : get_data_file('2014-11-03-all-tddft/processed.csv')}
                   'data_file'   : trainfile}

    #task_params = {'N_train'     : 20000,
    #               'N_valid'     : 1000,
    #               'N_test'      : 7000,
                   #'target_name' : 'Molecular Weight',
    #               'target_name' : 'Log Rate',
                   #'data_file'   : get_data_file('2014-11-03-all-tddft/processed.csv')}
    #               'data_file'   : trainfile}
    #target_name = 'Log Rate'
    #'Polar Surface Area'
    #'Number of Rings'
    #'Number of H-Bond Donors'
    #'Number of Rotatable Bonds'
    #'Minimum Degree'


    print "Morgan net training params: ", morgan_train_params
    print "Morgan deep net architecture params: ", morgan_deep_arch_params

    print "Task params", task_params
    print "Output directory:", output_dir()

    print "\nLoading data..."
    #traindata, valdata, testdata = load_data(task_params['data_file'],
    #                    (task_params['N_train'], task_params['N_valid'], task_params['N_test']))

    testdata = load_data(testfile, (task_params['N_test'],))[0]
    traindata, valdata = load_data(trainfile, (task_params['N_train'], task_params['N_valid']))


    train_inputs, train_targets = traindata['smiles'], traindata[task_params['target_name']]
    val_inputs, val_targets = valdata['smiles'], valdata[task_params['target_name']]
    test_inputs, test_targets = testdata['smiles'], testdata[task_params['target_name']]

    xfrm = lambda x: np.log( np.maximum(1e-26, x))
    train_targets = xfrm(train_targets)
    test_targets = xfrm(test_targets)
    val_targets = xfrm(val_targets)

    def print_performance(pred_func, filename=None):
        train_preds = pred_func(train_inputs)
        test_preds = pred_func(test_inputs)
        print "\nPerformance (RMSE) on " + task_params['target_name'] + ":"
        print "Train:", np.sqrt(np.mean((train_preds - train_targets)**2))
        print "Test: ", np.sqrt(np.mean((test_preds - test_targets)**2))
        print "-" * 80
        if filename:
            np.savez_compressed(file=get_output_file(filename),
                                train_preds=train_preds, train_targets=train_targets,
                                test_preds=test_preds, test_targets=test_targets,
                                target_name=task_params['target_name'])

    def save_net(weights, arch_params, filename):
        np.savez_compressed(file=get_output_file(filename), weights=weights,
                            arch_params=arch_params)

    print "-" * 80
    print "Mean predictor"
    y_train_mean = np.mean(train_targets)
    print_performance(lambda x : y_train_mean)

    plot_predictions(get_output_file('vanilla-predictions.npz'),
                     os.path.join(output_dir(), 'vanilla-prediction-plots'))

    print "-" * 80
    print "Old network and training:"
    #predictor = train_2layer_nn(train_inputs, train_targets, morgan_deep_arch_params, morgan_train_params)
    predictor = old_nn_train(train_inputs, train_targets, morgan_deep_arch_params, morgan_train_params)
    print_performance(predictor, 'old-vanilla-predictions')
    plot_predictions(get_output_file('old-vanilla-predictions.npz'),
                     os.path.join(output_dir(), 'vanilla-prediction-plots'))


    print "Training vanilla neural net"
    with tictoc():
        predictor, weights = train_nn(build_morgan_deep_net, train_inputs, train_targets,
                             morgan_deep_arch_params, morgan_train_params)#, val_inputs, val_targets)
        print "\n"
    print_performance(predictor, 'vanilla-predictions')
    plot_predictions(get_output_file('vanilla-predictions.npz'),
                     os.path.join(output_dir(), 'vanilla-prediction-plots'))
    save_net(weights, morgan_deep_arch_params, 'morgan-net-weights')
    plot_maximizing_inputs(build_morgan_deep_net, get_output_file('morgan-net-weights.npz'),
                           os.path.join(output_dir(), 'morgan-features'))

if __name__ == '__main__':
    sys.exit(main())
