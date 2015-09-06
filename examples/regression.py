# Example regression script using neural fingerprints.
#
# Compares Morgan fingerprints to neural fingerprints.

import autograd.numpy as np
import autograd.numpy.random as npr

from neuralfingerprint import load_data
from neuralfingerprint import build_morgan_deep_net
from neuralfingerprint import build_conv_deep_net
from neuralfingerprint import normalize_array, adam
from neuralfingerprint import build_batched_grad
from neuralfingerprint.util import rmse

from autograd import grad

task_params = {'N_data'     : 1120,
               'target_name' : 'measured log solubility in mols per litre',
               'data_file'   : 'delaney.csv'}

if True:
    num_iters = 10
    N_train = 80
    N_val   = 20
    N_test  = 20
else:
    num_iters = 2000
    N_train = 800
    N_val = 120
    N_test = 200

num_folds = 2
model_params = dict(net_type = 'conv',   # 'morgan' | 'conv'
                    fp_length = 512,
                    fp_depth = 4,
                    conv_width = 20,       # conv net only
                    h1_size = 100,
                    L2_reg = -2)
train_params = dict(num_iters = num_iters,
                    batch_size = 100,
                    init_scale = np.exp(-4),
                    step_size = np.exp(-6),
                    b1 = np.exp(-3),
                    b2 = np.exp(-2),
                    seed = 0)

# Define the architecture of the network that sits on top of the fingerprints.
vanilla_net_params = dict(
    layer_sizes = [model_params['fp_length'], model_params['h1_size']],
                   normalize=True, L2_reg = model_params['L2_reg'], nll_func = rmse)

def train_nn(pred_fun, loss_fun, num_weights, train_smiles, train_raw_targets, train_params,
             validation_smiles=None, validation_raw_targets=None):
    """loss_fun has inputs (weights, smiles, targets)"""
    print "Total number of weights in the network:", num_weights
    npr.seed(0)
    init_weights = npr.randn(num_weights) * train_params['init_scale']

    train_targets, undo_norm = normalize_array(train_raw_targets)
    training_curve = []
    def callback(weights, iter):
        if iter % 10 == 0:
            print "max of weights", np.max(np.abs(weights))
            train_preds = undo_norm(pred_fun(weights, train_smiles))
            cur_loss = loss_fun(weights, train_smiles, train_targets)
            training_curve.append(cur_loss)
            print "Iteration", iter, "loss", cur_loss, "train RMSE", \
                np.sqrt(np.mean((train_preds - train_raw_targets)**2)),
            if validation_smiles is not None:
                validation_preds = undo_norm(pred_fun(weights, validation_smiles))
                print "Validation RMSE", iter, ":", \
                    np.sqrt(np.mean((validation_preds - validation_raw_targets) ** 2)),

    # Build gradient using autograd.
    grad_fun = grad(loss_fun)
    grad_fun_with_data = build_batched_grad(grad_fun, train_params['batch_size'],
                                            train_smiles, train_targets)

    # Optimize weights.
    trained_weights = adam(grad_fun_with_data, init_weights, callback=callback,
                           num_iters=num_iters, step_size=train_params['step_size'],
                           b1=train_params['b1'], b2=train_params['b2'])

    def predict_func(new_smiles):
        """Returns to the original units that the raw targets were in."""
        return undo_norm(pred_fun(trained_weights, new_smiles))
    return predict_func, trained_weights, training_curve


def main():
    print "Loading data..."
    traindata, valdata, testdata = load_data(
        task_params['data_file'], (N_train, N_val, N_test),
        input_name='smiles', target_name=task_params['target_name'])
    train_inputs, train_targets = traindata
    val_inputs,   val_targets   = valdata
    test_inputs,  test_targets  = testdata

    def print_performance(pred_func):
        train_preds = pred_func(train_inputs)
        val_preds = pred_func(val_inputs)
        print "\nPerformance (RMSE) on " + task_params['target_name'] + ":"
        print "Train:", rmse(train_preds, train_targets)
        print "Test: ", rmse(val_preds,  val_targets)
        print "-" * 80
        return rmse(val_preds, val_targets)

    def run_morgan_experiment():
        loss_fun, pred_fun, net_parser = \
            build_morgan_deep_net(model_params['fp_length'],
                                  model_params['fp_depth'], vanilla_net_params)
        num_weights = len(net_parser)
        predict_func, trained_weights, conv_training_curve = \
            train_nn(pred_fun, loss_fun, num_weights, train_inputs, train_targets,
                     train_params, validation_smiles=val_inputs, validation_raw_targets=val_targets)
        return print_performance(predict_func)

    def run_conv_experiment():
        conv_layer_sizes = [model_params['conv_width']] * model_params['fp_depth']
        conv_arch_params = {'num_hidden_features' : conv_layer_sizes,
                            'fp_length' : model_params['fp_length'], 'normalize' : 1}
        loss_fun, pred_fun, conv_parser = \
            build_conv_deep_net(conv_arch_params, vanilla_net_params, model_params['L2_reg'])
        num_weights = len(conv_parser)
        predict_func, trained_weights, conv_training_curve = \
            train_nn(pred_fun, loss_fun, num_weights, train_inputs, train_targets,
                     train_params, validation_smiles=val_inputs, validation_raw_targets=val_targets)
        test_predictions = predict_func(test_inputs)
        return rmse(test_predictions, test_targets)

    print "Task params", task_params
    print
    print "Starting Morgan fingerprint experiment..."
    test_loss_morgan = run_morgan_experiment()
    print "Starting neural fingerprint experiment..."
    test_loss_neural = run_conv_experiment()
    print
    print "Morgan test RMSE:", test_loss_morgan, "Neural test RMSE:", test_loss_neural

if __name__ == '__main__':
    main()
