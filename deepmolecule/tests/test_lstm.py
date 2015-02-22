
from deepmolecule import build_lstm, rms_prop, conj_grad, build_rnn

import numpy as np
import numpy.random as npr

def one_hot(x, K):
    return np.array(x[:,None] == np.arange(K)[None, :], dtype=int)

def generate_counting_example(length, input_size):
    """Task is to count the number of zeros."""
    seq = npr.randint(low=0, high=input_size, size=(length,))
    answer = np.sum(seq == 0)
    return one_hot(seq, input_size), float(answer)

def generate_parens_example(length, input_size):
    """Task is to check whether the number of zeros equals number of ones."""
    seq = npr.randint(low=0, high=input_size, size=(length,))
    answer = (np.sum(seq == 0) == np.sum(seq == 1))
    return one_hot(seq, input_size), float(answer)

def build_dataset(N, seq_length, seq_width, example_generator):
    seqs = np.zeros((seq_length, N, seq_width))
    targets = np.zeros((N, 1))
    for ix in xrange(N):
        cur_seq, cur_target = example_generator(seq_length, seq_width)
        seqs[:, ix, :] = cur_seq
        targets[ix] = cur_target
    return seqs, targets

def test_lstm():
    npr.seed(1)
    N_train = 1000
    N_test = 100
    state_size = 3
    seq_width = 2
    seq_length = 4
    output_size = 1
    #datagen_fun = generate_counting_example
    datagen_fun = generate_parens_example

    train_seqs, train_targets = build_dataset(N_train, seq_length, seq_width, datagen_fun)
    test_seqs,  test_targets  = build_dataset(N_test,  seq_length, seq_width, datagen_fun)

    #loss_fun, grad_fun, pred_fun, hidden_fun, parser = \
    #    build_lstm(seq_width, state_size, output_size, l2_penalty=0.0)
    loss_fun, grad_fun, pred_fun, hidden_fun, parser = \
        build_rnn(seq_width, state_size, output_size, l2_penalty=0.0)

    def training_grad_with_idxs(idxs, weights):
        return grad_fun(weights, train_seqs[idxs], train_targets[idxs])
    def training_loss_with_idxs(idxs, weights):
        return loss_fun(weights, train_seqs[idxs], train_targets[idxs])
    def training_grad_all(weights):
        return grad_fun(weights, train_seqs, train_targets)
    def training_loss_all(weights):
        return loss_fun(weights, train_seqs, train_targets)

    def pred_rmse(weights, seqs, targets):
        preds = pred_fun(weights, seqs)
        return np.sqrt(np.mean((preds - targets)**2))

    def test_accuracy(weights):
        return pred_rmse(weights, test_seqs, test_targets)
    def train_accuracy(weights):
        return pred_rmse(weights, train_seqs, train_targets)

    print "Random accuracy: ", test_accuracy(npr.randn(len(parser.vect)))

    def callback(epoch, weights):
        print "Epoch", epoch, "Train loss: ", loss_fun(weights, train_seqs, train_targets), \
                               "Test error: ", test_accuracy(weights)

    #trained_weights = rms_prop(training_grad_with_idxs, N_train, parser.N, callback,
    #                           learn_rate = 0.001)

    trained_weights = conj_grad(training_loss_all, training_grad_all,
                                len(parser.vect), callback, param_scale = 0.1)

test_lstm()
