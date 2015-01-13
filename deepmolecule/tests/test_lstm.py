
from deepmolecule import build_lstm_rnn, rms_prop, conj_grad, build_vanilla_rnn

import numpy as np
import numpy.random as npr


def generate_counting_example(length=15, input_size=4):
    #length = npr.randint(low=10, high=30)
    seq = npr.randint(low=0, high=input_size, size=(length,1))
    answer = np.sum(seq == 1)
    return seq, answer

def generate_summing_example(length=15, input_size=4):
    #length = npr.randint(low=10, high=30)
    seq = npr.rand(low=0, high=input_size, size=(length,1))
    answer = np.sum(seq == 1)
    return float(seq), float(answer)

def generate_parens_example(length=15, input_size=4):
    #length = npr.randint(low=10, high=30)
    seq = npr.randint(low=0, high=input_size, size=(length,1))
    answer = np.sum(seq == 1) - 2.5*np.sum(seq == 2)
    return seq, answer

def build_dataset(N, seq_length, example_generator):
    seqs = np.zeros((N, seq_length))
    targets = np.zeros((N))
    for ix in xrange(N):
        cur_seq, cur_target = example_generator(seq_length)
        seqs[ix, :] = np.squeeze(cur_seq[:,])
        targets[ix] = cur_target
    return seqs, targets

def test_lstm():
    npr.seed(1)
    N_train = 1000
    N_test = 100
    state_size = 2
    input_size = 3
    seq_length = 2
    datagen_fun = generate_counting_example
    #datagen_fun = generate_parens_example

    train_seqs, train_targets = build_dataset(N_train, seq_length, datagen_fun)
    test_seqs,  test_targets  = build_dataset(N_test,  seq_length, datagen_fun)

#    loss_fun, grad_fun, pred_fun, hidden_fun, parser = \
#        build_lstm_rnn(input_size, state_size, l2_penalty=0.0)
    loss_fun, grad_fun, pred_fun, hidden_fun, parser = \
        build_vanilla_rnn(input_size, state_size, l2_penalty=0.0)

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

    print "Random accuracy: ", test_accuracy(npr.randn(parser.N))

    def callback(epoch, weights):
        print "Epoch", epoch, "Train loss: ", loss_fun(weights, train_seqs, train_targets), \
                               "Test error: ", test_accuracy(weights)

    #trained_weights = rms_prop(training_grad_with_idxs, N_train, parser.N, callback,
    #                           learn_rate = 0.001)

    trained_weights = conj_grad(training_loss_all, training_grad_all,
                                parser.N, callback)

test_lstm()
