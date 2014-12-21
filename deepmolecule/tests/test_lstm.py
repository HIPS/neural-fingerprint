
from deepmolecule import build_lstm_rnn, rms_prop

import numpy as np
import numpy.random as npr

def generate_counting_example(length=30, input_size=9):
    #length = npr.randint(low=10, high=30)
    seq = npr.randint(low=0, high=input_size, size=(length,1))
    answer = np.sum(seq == 1)
    return seq, answer

def generate_parens_example(length=30, input_size=9):
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
    N_train = 1000
    N_test = 100
    state_size = 20
    input_size = 9
    seq_length = 30
    datagen_fun = generate_counting_example

    training_seqs, training_targets = build_dataset(N_train, seq_length, datagen_fun)
    test_seqs, test_targets = build_dataset(N_test, seq_length, datagen_fun)

    loss_fun, grad_fun, pred_fun, hidden_fun, parser = \
        build_lstm_rnn(input_size, state_size, l2_penalty=0.0)

    def training_grad_with_idxs(idxs, weights):
        return grad_fun(weights, training_seqs[idxs], training_targets[idxs])

    def test_accuracy(weights):
        return loss_fun(weights, test_seqs, test_targets)

    print "Random accuracy: ", test_accuracy(npr.randn(parser.N))

    def callback(epoch, weights):
        print "Epoch", epoch, "Test accuracy: ", test_accuracy(weights)

    trained_weights = rms_prop(training_grad_with_idxs, N_train, parser.N, callback)

test_lstm()