import autograd.numpy as np
import autograd.numpy.random as npr

import sys, signal, pickle
from contextlib import contextmanager
from time import time
from functools import partial
from collections import OrderedDict

def collect_test_losses(num_folds):
    # Run this after CV results are in. e.g:
    # python -c "from deepmolecule.util import collect_test_losses; collect_test_losses(10)"
    results = {}
    for net_type in ['conv', 'morgan']:
        results[net_type] = []
        for expt_ix in range(num_folds):
            fname = "Final_test_loss_{0}_{1}.pkl.save".format(expt_ix, net_type)
            try:
                with open(fname) as f:
                    results[net_type].append(pickle.load(f))
            except IOError:
                print "Couldn't find file {0}".format(fname)

    print "Results are:"
    print results
    print "Means:"
    print {k : np.mean(v) for k, v in results.iteritems()}
    print "Std errors:"
    print {k : np.std(v) / np.sqrt(len(v) - 1) for k, v in results.iteritems()}

def record_loss(loss, expt_ix, net_type):
    fname = "Final_test_loss_{0}_{1}.pkl.save".format(expt_ix, net_type)
    with open(fname, 'w') as f:
        pickle.dump(float(loss), f)

def N_fold_split(N_folds, fold_ix, N_data):
    fold_ix = fold_ix % N_folds
    fold_size = N_data / N_folds
    test_fold_start = fold_size * fold_ix
    test_fold_end   = fold_size * (fold_ix + 1)
    test_ixs  = range(test_fold_start, test_fold_end)
    train_ixs = range(0, test_fold_start) + range(test_fold_end, N_data)
    return train_ixs, test_ixs

def rmse(X, Y):
    return np.sqrt(np.mean((X - Y)**2))

def slicedict(d, ixs):
    return {k : v[ixs] for k, v in d.iteritems()}

class memoize(object):
    def __init__(self, func):
        self.func = func
        self.cache = {}

    def __call__(self, *args):
        if args in self.cache:
            return self.cache[args]
        else:
            result = self.func(*args)
            self.cache[args] = result
            return result

    def __get__(self, obj, objtype):
        return partial(self.__call__, obj)

def normalize_array(A):
    mean, std = np.mean(A), np.std(A)
    A_normed = (A - mean) / std
    def restore_function(X):
        return X * std + mean

    return A_normed, restore_function

@contextmanager
def tictoc():
    print "--- Start clock ---"
    t1 = time()
    yield
    dt = time() - t1
    print "--- Stop clock: %s seconds elapsed ---" % dt

class WeightsParser(object):
    """A kind of dictionary of weights shapes,
       which can pick out named subsets from a long vector.
       Does not actually store any weights itself."""
    def __init__(self):
        self.idxs_and_shapes = OrderedDict()
        self.N = 0

    def add_weights(self, name, shape):
        start = self.N
        self.N += np.prod(shape)
        self.idxs_and_shapes[name] = (slice(start, self.N), shape)

    def get(self, vect, name):
        """Takes in a vector and returns the subset indexed by name."""
        idxs, shape = self.idxs_and_shapes[name]
        return np.reshape(vect[idxs], shape)

    def set(self, vect, name, value):
        """Takes in a vector and returns the subset indexed by name."""
        idxs, _ = self.idxs_and_shapes[name]
        vect[idxs] = np.ravel(value)

    def __len__(self):
        return self.N

def one_of_k_encoding(x, allowable_set):
    if x not in allowable_set:
        raise Exception("input {0} not in allowable set{1}:".format(x, allowable_set))
    return map(lambda s: x == s, allowable_set)

def one_of_k_encoding_unk(x, allowable_set):
    """Maps inputs not in the allowable set to the last element."""
    if x not in allowable_set:
        x = allowable_set[-1]
    return map(lambda s: x == s, allowable_set)

def dropout(weights, fraction, random_state):
    """Randomly sets fraction of weights to zero, and increases the rest
        such that the expected activation is the same."""
    mask = random_state.rand(len(weights)) > fraction
    return weights * mask / (1 - fraction)

def get_ith_minibatch_ixs(i, num_datapoints, batch_size):
    num_minibatches = num_datapoints / batch_size + ((num_datapoints % batch_size) > 0)
    i = i % num_minibatches
    start = i * batch_size
    stop = start + batch_size
    return slice(start, stop)

def build_batched_grad(grad, batch_size, inputs, targets):
    """Grad has signature(weights, inputs, targets)."""
    def batched_grad(weights, i):
        cur_idxs = get_ith_minibatch_ixs(i, len(targets), batch_size)
        return grad(weights, inputs[cur_idxs], targets[cur_idxs])
    return batched_grad

def add_dropout(grad, dropout_fraction, seed=0):
    """Actually, isn't this dropconnect?"""
    assert(dropout_fraction < 1.0)
    def dropout_grad(weights, i):
        mask = npr.RandomState(seed * 10**6 + i).rand(len(weights)) > dropout_fraction
        masked_weights = weights * mask / (1 - dropout_fraction)
        return grad(masked_weights, i)
    return dropout_grad

def catch_errors(run_fun, catch_fun):
    def signal_term_handler(signal, frame):
        catch_fun()
        sys.exit(0)
    signal.signal(signal.SIGTERM, signal_term_handler)
    try:
        result = run_fun()
    except:
        catch_fun()
        raise

    return result