import autograd.numpy as np
import autograd.numpy.random as npr

from contextlib import contextmanager
from time import time
from functools import partial
from collections import OrderedDict

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
    assert(dropout_fraction < 1.0)
    def dropout_grad(weights, i):
        mask = npr.RandomState(seed * 10**6 + i).rand(len(weights)) > dropout_fraction
        masked_weights = weights * mask / (1 - dropout_fraction)
        return grad(masked_weights, i)
    return dropout_grad
