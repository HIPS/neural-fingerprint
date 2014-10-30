import numpy as np
import sys
sys.path.append('../../Kayak/')
from contextlib import contextmanager
import kayak
from time import time

def normalize_array(A):
    mean, std = np.mean(A), np.std(A)
    A_normed = (A - mean) / std
    def restore_function(X):
        return X * std + mean

    return A_normed, restore_function

def batch_generator(batch_size, total_size):
    start = 0
    end = batch_size
    batches = []
    while True:
        if start >= total_size:
            break
        batches.append(slice(start, end))
        start += batch_size
        end += batch_size

    return batches

@contextmanager
def tictoc():
    print "--- Start clock ---"
    t1 = time()
    yield
    dt = time() - t1
    print "--- Stop clock: %s seconds elapsed ---" % dt

# "Counterfactual value" - helper function to allow testing different inputs
def c_value(k_node, node_values):
    node_old_values = {}
    for node, new_value in node_values.iteritems():
        node_old_values[node] = node.value
        node.value = new_value

    output_value = c_value.value

    for node, old_value in node_old_values.iteritems():
        node.value = old_value

    return output_value
