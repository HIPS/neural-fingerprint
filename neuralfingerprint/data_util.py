import csv
import numpy as np
import numpy.random as npr

from util import slicedict
from rdkit_utils import smile_to_fp

def randomize_order(data):
    data = {k : np.array(v) for k, v in data.iteritems()} # To array for fancy indexing
    N = len(data.values()[0])
    rand_ix = np.arange(N)
    npr.RandomState(0).shuffle(rand_ix)
    return slicedict(data, rand_ix)

def csv_to_dict(fname, colnums, colnames, coltypes, header=False, **kwargs):
    data = {name : [] for name in colnames}
    with open(fname) as f:
        reader = csv.reader(f, **kwargs)
        for rownum, row in enumerate(reader):
            if header and rownum == 0:
                continue
            try:
                for colname, colnum, coltype in zip(colnames, colnums, coltypes):
                    data[colname].append(coltype(row[colnum]))
            except:
                print "Couldn't parse row {0}".format(rownum)
    return data

def filter_on_other(predicate, test_values, return_values):
    return map(lambda x : x[1],
               filter(lambda x : predicate(test_values[x[0]]),
                      enumerate(return_values)))

def filter_dict(data, field, predicate):
    filter_vals = data[field]
    return {k : filter_on_other(predicate, filter_vals, v)
            for k, v in data.iteritems()}

def has_valid_shape(data):
    valid = True
    colnames = data.keys()
    N = len(data[colnames[0]])
    for name in colnames[1:]:
        valid = valid and (len(data[name]) == N)
    return valid

def dict_to_csv(data, fname):
    colnames = data.keys()
    with open(fname, 'w') as f:
        writer = csv.writer(f, delimiter=',', quotechar='\'')
        writer.writerow(colnames)
        for row in zip(*(data[name] for name in colnames)):
            writer.writerow(row)

def valid_smiles(smile):
    # Tests whether rdkit can parse smiles
    try:
        smile_to_fp(smile, 1, 10)
        return True
    except:
        print "Couldn't parse", smile
        return False

def has_duplicates(A):
    return len(set(A)) != len(A)

def remove_duplicates(values, key_lambda):
    output = []
    seen = set()
    for value in values:
        # If value has not been encountered yet, add it to both list and set.
        cur_key = key_lambda(value)
        if cur_key not in seen:
            output.append(value)
            seen.add(cur_key)
    return output
