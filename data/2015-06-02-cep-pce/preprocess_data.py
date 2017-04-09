# Dougal Maclaurin
# David Duvenaud
# 2014

# Preprocessing Johannes Hachmann's CEP data
# Inputs: files 'homo_dump.dat.tbz' and 'lumo_dump.dat.tbz'
# Outputs: csv of 30k random molecules, and their energy gaps

import csv
import subprocess
import numpy as np
import numpy.random as npr
import matplotlib.pyplot as plt
import copy

from deepmolecule.rdkit_utils import smile_to_fp
from rdkit.Chem import Descriptors
from rdkit import Chem
from rdkit.Chem import rdMolDescriptors

N_dpts = 3 * 10**4
slicedict = lambda d, ixs : {k : v[ixs] for k, v in d.iteritems()}

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

def merge_dicts(d1, d2, merge_col):
    assert d1[merge_col] == d2[merge_col]
    d_out = d1.copy()
    d_out.update(d2)
    return d_out

def has_duplicates(A):
    return len(set(A)) != len(A)

def valid_smiles(smile):
    # Tests whether rdkit can parse smiles
    try:
        smile_to_fp(smile, 1, 10)
        return True
    except:
        print "Couldn't parse", smile
        return False

def valid_PCE(pce):
    if -10.0 <= pce <= 100.0:
        return True
    else:
        print "Invalid pce", pce
        return False

subprocess.call(["tar", "-zxvf", "data_cep.tar.gz", "--directory", "/tmp"])
data = csv_to_dict("/tmp/data_tmp_moldata.csv", [0, 1], ['smiles', 'PCE'], [str, float],
                        header=True, delimiter=',', skipinitialspace=True)
data_subsample = slicedict(randomize_order(data), slice(N_dpts))
data_subsample = filter_dict(data_subsample, 'PCE', valid_PCE)
data_subsample = filter_dict(data_subsample, 'smiles', valid_smiles)
assert not has_duplicates(data_subsample['smiles'])
assert has_valid_shape(data_subsample)
dict_to_csv(data_subsample, "cep-processed.csv")

for name, vals in data.iteritems():
    if name in ['smiles']:
        continue  # Not-numeric so don't worry
    fig = plt.figure()
    plt.hist(vals, 50)
    plt.xlabel(name)
    plt.ylabel("Frequency")
    plt.savefig('histograms/' + name + '.png')
    plt.close()
