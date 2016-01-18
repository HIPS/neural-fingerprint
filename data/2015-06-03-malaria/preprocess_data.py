# Dougal Maclaurin
# David Duvenaud
# 2014

# Preprocessing Johannes Hachmann's CEP data
# Inputs: files 'homo_dump.dat.tbz' and 'lumo_dump.dat.tbz'
# Outputs: csv of 30k random molecules, and their energy gaps

import subprocess
import numpy as np
import numpy.random as npr
import matplotlib.pyplot as plt
import copy

from deepmolecule.data_util import (slicedict, filter_dict, has_valid_shape,
                                    has_duplicates, csv_to_dict, randomize_order,
                                    valid_smiles, dict_to_csv)
N_dpts = 10**4

data = csv_to_dict("raw_csv.csv", [4, 3], ['smiles', 'activity'], [str, float],
                   header=True, delimiter=',', skipinitialspace=True)
data['activity'] = np.log(data['activity'])
data_subsample = slicedict(randomize_order(data), slice(N_dpts))
data_subsample = filter_dict(data_subsample, 'smiles', valid_smiles)
assert not has_duplicates(data_subsample['smiles'])
assert has_valid_shape(data_subsample)
dict_to_csv(data_subsample, "malaria-processed.csv")

for name, vals in data.iteritems():
    if name in ['smiles']:
        continue  # Not-numeric so don't worry
    fig = plt.figure()
    plt.hist(vals, 50)
    plt.xlabel(name)
    plt.ylabel("Frequency")
    plt.savefig('histograms/' + name + '.png')
    plt.close()
