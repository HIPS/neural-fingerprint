# Dougal Maclaurin
# David Duvenaud
# 2014

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from rdkit.Chem import Descriptors
from rdkit import Chem
from rdkit.Chem import rdMolDescriptors

def slicedict(d, ixs):
    newd = {}
    for key in d:
        newd[key] = d[key][ixs]
    return newd

def randomize_order(data):
    N = len(data.values()[0])
    rand_ix = np.arange(N)
    np.random.seed(1)
    np.random.shuffle(rand_ix)
    return slicedict(data, rand_ix)

def minimum_degree_in_molecule(mol):
    degrees = [atom.GetDegree() for atom in mol.GetAtoms()]
    return np.min(degrees)

# Data from http://pubs.acs.org/doi/suppl/10.1021/ci034243x/suppl_file/ci034243xsi20040112_053635.txt
infilename = 'ci034243xsi20040112_053635.txt'
outfilename = 'delaney-processed.csv'
removed_outfilename = 'excluded_molecules.csv'
pandas_data = pd.io.parsers.read_csv(infilename, sep=',')

fields = ['Compound ID', 'measured log solubility in mols per litre', 'ESOL predicted log solubility in mols per litre', 'smiles']

rdkit_functions = {'Molecular Weight' : Descriptors.MolWt,
                   'Polar Surface Area' : rdMolDescriptors.CalcTPSA,
                   'Number of Rings' : rdMolDescriptors.CalcNumRings,
                   'Number of H-Bond Donors' : rdMolDescriptors.CalcNumHBD,
                   'Number of Rotatable Bonds' : rdMolDescriptors.CalcNumRotatableBonds,
                   'Minimum Degree' : minimum_degree_in_molecule}

print "Loading raw file..."
data = {field: np.array(pandas_data[field]) for field in fields}
data = randomize_order(data)

print "Computing molecule graphs from SMILES..."
mols = map(Chem.MolFromSmiles, data['smiles'])
print "Computing RDKit features on each molecule..."
for feature_name, fun in rdkit_functions.iteritems():
    print "Computing", feature_name, "..."
    data[feature_name] = np.array(map(fun, mols))

#transforms = [( 'rate', lambda x: np.log(np.maximum(x, 1e-25)), 'Log Rate'),
#              ( 'strength', lambda x: np.log(np.maximum(x, 1e-10)), 'Log Strength')]

#print "Transforming outputs..."
#for feature_name, fun, transformed_name in transforms:
#    print "Transforming", feature_name, "..."
#    data[transformed_name] = np.array(map(fun, data[feature_name]))

print "Identifying duplicates...",
all_inchis = {}
duplicated_mols = []
is_dup = [False] * len(data.values()[0])
for i, inchi in enumerate(data['smiles']):
    if inchi in all_inchis:
        is_dup[i] = True
    all_inchis[inchi] = True
print np.sum(is_dup), "found."

print "Identifying bad values...",
is_finite = [True] * len(data.values()[0])
for name, vals in data.iteritems():
    if name == 'measured log solubility in mols per litre':
        is_finite = np.logical_and(is_finite, np.isfinite(vals))
print np.sum(np.logical_not(is_finite)), " found."

#print "Finding molecules with atoms having 0 degree...",
#degree_zero = data['Minimum Degree'] == 0
#print np.sum(degree_zero), "found."

good_ixs = is_finite & np.logical_not(is_dup) # & np.logical_not(degree_zero)
good_data = slicedict(data, good_ixs)
bad_data = slicedict(data, np.logical_not(good_ixs))
print "Datapoints thrown away:", np.sum(np.logical_not(good_ixs))
print "Datapoints kept:", np.sum(good_ixs)

print "Writing output to file..."
pd.DataFrame(good_data).to_csv(outfilename, sep=',', header=True, index=False)
pd.DataFrame(bad_data).to_csv(removed_outfilename, sep=',', header=True, index=False)

print "Making histograms of all features..."
for name, vals in data.iteritems():
    if name in ['Compound ID', 'smiles']:
        continue  # Not-numeric so don't worry
    fig = plt.figure()
    plt.hist(vals, 50)
    plt.xlabel(name)
    plt.ylabel("Frequency")
    plt.savefig('histograms/' + name + '.png')
    plt.close()
