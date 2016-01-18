# Dougal Maclaurin
# David Duvenaud
# 2015
#
# These datasets are taken from
# https://tripod.nih.gov/tox21/challenge/

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

def maximum_degree_in_molecule(mol):
    degrees = [atom.GetDegree() for atom in mol.GetAtoms()]
    return np.max(degrees)

rdkit_functions = {'Molecular Weight' : Descriptors.MolWt,
                   'Polar Surface Area' : rdMolDescriptors.CalcTPSA,
                   'Number of Rings' : rdMolDescriptors.CalcNumRings,
                   'Number of H-Bond Donors' : rdMolDescriptors.CalcNumHBD,
                   'Number of Rotatable Bonds' : rdMolDescriptors.CalcNumRotatableBonds,
                   'Minimum Degree' : minimum_degree_in_molecule,
                   'Maximum Degree' : maximum_degree_in_molecule}

filenames = ['nr-ahr.smiles', 'nr-ar.smiles', 'sr-mmp.smiles']
#filenames = ['tiny.smiles']

for infilename in filenames:
    print "Processing", infilename
    outfilename = infilename + '-processed.csv'
    removed_outfilename = infilename + '-excluded_molecules.csv'
    pandas_data = pd.io.parsers.read_csv(infilename, sep='\t')

    fields = ['smiles', 'inchi_key', 'target']

    print "Loading raw file..."
    data = {field: np.array(pandas_data[field]) for field in fields}
    data = randomize_order(data)

    print "Computing molecule graphs from SMILES..."
    mols = map(Chem.MolFromSmiles, data['smiles'])
    N_mols_original = len(mols)
    bad_mol_ixs = np.array([x is None or maximum_degree_in_molecule(x) >= 5 for x in mols])
    print "{0} molecules couldn't be parsed.".format(np.sum(bad_mol_ixs))

    def make_fun_run_only_on_good(fun):
        def new_fun(mol):
            if mol is not None:
                return fun(mol)
            else:
                return np.NaN
        return new_fun

    print "Computing RDKit features on each molecule..."
    for feature_name, fun in rdkit_functions.iteritems():
        print "Computing", feature_name, "..."
        data[feature_name] = np.array(map(make_fun_run_only_on_good(fun), mols))

    print "Identifying duplicates...",
    all_inchis = {}
    duplicated_mols = []
    is_dup = [False] * len(data.values()[0])
    for i, inchi in enumerate(data['inchi_key']):
        if inchi in all_inchis:
            is_dup[i] = True
        all_inchis[inchi] = True
    print np.sum(is_dup), "found."

    print "Identifying bad values...",
    is_finite = [True] * len(data.values()[0])
    for names, vals in data.iteritems():
        if names in ['inchi_key', 'smiles']:
            continue  # Not-numeric so don't worry
        is_finite = np.logical_and(is_finite, np.isfinite(vals))
    print np.sum(np.logical_not(is_finite)), " found."

    good_ixs = is_finite & np.logical_not(is_dup) & np.logical_not(bad_mol_ixs)
    good_data = slicedict(data, good_ixs)
    bad_data = slicedict(data, np.logical_not(good_ixs))
    print "Datapoints thrown away:", np.sum(np.logical_not(good_ixs))
    print "Datapoints kept:", np.sum(good_ixs)

    print "Writing output to file..."
    pd.DataFrame(good_data).to_csv(outfilename, sep=',', header=True, index=False)
    pd.DataFrame(bad_data).to_csv(removed_outfilename, sep=',', header=True, index=False)

    print "Making histograms of all features..."
    for name, vals in good_data.iteritems():
        if name in ['inchi_key', 'smiles']:
            continue  # Not-numeric so don't worry
        fig = plt.figure()
        plt.hist(vals, 20)
        plt.xlabel(name)
        plt.ylabel("Frequency")
        plt.savefig('histograms/' + name + '.png')
        plt.close()
