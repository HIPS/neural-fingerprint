import pandas as pd
import rdkit.Chem as Chem
from rdkit.Chem import AllChem
import numpy as np

def load_molecules(filename, test=False, predict_field='rate', transform=None, cutoff=None):

    data = pd.io.parsers.read_csv(filename, sep=',')  # Load the data.
    fingerprints = [smile_to_fp(s) for s in list(data['smiles'])]

    alldata = {}
    alldata['fingerprints'] = stringlist2intarray(fingerprints)  # N by D
    alldata['inchi_key'] = data['inchi_key']
    alldata['smiles'] = np.array(data['smiles'])

    if predict_field in data:
        alldata['y'] = np.array(transform(data[predict_field])) # Load targets

        valid = np.isfinite(alldata['y'])
        if cutoff:
            above_cutoff = (alldata['y'] > cutoff) & valid
        valid_ixs = np.where(valid)[0]

        alldata = slicedict(alldata, valid_ixs)
        print "Datapoints kept:", len(valid_ixs), "of", len(alldata['y'])

    return alldata

def stringlist2intarray(A):
    '''This function will convert from a list of strings "10010101" into in integer numpy array '''
    return np.array([list(s) for s in A], dtype=int)

def compute_features(mol, size=512, radius=2):
    return AllChem.GetMorganFingerprintAsBitVect(mol, radius, nBits=size)

def slicedict(d, ixs):
    return {k : v[ixs] for k, v in d.iteritems()}

def smile_to_fp(s):
    fplength = 512
    radius = 4
    m = Chem.MolFromSmiles(s)
    return (AllChem.GetMorganFingerprintAsBitVect(m, radius,
                                                  nBits=fplength)).ToBitString()
