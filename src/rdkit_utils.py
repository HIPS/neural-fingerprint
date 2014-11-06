import rdkit.Chem as Chem
from rdkit.Chem import AllChem
import numpy as np

def smiles_to_fps(s, fp_length, fp_radius):
    fingerprints = np.array([smile_to_fp(s)
                             for s in list(data['smiles'])])

def smile_to_fp(s, fp_length, fp_radius):
    m = Chem.MolFromSmiles(s)
    return (AllChem.GetMorganFingerprintAsBitVect(
        m, radius, nBits=fplength)).ToBitString()
