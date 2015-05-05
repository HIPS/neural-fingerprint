import autograd.numpy as np
from rdkit import Chem
import time
time.sleep(0.2)   # To deal with a race condition bug in rdkit.


atom_types = ['C', 'N', 'O', 'S', 'F', 'Si', 'P', 'Cl']

def atom_features(atom, extra_features):
    if atom.GetSymbol() not in atom_types:
        raise Exception("Atom type %r not modeled" % (atom.GetSymbol()))

    if extra_features:
        ht = atom.GetHybridization()
        return np.concatenate(
            [np.array(map(lambda s: atom.GetSymbol() == s, atom_types)),  # One-of-k encoding.
            [atom.GetAtomicNum(),
            atom.GetMass(),
            atom.GetExplicitValence(),
            atom.GetImplicitValence(),
            atom.GetFormalCharge(),
            atom.GetIsAromatic(),
            ht == Chem.rdchem.HybridizationType.SP,
            ht == Chem.rdchem.HybridizationType.SP2,
            ht == Chem.rdchem.HybridizationType.SP3,
            ht == Chem.rdchem.HybridizationType.SP3D,
            ht == Chem.rdchem.HybridizationType.SP3D2]])
    else:
        return np.array(map(lambda s: atom.GetSymbol() == s, atom_types)),  # One-of-k encoding.

def bond_features(bond):
    bt = bond.GetBondType()
    return np.array(
        [bt == Chem.rdchem.BondType.SINGLE,
         bt == Chem.rdchem.BondType.DOUBLE,
         bt == Chem.rdchem.BondType.TRIPLE,
         bt == Chem.rdchem.BondType.AROMATIC,
         bond.GetIsConjugated(),
         bond.IsInRing()
         ])

def num_atom_features(extra_features):
    # Return length of feature vector using a very simple molecule.
    return len(atom_features(Chem.MolFromSmiles('C').GetAtoms()[0], extra_features))

def num_bond_features():
    # Return length of feature vector using a very simple molecule.
    simple_mol = Chem.MolFromSmiles('CC')
    Chem.SanitizeMol(simple_mol)
    return len(bond_features(simple_mol.GetBonds()[0]))

