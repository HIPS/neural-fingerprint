import numpy as np
from rdkit import Chem

atom_types = ['C', 'N', 'O', 'S', 'F', 'Si', 'P', 'Cl']

def atom_features(atom):
    if atom.GetSymbol() not in atom_types:
        raise Exception("Atom type %r not modeled" % (atom.GetSymbol()))

    ht = atom.GetHybridization()
    return np.concatenate(
        [np.array(map(lambda s: atom.GetSymbol() == s, atom_types)),
         [1.0,
          atom.GetAtomicNum(),
          atom.GetMass(),
          atom.GetExplicitValence(),
          atom.GetImplicitValence(),
          atom.GetFormalCharge(),
          atom.GetIsAromatic(),
          ht == Chem.rdchem.HybridizationType.SP,
          ht == Chem.rdchem.HybridizationType.SP2,
          ht == Chem.rdchem.HybridizationType.SP3,
          ht == Chem.rdchem.HybridizationType.SP3D,
          ht == Chem.rdchem.HybridizationType.SP3D2 ]])

def bond_features(bond):
    bt = bond.GetBondType()
    return np.array(
        [1.0,
         bt == Chem.rdchem.BondType.SINGLE,
         bt == Chem.rdchem.BondType.DOUBLE,
         bt == Chem.rdchem.BondType.TRIPLE,
         bt == Chem.rdchem.BondType.AROMATIC,
         bond.GetIsConjugated(),
         bond.IsInRing()])

def get_num_atom_features():
    # Return length of feature vector using a very simple molecule.
    return len(atom_features(Chem.MolFromSmiles('CCCCCCCCCCC').GetAtoms()[0]))

def get_num_bond_features():
    # Return length of feature vector using a very simple molecule.
    simple_mol = Chem.MolFromSmiles('CCCCCCCCCC')
    Chem.SanitizeMol(simple_mol)
    return len(bond_features(simple_mol.GetBonds()[0]))

N_atom_features = get_num_atom_features()
N_bond_features = get_num_bond_features()
