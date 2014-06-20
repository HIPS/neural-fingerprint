import numpy as np

from rdkit import Chem


atom_types = ['C', 'N', 'O', 'S', 'F']
def atom_features(atom):    
    if atom is None:
        # Return length of feature vector using a very simple molecule.
        return len(atom_features(Chem.MolFromSmiles('CC').GetAtoms()[0]))

    else:
        if atom.GetSymbol() not in atom_types:
            raise Exception("Atom type %r not modeled" % (atom.GetSymbol()))

        ht = atom.GetHybridization()
        return np.concatenate([np.array(map(lambda s: atom.GetSymbol() == s, atom_types)),
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
                                ht == Chem.rdchem.HybridizationType.SP3D2 ]])

def bond_features(bond):
    if bond is None:
        # Return length of feature vector using a very simple molecule.
        simple_mol = Chem.MolFromSmiles('CC')
        Chem.SanitizeMol(simple_mol)
        return len(bond_features(simple_mol.GetBonds()[0]))
    else:
        bt = bond.GetBondType()
        return np.array([ bt == Chem.rdchem.BondType.SINGLE,
                          bt == Chem.rdchem.BondType.DOUBLE,
                          bt == Chem.rdchem.BondType.TRIPLE,
                          bt == Chem.rdchem.BondType.AROMATIC,
                          bond.GetIsConjugated(),
                          bond.IsInRing()])
