import sys
import time
import numpy as np

from rdkit  import Chem

import features

from edge   import Edge
from vertex import Vertex
from graph  import Graph

class Molecule:

    def __init__(self, smiles, layers):
        self.graph = Graph()

        self.parse_smiles(smiles)

        print features.bond_features(None)

    def parse_smiles(self, smiles):

        # Use RDKit to parse the SMILES representation.
        rdmol = Chem.MolFromSmiles(smiles)

        # Replicate the graph that RDKit produces.

        # Iterate over the atoms.
        rd_atoms = {}
        for atom in rdmol.GetAtoms():
            rd_atoms[atom.GetIdx()] = Vertex( features=features.atom_features(atom) )
            self.graph.add_vertex( rd_atoms[atom.GetIdx()] )


        # Iterate over the bonds.
        for bond in rdmol.GetBonds():
            atom1 = bond.GetBeginAtom()
            atom2 = bond.GetEndAtom()

            self.graph.add_edge( rd_atoms[atom1.GetIdx()],
                                 rd_atoms[atom2.GetIdx()],
                                 Edge(features=features.bond_features(bond)) )




def main():
    #smiles = 'Cc1cc(N2c3ccccc3N(C)c3ccccc32)cc(C)c1C1=CC(=O)C=CC1=O'
    #smiles = 'COc1ccc(N(C)c2c(C)c(C)c(-c3cnc4nccnc4n3)c(C)c2C)cc1'
    smiles = 'C'

    mol = Molecule(smiles, [5, 3, 2])

    return 0


if __name__ == '__main__':
    sys.exit(main())
