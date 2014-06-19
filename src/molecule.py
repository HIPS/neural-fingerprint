import sys

from rdkit  import Chem

from edge   import Edge
from vertex import Vertex
from graph  import Graph

class Molecule:

    def __init__(self, smiles, layers):
        self.graph = Graph()

        self._parse_smiles(smiles)

    def _parse_smiles(self, smiles):

        # Use RDKit to parse the SMILES representation.
        rdmol = Chem.MolFromSmiles(smiles)

        # Replicate the graph that RDKit produces.

        # Iterate over the atoms.
        rd_atoms = {}
        for atom in rdmol.GetAtoms():
            rd_atoms[atom] = Vertex( rd=atom )

            

            # Create a vertex object that owns this atom.
            #self.graph.add_vertex(Vertex( rd=atom ))

        # Iterate over the bonds.


def main():
    smiles = 'Cc1cc(N2c3ccccc3N(C)c3ccccc32)cc(C)c1C1=CC(=O)C=CC1=O'

    mol = Molecule(smiles, [5, 3, 2])

    return 0


if __name__ == '__main__':
    sys.exit(main())
