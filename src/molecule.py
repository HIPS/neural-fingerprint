import sys
import time
import numpy as np

from rdkit  import Chem

import util
import features

from edge   import Edge
from vertex import Vertex
from graph  import Graph, GraphNet

class Molecule:

    def __init__(self, smiles, layers):
        self.graph  = Graph()
        self.layers = layers

        self.parse_smiles(smiles)

    def parse_smiles(self, smiles):

        # Use RDKit to parse the SMILES representation.
        rdmol = Chem.MolFromSmiles(smiles)

        # Replicate the graph that RDKit produces.
        # Go on and extract features using RDKit also.

        # Iterate over the atoms.
        rd_atoms = {}
        for atom in rdmol.GetAtoms():

            units = [ features.atom_features(atom) ] + [None]*len(self.layers)

            rd_atoms[atom.GetIdx()] = Vertex( units=units )
            self.graph.add_vert( rd_atoms[atom.GetIdx()] )


        # Iterate over the bonds.
        for bond in rdmol.GetBonds():
            atom1 = bond.GetBeginAtom()
            atom2 = bond.GetEndAtom()

            units = [ features.bond_features(bond) ] + [None]*len(self.layers)

            self.graph.add_edge( Edge(rd_atoms[atom1.GetIdx()],
                                      rd_atoms[atom2.GetIdx()],
                                      units=units) )

class MolNet:

    def __init__(self, layers, max_degree=4): # TODO nonlinearity

        first_layer    = [(features.atom_features(),features.bond_features())]
        self.graph_net = GraphNet(first_layer + [(sz,sz) for sz in layers], max_degree)

    def apply(self, mol):
        return self.graph_net.apply(mol.graph)

    def learn(self, mol, err, rate):
        return self.graph_net.learn(mol.graph, err, rate)

    def checkgrad(self, mol):
        return self.graph_net.checkgrad(mol.graph)

def main():
    layers = [2]

    filename = '~/Dropbox/Collaborations/MolecularML.shared/data/ML_exploit_1k/tddft_hyb_b3l_lifetime.csv'
    data = util.load_csv(filename)

    # Turn the data into our molecule objects.
    num_mols = 10
    mols     = [Molecule(m['smiles'], layers) for m in data[:num_mols]]
    targets  = np.array([float(m['rate']) for m in data[:num_mols]])

    # FIXME: Strip out targets that are exactly zero.
    #valid = targets > 0.0
    
    targets = np.log(targets)

    net = MolNet(layers)

    simple_mol = Molecule('CC', layers)

    net.checkgrad(simple_mol)

    #for ii, mol in enumerate(mols):
    #    output = net.apply(mol)
    #    target = targets[ii]
    #    grad   = output - target
        
        #net.learn(mol, grad, 0.01)


    return 0


if __name__ == '__main__':
    sys.exit(main())
