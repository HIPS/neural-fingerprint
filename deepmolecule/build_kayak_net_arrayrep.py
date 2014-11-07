import itertools as it
import kayak as ky
import kayak_ops as mk
from features import N_atom_features, N_bond_features
from util import WeightsContainer, c_value, c_grad

def all_permutations(N):
    return [permutation for permutation in it.permutations(range(N))]

def softened_max(X_list):
    X_cat = ky.ListToArray(*X_list)
    X_feature_0 = ky.Take(X_cat, slice(0,1), axis=2)
    return ky.MatSum(ky.MatElemMult(X_cat, ky.SoftMax(X_feature_0, axis=0)),
                     axis=0, keepdims=False)

def matmult_neighbors(mol_graph, self_ntype, other_ntypes, feature_sets,
                      weights_gen, permutations=False):
    def neighbor_list(degree, other_ntype):
        return mol_graph.get_neighbor_list((self_ntype, degree), other_ntype)
    result_by_degree = []
    for degree in [1, 2, 3, 4]:
        # dims of stacked_neighbors are [atoms, neighbors (as in atom-bond pairs), features]
        stacked_neighbors = ky.Concatenate(2,
            *[mk.NeighborStack(neighbor_list(degree, other_ntype), features, degree)
              for other_ntype, features in zip(other_ntypes, feature_sets)])
        if permutations:
            weightses = [weights_gen() for i in range(degree)]
            neighbors = [ky.Take(stacked_neighbors, i, axis=1) for i in range(degree)]
            products = [[ky.MatMult(n, w) for w in weightses] for n in neighbors]
            candidates = [ky.MatAdd(*[products[i][j] for i, j in enumerate(p)])
                          for p in all_permutations(degree)]
            # dims of candidates are (atoms, features)
            result_by_degree.append(softened_max(candidates))
        else:
            result_by_degree.append(ky.MatSum(
                ky.TensorMult(stacked_neighbors, weights_gen(), axes=((2,), (0,))),
                axis=1, keepdims=False))

    # This is brittle! Relies on atoms being sorted by degree in the first place,
    # in Node.graph_from_smiles_tuple()
    return ky.Concatenate(0, *result_by_degree)

def build_universal_net(num_hidden_features=[50, 50], permutations=False):
    """The number of hidden layers is the length of num_hidden_features."""
    layer_sizes = [N_atom_features] + num_hidden_features
    weights = WeightsContainer()
    smiles_input = ky.Blank()
    mol_graph = mk.MolGraphNode(smiles_input) 
    cur_atoms = mol_graph.get_feature_array('atom')
    cur_bonds = mol_graph.get_feature_array('bond')
    for N_prev, N_cur in zip(layer_sizes[:-1], layer_sizes[1:]):
        cur_atoms = ky.Logistic(ky.MatAdd(
            ky.MatMult(cur_atoms, weights.new((N_prev, N_cur))),
            matmult_neighbors(mol_graph, 'atom', ('atom', 'bond'), (cur_atoms, cur_bonds),
                              lambda : weights.new((N_prev + N_bond_features, N_cur)),
                              permutations)))

    mol_atom_neighbors = mol_graph.get_neighbor_list('molecule', 'atom')
    fixed_sized_output = mk.NeighborSoftenedMax(mol_atom_neighbors, cur_atoms)
    output = ky.MatMult(fixed_sized_output, weights.new((N_cur, )))
    target = ky.Blank()
    loss = ky.L2Loss(output, target)

    def grad_fun(w, s, t):
        return c_grad(loss, weights, {weights : w, smiles_input : s, target : t})
    def loss_fun(w, s, t):
        return c_value(loss, {weights : w, smiles_input : s, target : t})
    def pred_fun(w, s):
        return c_value(output, {weights : w, smiles_input : s})
    def output_layer_fun(w, s):
        return c_value(fixed_sized_output, {weights : w, smiles_input : s})

    return loss_fun, grad_fun, pred_fun, output_layer_fun, weights.N
