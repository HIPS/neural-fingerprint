'''
Kayak objects to implement operations on array-representation of molecules.
'''
import numpy as np
import kayak as ky

def get_feature_array(ky_mol_graph, ntype):
    return ky.Blank([ky_mol_graph], lambda p :
                    p[0].value.feature_array(ntype))

def get_neighbor_list(ky_mol_graph, ntypes):
    return ky.Blank([ky_mol_graph], lambda p :
                    p[0].value.neighbor_list(*ntypes))

class NeighborStack(ky.Differentiable):
    def __init__(self, idxs, features):
        super(NeighborStack, self).__init__((features, idxs))
        self.idxs = idxs
        self.features = features

    def _compute_value(self):
        # dims of result are (atoms, neighbors, features)
        idxs = self.idxs.value
        features = self.features.value
        result_rows = []
        for idx_list in idxs:
            result_rows.append([features[i, :] for i in idx_list])
        return np.array(result_rows)

    def _local_grad(self, parent, d_out_d_self):
        if parent is not 0:
            raise ValueError("Not a valid parent to be differentiating with respect to")

        idxs = self.idxs.value
        features = self.features.value
        result = np.zeros(features.shape)
        for i, idx_list in enumerate(idxs):
            for j, row in enumerate(idx_list):
                result[row, :] += d_out_d_self[i, j, :]

        return result

class NeighborCat(ky.Differentiable):
    def __init__(self, idxs, features):
        super(NeighborCat, self).__init__((features, idxs))
        self.idxs = idxs
        self.features = features

    def _compute_value(self):
        idxs = self.idxs.value
        features = self.features.value
        result_rows = []
        for idx_list in idxs:
            result_rows.append(np.concatenate(
                [features[i, :] for i in idx_list]))
        return np.array(result_rows)

    def _local_grad(self, parent, d_out_d_self):
        if parent is not 0:
            raise ValueError("Not a valid parent to be differentiating with respect to")

        idxs = self.idxs.value
        features = self.features.value
        result = np.zeros(features.shape)
        for i, idx_list in enumerate(idxs):
            uncat_input_rows = np.split(d_out_d_self[i, :], len(idx_list))
            for idx, row in zip(idx_list, uncat_input_rows):
                result[idx, :] += row

        return result

class NeighborSoftenedMax(ky.Differentiable):
    def __init__(self, idxs, features):
        super(NeighborSoftenedMax, self).__init__((features, idxs))
        self.features = features
        self.idxs = idxs

    def _compute_value(self):
        idxs = self.idxs.value
        features = self.features.value
        result_rows = []
        for idx_list in idxs:
            result_rows.append(self._softened_max_func(features[idx_list, :]))

        return np.array(result_rows)

    def _softened_max_func(self, X):
        exp_X = np.exp(X)
        return np.sum(exp_X * X, axis=0) / np.sum(exp_X, axis=0)
            
    def _softened_max_grad(self, X):
        exp_X = np.exp(X)
        sum_exp_X = np.sum(exp_X, axis=0, keepdims=True)
        sum_X_exp_X = np.sum(X * exp_X, axis=0, keepdims=True)
        return exp_X * ((X + 1) * sum_exp_X - sum_X_exp_X) / sum_exp_X**2

    def _local_grad(self, parent, d_out_d_self):
        if parent is not 0:
            raise ValueError("Not a valid parent to be differentiating with respect to")

        idxs = self.idxs.value
        features = self.features.value
        result = np.zeros(features.shape)
        for i, idx_list in enumerate(idxs):
            result[idx_list] += (self._softened_max_grad(features[idx_list, :]) *
                                 d_out_d_self[i, :][None, :])

        return result

# class NeighborMatMult(ky.Differentiable):
#     def __init__(self, idxs, features, weights):
#         super(NeighborMatMult, self).__init__(
#             (features, idxs) + tuple(weights.values()))

#         self.features = features
#         self.idxs = idxs
#         self.weights = weights

#     def _compute_value(self):
#         features = self.features.value
#         weights = {k : v.value for k, v in self.weights.iteritems()}
#         idxs = self.idxs.value
#         result_rows = []
#         for idx_list in idxs:
#             cat_row = np.concatenate([features[idx, :] for idx in idx_list], axis=1)
#             result_rows.append(np.dot(cat_row, weights[len(idx_list)]))

#         return np.array(result_rows)

#     def _local_grad(self, parent, d_out_d_self):
#         features = self.features.value
#         idxs = self.idxs.value
#         weights = {k : v.value for k, v in self.weights.iteritems()}
#         if parent is 0:
#             # deriv wrt features
#             result = np.zeros(features.shape)
#             for i, idx_list in enumerate(idxs):
#                 degree = len(idx_list)
#                 cat_input_rows = np.dot(d_out_d_self[i, :], weights[degree].T)
#                 uncat_input_rows = np.split(cat_input_rows, degree)
#                 for idx, row in zip(idx_list, uncat_input_rows):
#                     result[idx, :] += row

#             return result

#         elif parent in (2, 3, 4, 5):
#             # deriv wrt weights
#             D_in, D_out = weights[1].shape
#             degree = parent - 1 # degrees of 1 - 4 are allowed
#             contributors = [i for i, idx_list in enumerate(idxs) if len(idx_list) is degree]
#             N_contributors = len(contributors)
#             output_rows = d_out_d_self[contributors, :] # Only these rows contribute
#             input_cats = np.zeros((degree * D_in, N_contributors))
#             slices = [slice(D_in * i, D_in * (i + 1)) for i in range(degree)]
#             for s_idx, s in enumerate(slices):
#                 for i, contributor in enumerate(contributors):
#                     input_cats[s, i] = features[idxs[contributor][s_idx], :]

#             return np.dot(input_cats, output_rows)

#         else:
#             # Trying to differentiate wrt idxs
#             raise ValueError("Not a valid parent to be differentiating with respect to")

