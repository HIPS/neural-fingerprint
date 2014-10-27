'''
Kayak objects to implement operations on array-representation of molecules.
'''
import numpy as np
import kayak

class NeighborMatMult(kayak.Differentiable):
    def __init__(self, features, idxs, weights):
        super(NeighborMatMult, self).__init__(
            (features, idxs) + tuple(weights.values()))

        self.features = features
        self.idxs = idxs
        self.weights = weights

    def _compute_value(self):
        features = self.features.value
        weights = {k : v.value for k, v in self.weights.iteritems()}
        idxs = self.idxs.value
        D_out = weights[1].shape[1]
        N_out = len(idxs)
        result = np.zeros((N_out, D_out))
        for i, idx_list in enumerate(idxs):
            cat_row = np.concatenate([features[idx, :] for idx in idx_list], axis=1)
            result[i, :] = np.dot(cat_row, weights[len(idx_list)])

        return result

    def _local_grad(self, parent, d_out_d_self):
        features = self.features.value
        idxs = self.idxs.value
        weights = {k : v.value for k, v in self.weights.iteritems()}
        if parent is 0:
            # deriv wrt features
            result = np.zeros(features.shape)
            for i, idx_list in enumerate(idxs):
                degree = len(idx_list)
                cat_input_rows = np.dot(d_out_d_self[i, :], weights[degree].T)
                uncat_input_rows = np.split(cat_input_rows, degree)
                for idx, row in zip(idx_list, uncat_input_rows):
                    result[idx, :] += row

            return result

        elif parent in (2, 3, 4, 5):
            # deriv wrt weights
            D_in, D_out = weights[1].shape
            degree = parent - 1 # degrees of 1 - 4 are allowed
            contributors = [i for i, idx_list in enumerate(idxs) if len(idx_list) is degree]
            N_contributors = len(contributors)
            output_rows = d_out_d_self[contributors, :] # Only these rows contribute
            input_cats = np.zeros((degree * D_in, N_contributors))
            slices = [slice(D_in * i, D_in * (i + 1)) for i in range(degree)]
            for i, contributor in enumerate(contributors):
                for s, idx in zip(slices, idxs[contributor]):
                    input_cats[s, i] = features[idx, :]

            return np.dot(input_cats, output_rows)

        else:
            # Trying to differentiate wrt idxs
            raise ValueError("Not a valid parent to be differentiating with respect to")
