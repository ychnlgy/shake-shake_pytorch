import torch
import imblearn.over_sampling

def smote(X, Y, k_neighbors):

    '''

    Input:
        X - torch Tensor of size (N, D).
        Y - torch Tensor of size (N).

    Output:
        X' - torch Tensor of size (N', D), where N' >= N in
            the case of class misbalance.
        Y' - torch Tensor of size (N').

    '''
    X, Y = smote_array(X.numpy(), Y.numpy(), k_neighbors)
    return torch.from_numpy(X), torch.from_numpy(Y)

def smote_array(X, Y, k_neighbors):
    smote_obj = imblearn.over_sampling.SMOTE(k_neighbors=k_neighbors)
    return smote_obj.fit_sample(X, Y)
