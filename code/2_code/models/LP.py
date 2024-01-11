import numpy as np
import scipy.spatial.distance as dist


class LP_Model:
    def __init__(self, input_data=None, hyper_paras=(0.9,), **kwargs):
        self.hyper_paras = hyper_paras
        self.kwargs = kwargs

        self.init_data(input_data)

    def init_data(self, input_data):
        self.R = input_data[2]
        S_d = dist.squareform(dist.pdist(self.R, 'cos'))
        S_d[np.isnan(S_d)] = 0
        row_sum = np.sum(S_d, axis=1, keepdims=True)
        row_sum = row_sum + (row_sum == 0).astype('float32')
        row_norm = row_sum ** -0.5
        self.S_d_norm = row_norm * S_d * row_norm.T

    def get_config(self):
        return self.kwargs

    def fit(self, **kwargs):
        alpha = self.hyper_paras[0]
        self.R_pred = (1 - alpha) * np.linalg.inv(
            np.eye(self.S_d_norm.shape[0]) - alpha * self.S_d_norm) @ self.R

    def predict(self, **kwargs):
        return [self.R_pred, None, None,None]
