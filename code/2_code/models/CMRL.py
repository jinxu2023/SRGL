import random
import numpy as np
import tensorflow as tf
from tensorflow.python.keras.layers import Dense
from tensorflow.python.keras.models import Model
from tensorflow.python.keras.initializers.initializers_v2 import GlorotUniform
from tensorflow.python.keras.callbacks import EarlyStopping

from common_functions.GNN_layers import GCN_Layer
from common_functions.metrics import MetricUtils
from common_functions.configs import Configs


class CMRL_Model:

    def __init__(self, input_data=None, gnn_levels=2, mlp_levels=1, units=200,
                 epochs=200, optimizer='adam', loss_mode='RDT',
                 seed=0, diffuse_paras=None, subgraph_paras=None, hyper_paras=(1,), **kwargs):
        if diffuse_paras is None:
            self.diffuse_paras = {
                'diffuse_method': 'PPR',
                't': 300,
                'alpha': 0.15
            }
        if subgraph_paras is None:
            self.subgraph_paras = {
                'sample_subgraph': False,
                'subgraph_sampler': 'RW',  # RW, NN
                'subgraph_num': 100,
                'path_len': 100,  # RW
                'path_num': 200,  # RW
                'min_node_num': 50,  # RW and NN
                'neigh_level': 3,  # NN
                'max_node_num': 1000  # NN
            }

        self.gnn_levels = gnn_levels
        self.mlp_levels = mlp_levels
        if kwargs.__contains__('levels'):
            self.gnn_levels = kwargs['levels']
        self.units = units
        self.epochs = epochs
        self._optimizer = optimizer
        self.seed = seed
        self.hyper_paras = hyper_paras
        self.loss_mode = loss_mode
        self.kwargs = kwargs

        self.init_data(input_data)
        self.fill_mat()
        self.diffuse()
        self.cor_lists()

    def init_data(self, input_data):
        [D, T, R_train, self.R_truth, H_d, H_t, self.mask] = input_data[:7]
        self.R = R_train
        self.A = np.r_[np.c_[D, R_train], np.c_[R_train.T, T]]
        self.H_raw = np.r_[H_d, H_t]
        self.H_dif = np.r_[H_d, H_t]
        self.d_num = D.shape[0]
        self.t_num = T.shape[0]
        self.R_pred = None
        self.H_d_out = None
        self.H_t_out = None

    def fill_mat(self):
        self.A_fill = self.A + 0
        all_zero_rows = np.flatnonzero(np.sum(self.A, axis=0) == 0)
        self.A_fill[all_zero_rows, :] = 1 / (self.d_num + self.t_num)
        self.A_fill[:, all_zero_rows] = 1 / (self.d_num + self.t_num)

    def diffuse(self):
        if self.diffuse_paras['diffuse_method'] == 'heat':
            t = self.diffuse_paras['t']
            D = np.sum(self.A_fill, axis=1, keepdims=True)
            D_inv = 1 / D
            D_inv[np.isnan(D_inv)] = 0
            D_inv[np.isinf(D_inv)] = 0
            self.A_dif = np.exp(t * self.A_fill * D_inv - t)
        elif self.diffuse_paras['diffuse_method'] == 'PPR':
            alpha = self.diffuse_paras['alpha']
            D = np.sum(self.A_fill, axis=1, keepdims=True)
            D_inv = 1 / D
            D_inv[np.isinf(D_inv)] = 0
            D_inv[np.isnan(D_inv)] = 0
            D_inv_sqrt = np.sqrt(D_inv)
            A_dif = alpha * tf.linalg.inv(np.eye(self.A_fill.shape[0], dtype='float32') -
                                          (
                                                  1 - alpha) * D_inv_sqrt * self.A_fill * D_inv_sqrt.T).numpy()
            A_dif[np.isnan(A_dif)] = 0
            A_dif[np.isinf(A_dif)] = 0
            self.A_dif = A_dif * (1 - np.eye(self.A_fill.shape[0], dtype='float32'))

    def cor_lists(self):
        self.A_cor = [None] * (self.d_num + self.t_num)

        for i in range(self.d_num + self.t_num):
            self.A_cor[i] = np.flatnonzero(self.A[i, :])

    def subgraph_sample(self):
        if self.subgraph_paras['sample_subgraph'] is False:
            return [np.arange(self.d_num + self.t_num).tolist(),
                    np.arange(self.d_num).tolist(),
                    np.arange(self.t_num).tolist()]

        elif self.subgraph_paras['subgraph_sampler'] == 'RW':
            path_len = self.subgraph_paras['path_len']
            path_num = self.subgraph_paras['path_num']
            min_node_num = self.subgraph_paras['min_node_num']
            sampled_idxs = set()
            sampled_d_idxs = set()
            sampled_t_idxs = set()

            while 1:
                start_node = random.randint(0, self.A.shape[0] - 1)

                if len(self.A_cor[start_node]) == 0:
                    continue
                else:
                    sampled_idxs.add(start_node)
                    if start_node < self.d_num:
                        sampled_d_idxs.add(start_node)
                    else:
                        sampled_t_idxs.add(start_node - self.d_num)
                for i in range(path_num):
                    current_node = start_node
                    for j in range(path_len):
                        permed_nodes = np.random.permutation(self.A_cor[current_node])
                        current_node = permed_nodes[0]
                        sampled_idxs.add(current_node)
                        if current_node < self.d_num:
                            sampled_d_idxs.add(current_node)
                        else:
                            sampled_t_idxs.add(current_node - self.d_num)
                if len(sampled_d_idxs) > min_node_num and len(sampled_t_idxs) > min_node_num:
                    break

            return list(sampled_idxs), list(sampled_d_idxs), list(sampled_t_idxs)

        elif self.subgraph_paras['subgraph_sampler'] == 'NN':
            while 1:
                min_node_num = self.subgraph_paras['min_node_num']

                neigh_level = self.subgraph_paras['neigh_level']
                A_power = np.linalg.matrix_power(
                    self.A + np.eye(self.A.shape[0], dtype='float32'), neigh_level)
                A_power[A_power > 0] = 1

                center_node = random.randint(0, self.A.shape[0] - 1)
                sampled_idxs = np.flatnonzero(A_power[center_node, :])
                sampled_d_idxs = sampled_idxs[np.where(sampled_idxs < self.d_num)]
                sampled_t_idxs = sampled_idxs[np.where(sampled_idxs >= self.d_num)] - self.d_num
                if len(sampled_idxs) > 0:
                    break

            return list(sampled_idxs), list(sampled_d_idxs), list(sampled_t_idxs)

    def get_config(self):
        config = {
            'gnn_levels': self.gnn_levels,
            'mlp_levels': self.mlp_levels,
            'units': self.units,
            'epochs': self.epochs,
            'optimizer': self._optimizer,
            'loss_mode': self.loss_mode,
            'diffuse_paras': self.diffuse_paras,
            'subgraph_paras': self.subgraph_paras,
            'seed': self.seed
        }
        return dict(config, **self.kwargs)

    def fit(self, epochs):
        if not self.subgraph_paras['sample_subgraph']:
            n_subgraph = 1
        else:
            n_subgraph = self.subgraph_paras['subgraph_num']

        for i in range(n_subgraph):
            tf.compat.v1.set_random_seed(self.seed)
            np.random.seed(self.seed)
            sampled_idxs, sampled_d_idxs, sampled_t_idxs = self.subgraph_sample()

            _A = self.A[sampled_idxs, :][:, sampled_idxs]
            _A_dif = self.A_dif[sampled_idxs, :][:, sampled_idxs]
            _R = self.R[sampled_d_idxs, :][:, sampled_t_idxs]
            _R_truth = self.R_truth[sampled_d_idxs, :][:, sampled_t_idxs]
            _mask = self.mask[sampled_d_idxs, :][:, sampled_t_idxs]
            _H_raw = self.H_raw[sampled_idxs, :]
            _H_dif = self.H_dif[sampled_idxs, :]

            model = _CMRL_Model(_A, _A_dif, _R, _R_truth, _mask, _H_raw, _H_dif,
                                gnn_levels=self.gnn_levels, mlp_levels=self.mlp_levels,
                                units=self.units,
                                epochs=self.epochs, optimizer=self._optimizer,
                                hyper_paras=self.hyper_paras, seed=self.seed,
                                loss_mode=self.loss_mode, kwargs=self.kwargs)
            model.fit()
            [_H_raw_out, _H_dif_out] = model.predict()

            self.H_raw = _H_raw_out
            self.H_dif = _H_dif_out

        self.H_d_pred = (self.H_raw[:self.d_num, :] + self.H_dif[:self.d_num, :]) / 2
        self.H_t_pred = (self.H_raw[self.d_num:, :] + self.H_dif[self.d_num:, :]) / 2
        self.R_pred = self.H_d_pred @ self.H_t_pred.T

    def predict(self):
        return [self.R_pred, self.H_d_pred, self.H_t_pred, None]

    def calc_loss(self):
        pass


class _CMRL_Model(Model):

    def __init__(self, A, A_dif, R, R_truth, mask, H_raw, H_dif,
                 gnn_levels=2, mlp_levels=1, units=200,
                 epochs=100, optimizer='adam', loss_mode='RDT',
                 hyper_paras=(), seed=0, kwargs=None):
        super(_CMRL_Model, self).__init__()

        self.A = tf.convert_to_tensor(A, 'float32')
        self.A_dif = tf.convert_to_tensor(A_dif, 'float32')
        self.R = tf.convert_to_tensor(R, 'float32')
        self.R_truth = tf.convert_to_tensor(R_truth, 'float32')
        self.mask = tf.convert_to_tensor(mask, 'float32')
        self.d_num, self.t_num = R.shape
        self.gnn_levels = gnn_levels
        self.mlp_levels = mlp_levels
        self.units = units
        self.epochs = epochs
        self.hyper_paras = hyper_paras
        self.seed = seed
        self.loss_mode = loss_mode
        self.kwargs = kwargs
        self.H_raw = H_raw
        self.H_dif = H_dif

        self.GNNs_raw = [GCN_Layer(units=units,
                                   activation=tf.nn.relu,
                                   seed=seed)
                         for i in range(gnn_levels)]
        self.GNNs_dif = [GCN_Layer(units=units,
                                   activation=tf.nn.relu,
                                   seed=seed)
                         for i in range(gnn_levels)]

        self.MLPs_gnn = [Dense(units=units,
                               use_bias=True,
                               activation=tf.nn.relu,
                               kernel_initializer=GlorotUniform(seed))
                         for i in range(mlp_levels)]
        self.MLPs_pool = [Dense(units=units,
                                use_bias=True,
                                activation=tf.nn.relu,
                                kernel_initializer=GlorotUniform(seed))
                          for i in range(mlp_levels)]

        self.W_pool = self.add_weight(name='W_pool',
                                      shape=(units * gnn_levels, units),
                                      initializer=GlorotUniform(self.seed))
        if Configs.metric_config['metric_group_idx'] == 0:
            self.es_callback = EarlyStopping(monitor=Configs.metric_config['cv_metric'],
                                             patience=Configs.metric_config['patience'],
                                             mode='max', restore_best_weights=True)
        else:
            self.es_callback = EarlyStopping(monitor='loss',
                                             patience=Configs.metric_config['patience'],
                                             mode='min', restore_best_weights=True)
        self.compile(optimizer=optimizer)

    def call(self, inputs, training=None, mask=None):
        H_raw, H_dif = inputs[0][0], inputs[1][0]
        Hs_raw = [self.GNNs_raw[0]([self.A, H_raw])] + [None] * (self.gnn_levels - 1)
        Hs_dif = [self.GNNs_dif[0]([self.A_dif, H_dif])] + [None] * (self.gnn_levels - 1)
        for i in range(1, self.gnn_levels):
            Hs_raw[i] = self.GNNs_raw[i]([self.A, Hs_raw[i - 1]])
            Hs_dif[i] = self.GNNs_dif[i]([self.A_dif, Hs_dif[i - 1]])

        hs_raw = [tf.reduce_mean(H_raw, axis=0) for H_raw in Hs_raw]
        hs_dif = [tf.reduce_mean(H_dif, axis=0) for H_dif in Hs_dif]

        h_raw = tf.nn.sigmoid(tf.einsum('i,ij->j', tf.concat(hs_raw, axis=0), self.W_pool))
        h_dif = tf.nn.sigmoid(tf.einsum('i,ij->j', tf.concat(hs_dif, axis=0), self.W_pool))
        h_raw = tf.expand_dims(h_raw, axis=0)
        h_dif = tf.expand_dims(h_dif, axis=0)

        for i in range(self.mlp_levels):
            h_raw = self.MLPs_pool[i](h_raw)
            h_dif = self.MLPs_pool[i](h_dif)

        H_raw = Hs_raw[-1]
        H_dif = Hs_dif[-1]

        for i in range(self.mlp_levels):
            H_raw = self.MLPs_gnn[i](H_raw)
            H_dif = self.MLPs_gnn[i](H_dif)
        H = (H_raw + H_dif) / 2

        if self.loss_mode == 'RDT' and self.kwargs['use_D'] and self.kwargs['use_T']:
            A_pred = tf.einsum('ij,kj->ik', H, H)
            loss = self.calc_loss(H_raw, H_dif, h_raw, h_dif, A_pred, self.A)
        else:
            R_pred = tf.einsum('ij,kj->ik', H[:self.d_num, :], H[self.d_num:, :])
            loss = self.calc_loss(H_raw, H_dif, h_raw, h_dif, R_pred, self.R)
        self.add_loss(loss)

        return [tf.expand_dims(H_raw, axis=0),
                tf.expand_dims(H_dif, axis=0)]

    def calc_loss(self, H_raw, H_dif, h_raw, h_dif, X_pred, X):
        alpha = self.hyper_paras[0]

        loss_MI = -tf.einsum('ij,kj->ik', H_raw, h_dif) \
                  - tf.einsum('ij,kj->ik', H_dif, h_raw)
        loss_MI = tf.reduce_mean(loss_MI)

        if Configs.dataset not in {'Davis', 'KIBA'}:
            loss_LP_pos = tf.reduce_mean(tf.square(
                tf.cast(X == 1, 'float32') * (X_pred - X)))
            loss_LP_neg = tf.reduce_mean(tf.square(
                tf.cast(X == 0, 'float32') * (X_pred - X)))
            loss_LP = loss_LP_pos + 0.2 * loss_LP_neg
        else:
            X_ind = (X > 0).astype('float32')
            loss_LP = tf.reduce_sum((X_ind * (X_pred - X)) ** 2) / tf.reduce_sum(X_ind)

        return alpha * loss_MI + loss_LP

    def fit(self):
        H_raw = tf.expand_dims(self.H_raw, axis=0)
        H_dif = tf.expand_dims(self.H_dif, axis=0)
        x = [H_raw, H_dif]
        Model.fit(self, x=x, batch_size=1, epochs=self.epochs, verbose=1,
                  callbacks=[self.es_callback])
        [H_raw_out, H_dif_out] = Model.predict(self, x=x, batch_size=1, verbose=2)
        self.H_raw_out = np.squeeze(H_raw_out)
        self.H_dif_out = np.squeeze(H_dif_out)

    def predict(self):
        return [self.H_raw_out, self.H_dif_out]
