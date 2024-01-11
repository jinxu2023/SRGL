from common_functions.GNN_layers import GCN_Layer
import numpy as np
import tensorflow as tf
from tensorflow.python.keras.initializers.initializers_v2 import GlorotUniform, Zeros
from tensorflow.python.keras.layers import Layer
from tensorflow.python.keras.models import Model
from tensorflow.python.keras.callbacks import EarlyStopping

from common_functions.metrics import MetricUtils
from common_functions.configs import Configs


class EGLN_Model(Model):

    def __init__(self, input_data=None,
                 levels=2, GL_units=200, GC_units=200, top_k=10, neg_sampler='FE',
                 epochs=200, optimizer='adam',
                 seed=0, hyper_paras=(1, 1), **kwargs):
        super(EGLN_Model, self).__init__()
        self.levels = levels
        self.GL_units = GL_units
        self.GC_units = GC_units
        if kwargs.__contains__('units'):
            self.GL_units = kwargs['units']
            self.GC_units = kwargs['units']
        # self.GL_units = self.GC_units = 32
        self.top_k = top_k
        self.neg_sampler = neg_sampler
        self.epochs = epochs
        self._optimizer = optimizer
        self.seed = seed
        self.hyper_paras = hyper_paras
        self.kwargs = kwargs

        self.init_data(input_data)
        # self.get_BPR_pos_neg_idxs()

        self.GL_layers = [GL_Layer(units=self.GL_units,
                                   top_k=self.top_k,
                                   seed=self.seed)
                          for i in range(self.levels)]
        self.GC_layers = [GC_Layer(units=self.GC_units,
                                   seed=self.seed)
                          for i in range(self.levels)]
        self.GCN_layers = [GCN_Layer(units=self.GC_units,
                                     seed=self.seed)
                           for i in range(self.levels)]
        if Configs.metric_config['metric_group_idx'] == 0:
            self.es_callback = EarlyStopping(monitor=Configs.metric_config['cv_metric'],
                                             patience=Configs.metric_config['patience'],
                                             mode='max', restore_best_weights=True)
        else:
            self.es_callback = EarlyStopping(monitor='loss',
                                             patience=Configs.metric_config['patience'],
                                             mode='min', restore_best_weights=True)
        self.compile(optimizer=self._optimizer)

    def init_data(self, input_data):
        [D, T, R_train, R_truth, H_d, H_t, mask] = input_data[:7]
        self.R_truth = tf.convert_to_tensor(R_truth, dtype='float32')
        self.mask = tf.convert_to_tensor(mask, dtype='float32')
        [self.d_num, self.t_num] = R_train.shape
        self.R = tf.convert_to_tensor(R_train, dtype='float32')
        A = np.r_[np.c_[np.zeros((self.d_num, self.d_num)), R_train],
        np.c_[R_train.T, np.zeros((self.t_num, self.t_num))]]
        self.A = tf.convert_to_tensor(A, dtype='float32')
        A = A + np.eye(A.shape[0], dtype='float32')
        row_sum = np.sum(A, axis=1, keepdims=True)
        row_norm = row_sum ** -0.5
        row_norm[np.isnan(row_norm)] = 0
        row_norm[np.isinf(row_norm)] = 0
        col_norm = row_norm.T
        self.A_norm = tf.convert_to_tensor(row_norm * A * col_norm, dtype='float32')

        self.H_d = H_d
        self.H_t = H_t
        # self.H_d = tf.random.normal(shape=(self.d_num, 32), mean=0, stddev=0.01)
        # self.H_t = tf.random.normal(shape=(self.t_num, 32), mean=0, stddev=0.01)
        self.R_pred = None
        self.H_d_out = None
        self.H_t_out = None

        self.W_d = self.add_weight(name='W_d',
                                   shape=(2 * self.GC_units, 2 * self.GC_units),
                                   initializer=GlorotUniform(self.seed))

    def get_BPR_pos_neg_idxs(self):
        self.BPR_pos_idxs = tf.where(self.R == 1)

        R = self.R.numpy()
        BPR_neg_idxs = np.zeros((0, 2))
        for i in range(self.d_num):
            row = R[i, :]
            zero_idxs = np.where(row == 0)[0]
            np.random.shuffle(zero_idxs)
            sampled_neg_num = np.sum(row == 1)
            neg_idxs_c = zero_idxs[:sampled_neg_num]
            neg_idxs = np.c_[np.ones((sampled_neg_num, 1)) * i, neg_idxs_c.reshape((-1, 1))]
            BPR_neg_idxs = np.r_[BPR_neg_idxs, neg_idxs]
        self.BPR_neg_idxs = tf.convert_to_tensor(BPR_neg_idxs, 'int32')

    def call(self, inputs, training=None, mask=None):
        H_d, H_t = inputs[0][0], inputs[1][0]
        A_E = self.A

        for i in range(self.levels):
            [A_R, A_R_filtered] = self.GL_layers[i]([H_d, H_t])
            A_E = A_E + A_R_filtered
            # A_E = A_R + 0
            # [H_d, H_t] = self.GC_layers[i]([A_E, H_d, H_t])
            H = self.GCN_layers[i]([A_E, tf.concat([H_d, H_t], axis=0)])
            H_d, H_t = H[:self.d_num, :], H[self.d_num:, :]

        R_pred = tf.einsum('ij,kj->ik', H_d, H_t)
        loss = self.calc_loss(self.A, A_R, A_R_filtered, A_E, H_d, H_t, R_pred, self.R)
        self.add_loss(loss)

        return [tf.expand_dims(R_pred, axis=0),
                tf.expand_dims(H_d, axis=0),
                tf.expand_dims(H_t, axis=0)]
        # return [tf.expand_dims(A_R[:self.d_num, self.d_num:], axis=0),
        #         tf.expand_dims(H_d, axis=0),
        #         tf.expand_dims(H_t, axis=0)]

    def get_config(self):
        config = {
            'levels': self.levels,
            'GL_units': self.GL_units,
            'GC_units': self.GC_units,
            'top_k': self.top_k,
            'MI_neg_sampler': self.neg_sampler,
            'epochs': self.epochs,
            'optimizer': self._optimizer,
            'seed': self.seed
        }
        return dict(config, **self.kwargs)

    def fit(self, **kwargs):
        H_d = np.expand_dims(self.H_d, axis=0)
        H_t = np.expand_dims(self.H_t, axis=0)
        x = [H_d, H_t]
        Model.fit(self, x=x, batch_size=1, epochs=self.epochs, verbose=1,
                  callbacks=[self.es_callback])
        [R_pred, H_d_out, H_t_out] = Model.predict(self, x=x, batch_size=1, verbose=2)

        self.R_pred = np.squeeze(R_pred)
        self.R_pred[np.isnan(self.R_pred)] = 0
        self.R_pred[np.isinf(self.R_pred)] = 0
        self.H_d_out = np.squeeze(H_d_out)
        self.H_d_out[np.isnan(self.H_d_out)] = 0
        self.H_d_out[np.isinf(self.H_d_out)] = 0
        self.H_t_out = np.squeeze(H_t_out)
        self.H_t_out[np.isnan(self.H_t_out)] = 0
        self.H_t_out[np.isinf(self.H_t_out)] = 0

    def predict(self, **kwargs):
        return [self.R_pred, self.H_d_out, self.H_t_out,None]

    def calc_loss(self, A, A_R, A_R_filtered, A_E, H_d, H_t, R_pred, R):
        [alpha, beta] = self.hyper_paras[:2]

        loss_global = tf.reduce_mean(tf.square(A - A_R_filtered))
        self.add_metric(loss_global, name='loss_global', aggregation='mean')

        non_zero_idxs = tf.where(A_E[:self.d_num, self.d_num:] > 0)
        H_d_gathered = tf.gather(H_d, non_zero_idxs[:, 0], axis=0)
        H_t_gathered = tf.gather(H_t, non_zero_idxs[:, 1], axis=0)
        H_edge = tf.nn.sigmoid(tf.concat([H_d_gathered, H_t_gathered], axis=1))
        h_graph = tf.reduce_mean(H_edge, axis=0)

        pos_idxs = tf.where(self.R == 1)
        if tf.shape(pos_idxs)[0] > 100:
            r = 1.5
        else:
            r = 10.
        pos_rate = tf.reduce_mean(R)
        mask = (tf.random.uniform(R.shape) < (pos_rate * r)).astype('float32')
        neg_idxs = tf.where((1 - R) * mask > 0)
        neg_idxs = tf.random.shuffle(neg_idxs)[:tf.shape(pos_idxs)[0]]

        if self.neg_sampler == 'FS':
            shuffle_idxs_d = tf.argsort(tf.random.uniform((H_d.shape[0],)))
            H_d_shuffled = tf.gather(H_d, shuffle_idxs_d, axis=0)
            shuffle_idxs_t = tf.argsort(tf.random.uniform((H_t.shape[0],)))
            H_t_shuffled = tf.gather(H_t, shuffle_idxs_t, axis=0)
            H_d_neg_gathered = tf.gather(H_d_shuffled, non_zero_idxs[:, 0], axis=0)
            H_t_neg_gathered = tf.gather(H_t_shuffled, non_zero_idxs[:, 1], axis=0)
            H_edge_neg = tf.nn.sigmoid(tf.concat([H_d_neg_gathered, H_t_neg_gathered], axis=1))
        elif self.neg_sampler == 'FE':
            H_d_fake = tf.gather(H_d, neg_idxs[:, 0], axis=0)
            H_t_fake = tf.gather(H_t, neg_idxs[:, 1], axis=0)
            H_edge_neg = tf.nn.sigmoid(tf.concat([H_d_fake, H_t_fake], axis=1))

        MI_pos = tf.einsum('ij,jk,k->i', H_edge, self.W_d, h_graph)
        loss_MI_pos = -tf.reduce_mean(tf.math.log_sigmoid(MI_pos))
        MI_neg = tf.einsum('ij,jk,k->i', H_edge_neg, self.W_d, h_graph)
        loss_MI_neg = -tf.reduce_mean(1 - tf.math.log_sigmoid(MI_neg))
        loss_MI = loss_MI_pos + loss_MI_neg
        self.add_metric(loss_MI, name='loss_MI', aggregation='mean')

        R_pred_pos = tf.gather_nd(R_pred, pos_idxs)
        R_pred_neg = tf.gather_nd(R_pred, neg_idxs)
        loss_BPR = -tf.reduce_mean(tf.math.log_sigmoid(R_pred_pos - R_pred_neg))
        self.add_metric(loss_BPR, name='loss_BPR', aggregation='mean')

        # loss_LP_pos = tf.reduce_mean(tf.square(
        #     tf.cast(R == 1, 'float32') * (R_pred - R)))
        # loss_LP_neg = tf.reduce_mean(tf.square(
        #     tf.cast(R == 0, 'float32') * (R_pred - R)))
        # loss_LP = loss_LP_pos + 0.2 * loss_LP_neg
        # self.add_metric(loss_LP, name='loss_LP', aggregation='mean')

        return loss_BPR + alpha * loss_global + beta * loss_MI
        # return loss_LP + 1 * loss_global


class GL_Layer(Layer):
    def __init__(self, units=200, top_k=10, seed=0):
        super(GL_Layer, self).__init__()
        self.units = units
        self.top_k = top_k
        self.seed = seed

    def build(self, input_shape):
        d_dim = input_shape[0][-1]
        t_dim = input_shape[1][-1]
        self.W1 = self.add_weight(name='W1',
                                  shape=(d_dim, self.units),
                                  initializer=GlorotUniform(self.seed))
        self.W2 = self.add_weight(name='W2',
                                  shape=(t_dim, self.units),
                                  initializer=GlorotUniform(self.seed + 1))

    def call(self, inputs, **kwargs):
        [H_d, H_t] = inputs
        d_num = H_d.shape[0]
        t_num = H_t.shape[0]

        H_d = tf.matmul(H_d, self.W1)
        H_t = tf.matmul(H_t, self.W2)
        H_d = tf.nn.l2_normalize(H_d, axis=1)
        H_t = tf.nn.l2_normalize(H_t, axis=1)
        S = tf.nn.sigmoid(tf.einsum('ij,kj->ik', H_d, H_t))

        if self.top_k > t_num:
            self.top_k = t_num

        gather_idxs = tf.argsort(S, axis=1, direction='DESCENDING')
        sorted_S = tf.gather(S, gather_idxs, axis=1, batch_dims=1)
        mask = tf.concat([tf.ones((d_num, self.top_k)), tf.zeros((d_num, t_num - self.top_k))],
                         axis=1)
        sorted_S = sorted_S * mask
        gather_idxs_inv = tf.argsort(gather_idxs, axis=1)
        S_filtered = tf.gather(sorted_S, gather_idxs_inv, axis=1, batch_dims=1)
        # sorted_S = tf.sort(S, axis=1, direction='DESCENDING')
        # thresholds = tf.expand_dims(sorted_S[:, self.top_k - 1], axis=1)
        # update_idxs = tf.where(S < thresholds)
        # S_filtered = tf.tensor_scatter_nd_update(S, update_idxs, tf.zeros(tf.shape(update_idxs)[0]))

        A_R = tf.concat([tf.concat([tf.zeros((d_num, d_num)), S], axis=1),
                         tf.concat([tf.transpose(S), tf.zeros((t_num, t_num))], axis=1)],
                        axis=0)
        A_R_filtered = tf.concat([tf.concat([tf.zeros((d_num, d_num)), S_filtered], axis=1),
                                  tf.concat([tf.transpose(S_filtered), tf.zeros((t_num, t_num))],
                                            axis=1)],
                                 axis=0)
        return A_R, A_R_filtered


class GC_Layer(Layer):
    def __init__(self, units=200, seed=0):
        super(GC_Layer, self).__init__()
        self.units = units
        self.seed = seed

    def call(self, inputs, **kwargs):
        [A_E, H_d, H_t] = inputs
        d_num = H_d.shape[0]
        t_num = H_t.shape[0]
        H = tf.concat([H_d, H_t], axis=0)
        D = tf.reduce_sum(A_E, axis=1, keepdims=True)
        zero_idxs = tf.where(D == 0)
        D = tf.tensor_scatter_nd_update(D, zero_idxs, tf.ones(tf.shape(zero_idxs)[0]))
        D_inv = D ** -1
        # nan_idxs = tf.where(tf.math.is_nan(D_inv))
        # D_inv = tf.tensor_scatter_nd_update(D_inv, nan_idxs, tf.zeros(tf.shape(nan_idxs)[0]))
        # inf_idxs = tf.where(tf.math.is_inf(D_inv))
        # D_inv = tf.tensor_scatter_nd_update(D_inv, inf_idxs, tf.zeros(tf.shape(inf_idxs)[0]))
        H = H + tf.matmul(D_inv * A_E, H)
        return [H[:d_num, :], H[d_num:, :]]
