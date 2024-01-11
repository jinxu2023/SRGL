import numpy as np
import tensorflow as tf
from tensorflow.python.keras.callbacks import EarlyStopping
from tensorflow.python.keras.models import Model
from tensorflow.python.keras.layers import Dense
from tensorflow.python.keras.initializers.initializers_v2 import GlorotUniform

from common_functions.GNN_layers import GCN_Layer
from common_functions.configs import Configs


class GCA_Model(Model):

    def __init__(self, input_data=None,
                 levels=2, units=200, epochs=200, optimizer='adam', loss_mode='RDT',
                 seed=0, hyper_paras=(1, 0.5, 0.5, 0.2, 0.1, 0.7, 0.1), **kwargs):
        super(GCA_Model, self).__init__()
        self.levels = levels
        self.units = units
        self.epochs = epochs
        self._optimizer = optimizer
        self.seed = seed
        self.hyper_paras = hyper_paras
        self.loss_mode = loss_mode
        self.kwargs = kwargs

        self.init_data(input_data)
        self.GCN_layers = [GCN_Layer(units=units,
                                     activation=tf.nn.relu,
                                     seed=seed)
                           for i in range(levels)]
        self.MLP = Dense(units=units,
                         activation=tf.nn.relu,
                         use_bias=True)
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
        self.R = tf.convert_to_tensor(R_train, 'float32')
        self.R_truth = tf.convert_to_tensor(R_truth, dtype='float32')
        self.mask = tf.convert_to_tensor(mask, dtype='float32')
        A = np.r_[np.c_[D, R_train], np.c_[R_train.T, T]]
        self.A = tf.convert_to_tensor(A, 'float32')
        self.H = np.r_[H_d, H_t]
        self.d_num = D.shape[0]
        self.t_num = T.shape[0]
        self.R_pred = None
        self.H_d_out = None
        self.H_t_out = None
        self.w_MI, self.p_e1, self.p_e2, self.p_f1, self.p_f2, self.p_t, self.t = self.hyper_paras

    @staticmethod
    def graph_augmenting(A, p_e, p_t):
        node_num = A.shape[0]
        deg_A = tf.reduce_sum(A, axis=1)
        deg_A1 = tf.repeat(deg_A, node_num)
        deg_A2 = tf.concat([deg_A] * node_num, axis=0)
        W_e = 0.5 * (deg_A1 + deg_A2).reshape((node_num, node_num))
        S_e = tf.math.log(W_e)
        S_e_max = tf.reduce_max(S_e)
        S_e_avg = tf.reduce_mean(S_e)
        P_e = tf.minimum((S_e_max - S_e) / (S_e_max - S_e_avg) * p_e, p_t)
        A_aug = A * (tf.random.uniform(A.shape) > P_e).astype('float32')
        return A_aug

    @staticmethod
    def feature_augmenting(A, H, p_f, p_t):
        deg_A = tf.reduce_sum(A, axis=1, keepdims=True)
        w_f = tf.reduce_sum(H * deg_A, axis=0)
        s_f = tf.math.log(w_f)
        s_f_max = tf.reduce_max(s_f)
        s_f_avg = tf.reduce_mean(s_f)
        pro_f = tf.minimum((s_f_max - s_f_avg) / (s_f_max - s_f_avg) * p_f, p_t)
        H_aug = H * pro_f
        return H_aug

    def call(self, inputs, training=None, mask=None):
        H = inputs[0][0]

        A_aug1 = self.graph_augmenting(self.A, self.p_e1, self.p_t)
        A_aug2 = self.graph_augmenting(self.A, self.p_e2, self.p_t)
        H_aug1 = self.feature_augmenting(self.A, H, self.p_f1, self.p_t)
        H_aug2 = self.feature_augmenting(self.A, H, self.p_f2, self.p_t)
        for i in range(self.levels):
            H_aug1 = self.GCN_layers[i]([A_aug1, H_aug1])
            H_aug2 = self.GCN_layers[i]([A_aug2, H_aug2])
        H_out = 0.5 * (H_aug1 + H_aug2)
        H_d_out = H_out[:self.d_num, :]
        H_t_out = H_out[self.d_num:, :]
        A_pred = tf.einsum('ij,kj->ik', H_out, H_out)
        R_pred = tf.einsum('ij,kj->ik', H_d_out, H_t_out)

        if self.loss_mode == 'RDT' and self.kwargs['use_D'] and self.kwargs['use_T']:
            loss_LP = self.loss_LP(A_pred, self.A)
        else:
            loss_LP = self.loss_LP(R_pred, self.R)
        loss_MI = self.loss_MI(H_aug1, H_aug2, self.t, self.MLP)
        self.add_loss(loss_LP + self.w_MI * loss_MI)

        return [tf.expand_dims(R_pred, axis=0),
                tf.expand_dims(H_d_out, axis=0),
                tf.expand_dims(H_t_out, axis=0)]

    def get_config(self):
        config = {
            'levels': self.levels,
            'units': self.units,
            'epochs': self.epochs,
            'optimizer': self._optimizer,
            'seed': self.seed,
            'loss_mode': self.loss_mode
        }
        return dict(config, **self.kwargs)

    def fit(self, **kwargs):
        x = tf.expand_dims(self.H, axis=0)
        Model.fit(self, x=[x], batch_size=1, epochs=self.epochs, verbose=1,
                  callbacks=[self.es_callback])
        [R_pred, H_d_out, H_t_out] = Model.predict(self, x=[x], batch_size=1, verbose=2)
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

    @staticmethod
    def loss_LP(X_pred, X):
        loss_LP_pos = tf.reduce_mean(tf.square(
            tf.cast(X == 1, 'float32') * (X_pred - X)))
        loss_LP_neg = tf.reduce_mean(tf.square(
            tf.cast(X == 0, 'float32') * (X_pred - X)))
        return loss_LP_pos + 0.2 * loss_LP_neg

    @staticmethod
    def loss_MI(H_aug1, H_aug2, t, MLP):
        node_num = H_aug1.shape[0]
        H_aug1_out = MLP(H_aug1)
        H_aug2_out = MLP(H_aug2)
        H_aug1_out = tf.nn.l2_normalize(H_aug1_out, axis=1)
        H_aug2_out = tf.nn.l2_normalize(H_aug2_out, axis=1)
        S_cosine11 = H_aug1_out @ H_aug1_out.T
        S_cosine12 = H_aug1_out @ H_aug2_out.T
        S_cosine21 = H_aug2_out @ H_aug1_out.T
        S_cosine22 = H_aug2_out @ H_aug2_out.T

        MI1 = MI2 = tf.constant(0.0)
        for i in tf.range(node_num):
            MI_pos12 = tf.math.exp(S_cosine12[i, i] / t)
            MI_neg12 = tf.reduce_sum(tf.math.exp(S_cosine12[i, :i] / t)) + \
                       tf.reduce_sum(tf.math.exp(S_cosine12[i, i + 1:] / t))
            MI_neg11 = tf.reduce_sum(tf.math.exp(S_cosine11[i, :i] / t)) + \
                       tf.reduce_sum(tf.math.exp(S_cosine11[i, i + 1:] / t))
            MI1 += tf.math.log(MI_pos12 / (MI_pos12 + MI_neg12 + MI_neg11))
            MI_pos21 = tf.math.exp(S_cosine21[i, i] / t)
            MI_neg21 = tf.reduce_sum(tf.math.exp(S_cosine21[i, :i] / t)) + \
                       tf.reduce_sum(tf.math.exp(S_cosine21[i, i + 1:] / t))
            MI_neg22 = tf.reduce_sum(tf.math.exp(S_cosine22[i, :i] / t)) + \
                       tf.reduce_sum(tf.math.exp(S_cosine22[i, i + 1:] / t))
            MI2 += tf.math.log(MI_pos21 / (MI_pos21 + MI_neg21 + MI_neg22))

        loss_MI = -0.5 / node_num * (MI1 + MI2)
        return loss_MI
