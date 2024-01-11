import numpy as np
import tensorflow as tf
from tensorflow.python.keras.callbacks import EarlyStopping
from tensorflow.python.keras.models import Model
from tensorflow.python.keras.initializers.initializers_v2 import GlorotUniform

from common_functions.GNN_layers import GCN_Layer
from common_functions.metrics import MetricUtils
from common_functions.configs import Configs


class DGI_Model(Model):

    def __init__(self, input_data=None,
                 levels=2, units=200, epochs=200, optimizer='adam', loss_mode='RDT',
                 seed=0, hyper_paras=(1,), **kwargs):
        super(DGI_Model, self).__init__()
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
        # self.R = R
        self.d_num = D.shape[0]
        self.t_num = T.shape[0]
        self.R_pred = None
        self.H_d_out = None
        self.H_t_out = None

        self.W = self.add_weight(name='W',
                                 shape=(self.units, self.units),
                                 initializer=GlorotUniform(self.seed))

    def call(self, inputs, training=None, mask=None):
        H_raw, H_crp = inputs[0][0], inputs[1][0]

        for i in range(self.levels):
            H_raw = self.GCN_layers[i]([self.A, H_raw])
            H_crp = self.GCN_layers[i]([self.A, H_crp])
        h_raw = tf.reduce_mean(H_raw, axis=0)
        H_out = (H_raw + H_crp) / 2
        H_d_out = H_out[:self.d_num, :]
        H_t_out = H_out[self.d_num:, :]
        A_pred = tf.einsum('ij,kj->ik', H_out, H_out)
        R_pred = tf.einsum('ij,kj->ik', H_d_out, H_t_out)

        if self.loss_mode == 'RDT' and self.kwargs['use_D'] and self.kwargs['use_T']:
            loss = self.calc_loss(H_raw, H_crp, h_raw, A_pred, self.A)
        else:
            loss = self.calc_loss(H_raw, H_crp, h_raw, R_pred, self.R)
        self.add_loss(loss)

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
        H_raw = self.H
        H_crp = np.random.permutation(H_raw)
        H_raw = np.expand_dims(H_raw, axis=0)
        H_crp = np.expand_dims(H_crp, axis=0)
        x = [H_raw, H_crp]
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
        return [self.R_pred, self.H_d_out, self.H_t_out, None]

    def calc_loss(self, H_raw, H_crp, h_raw, X_pred, X):
        alpha = self.hyper_paras[0]

        MI_pos = tf.einsum('ij,jk,k->i', H_raw, self.W, h_raw)
        loss_MI_pos = -tf.reduce_mean(tf.math.log_sigmoid(MI_pos))

        MI_neg = tf.einsum('ij,jk,k->i', H_crp, self.W, h_raw)
        loss_MI_neg = -tf.reduce_mean(tf.math.log_sigmoid(1 - MI_neg))

        loss_MI = loss_MI_pos + loss_MI_neg

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
