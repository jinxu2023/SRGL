import numpy as np
import tensorflow as tf
from tensorflow.python.keras.callbacks import EarlyStopping
from tensorflow.python.keras.models import Model
from tensorflow.python.keras.layers import Layer
from tensorflow.python.keras.initializers.initializers_v2 import GlorotUniform

from common_functions.metrics import MetricUtils
from common_functions.configs import Configs


class NGCF_Model(Model):

    def __init__(self, input_data=None,
                 levels=2, units=200, epochs=200, optimizer='adam', loss_mode='RDT',
                 seed=0, **kwargs):
        super(NGCF_Model, self).__init__()
        self.levels = levels
        self.units = units
        self.epochs = epochs
        self._optimizer = optimizer
        self.loss_mode = loss_mode
        self.seed = seed
        self.kwargs = kwargs

        self.init_data(input_data)
        self.GC_layers = [GC_Layer(L=self.L,
                                   units=units,
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
        self.R_truth = tf.convert_to_tensor(R_truth, dtype='float32')
        self.mask = tf.convert_to_tensor(mask, dtype='float32')

        A = np.r_[np.c_[D, R_train], np.c_[R_train.T, T]]
        row_sum = np.sum(A, axis=1, keepdims=True)
        row_sum = row_sum + (row_sum == 0)
        row_norm = row_sum ** -0.5
        self.L = tf.convert_to_tensor(row_norm * A * row_norm.T)
        self.R = tf.convert_to_tensor(R_train, 'float32')
        self.D = tf.convert_to_tensor(D, 'float32')
        self.T = tf.convert_to_tensor(T, 'float32')
        self.H = np.r_[H_d, H_t]

        self.d_num, self.t_num = self.R.shape
        self.R_pred = None
        self.H_d_out = None
        self.H_t_out = None

    def call(self, inputs, training=None, mask=None):
        H = inputs[0][0]
        for i in range(self.levels):
            H = self.GC_layers[i](H)
        H_d, H_t = H[:self.d_num], H[self.d_num:]

        loss = 0
        R_pred = H_d @ H_t.T
        if self.loss_mode.__contains__('R'):
            loss += self.loss_BPR(R_pred, self.R)
        if self.loss_mode.__contains__('DT'):
            D_pred = H_d @ H_d.T
            T_pred = H_t @ H_t.T
            loss += self.loss_BPR(D_pred, self.D)
            loss += self.loss_BPR(T_pred, self.T)
        self.add_loss(loss)

        return [tf.expand_dims(R_pred, axis=0),
                tf.expand_dims(H_d, axis=0),
                tf.expand_dims(H_t, axis=0)]

    def get_config(self):
        config = {
            'levels': self.levels,
            'units': self.units,
            'epochs': self.epochs,
            'optimizer': self._optimizer,
            'loss_mode': self.loss_mode,
            'seed': self.seed
        }
        return dict(config, **self.kwargs)

    def fit(self, **kwargs):
        H = tf.expand_dims(self.H, axis=0)
        Model.fit(self, x=[H], batch_size=1, epochs=self.epochs, verbose=1,
                  callbacks=[self.es_callback])
        [R_pred, H_d_out, H_t_out] = Model.predict(self, x=[H], batch_size=1, verbose=2)
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
    def loss_BPR(X_pred, X):
        pos_idxs = tf.where(X == 1)
        if tf.shape(pos_idxs)[0] > 100:
            r = 1.5
        else:
            r = 10.
        pos_rate = tf.reduce_mean(X)
        mask = (tf.random.uniform(X.shape) < (pos_rate * r)).astype('float32')
        neg_idxs = tf.where((1 - X) * mask > 0)
        neg_idxs = tf.random.shuffle(neg_idxs)[:tf.shape(pos_idxs)[0]]
        X_pred_pos = tf.gather_nd(X_pred, pos_idxs)
        X_pred_neg = tf.gather_nd(X_pred, neg_idxs)
        loss = -tf.reduce_mean(tf.math.log_sigmoid(X_pred_pos - X_pred_neg))
        return loss


class GC_Layer(Layer):
    def __init__(self, L=None, units=200, seed=0):
        super(GC_Layer, self).__init__()
        self.L = L
        self.units = units
        self.seed = seed

    def build(self, input_shapes):
        self.W1 = self.add_weight(name='W1',
                                  shape=(input_shapes[-1], self.units),
                                  initializer=GlorotUniform(self.seed))
        self.W2 = self.add_weight(name='W2',
                                  shape=(input_shapes[-1] * 2, self.units),
                                  initializer=GlorotUniform(self.seed + 1))

    def call(self, inputs, **kwargs):
        H = inputs
        H_left = (self.L + tf.eye(self.L.shape[0])) @ H @ self.W1
        H_right = tf.concat([self.L @ H, H], axis=1) @ self.W2
        H_out = tf.nn.leaky_relu(H_left + H_right)
        return H_out
