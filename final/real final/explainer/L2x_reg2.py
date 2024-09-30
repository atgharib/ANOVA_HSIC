import numpy as np
import tensorflow as tf
from tensorflow.keras import backend as K
from tensorflow.keras.layers import Layer
from tensorflow.keras.layers import Dense, Input, Flatten, Add, Multiply, Lambda, BatchNormalization
from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras import regularizers
from tensorflow.keras import optimizers
import random
import os
import time

BATCH_SIZE = 1000
np.random.seed(0)
tf.random.set_seed(0)
random.seed(0)

# The number of key features for each data set.
ks = {'orange_skin': 4, 'XOR': 2, 'nonlinear_additive': 4, 'switch': 5}

class Sample_Concrete(Layer):
    """
    Layer for sampling Concrete / Gumbel-Softmax variables.
    """
    def __init__(self, tau0, k, **kwargs):
        self.tau0 = tau0
        self.k = k
        super(Sample_Concrete, self).__init__(**kwargs)

    def call(self, logits, training=None):
        # logits: [BATCH_SIZE, d]
        logits_ = tf.expand_dims(logits, -2)  # [BATCH_SIZE, 1, d]

        batch_size = tf.shape(logits_)[0]
        d = tf.shape(logits_)[2]
        uniform = tf.random.uniform(shape=(batch_size, self.k, d),
                                    minval=tf.keras.backend.epsilon(),
                                    maxval=1.0)

        gumbel = -K.log(-K.log(uniform))
        noisy_logits = (gumbel + logits_) / self.tau0
        samples = K.softmax(noisy_logits)
        samples = K.max(samples, axis=1)

        # Explanation Stage output.
        # threshold = tf.expand_dims(tf.math.top_k(logits, self.k, sorted=True)[0][:, -1], -1)
        # discrete_logits = tf.cast(tf.greater_equal(logits, threshold), tf.float32)

        # Use `training` to switch between training and inference mode
        # return samples if training else discrete_logits
        return samples

    def compute_output_shape(self, input_shape):
        return input_shape

def L2X(X, y, input_shape, num_feature_imp):
   
    activation = 'relu'
    tau = 0.1

    model_input = Input(shape=(input_shape,), dtype='float32')

    # P(S|X) network
    net = Dense(100, activation=activation, name='s_dense1',
                kernel_regularizer=regularizers.l2(1e-3))(model_input)
    net = Dense(100, activation=activation, name='s_dense2',
                kernel_regularizer=regularizers.l2(1e-3))(net)
    logits = Dense(input_shape)(net)

    samples = Sample_Concrete(tau, num_feature_imp, name='sample')(logits)

    # q(X_S) network
    new_model_input = Multiply()([model_input, samples])
    net = Dense(200, activation=activation, name='dense1',
                kernel_regularizer=regularizers.l2(1e-3))(new_model_input)
    net = BatchNormalization()(net)
    net = Dense(200, activation=activation, name='dense2',
                kernel_regularizer=regularizers.l2(1e-3))(net)
    net = BatchNormalization()(net)
    preds = Dense(1, activation='linear', name='dense4',
                  kernel_regularizer=regularizers.l2(1e-3))(net)

    model = Model(model_input, preds)


    adam = optimizers.Adam(learning_rate=1e-3)
    model.compile(loss='mse',
                    optimizer=adam,
                    metrics=['acc'])
    model.fit(X, y, epochs=1, batch_size=BATCH_SIZE)


    pred_model = Model(model_input, samples)
    pred_model.compile(loss=None, optimizer='rmsprop')

   

    return pred_model

  
