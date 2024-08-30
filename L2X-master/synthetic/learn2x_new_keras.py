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

def create_data(datatype, n=1000):
    """
    Create train and validation datasets.
    """
    from make_data2 import generate_data
    x_train, y_train, _ = generate_data(n=n, datatype=datatype, seed=0)
    x_val, y_val, datatypes_val = generate_data(n=10**5, datatype=datatype, seed=1)

    input_shape = x_train.shape[1]

    return x_train, y_train, x_val, y_val, datatypes_val, input_shape

def create_rank(scores, k):
    """
    Compute rank of each feature based on weight.
    """
    scores = abs(scores)
    n, d = scores.shape
    ranks = []
    for i, score in enumerate(scores):
        # Random permutation to avoid bias due to equal weights.
        idx = np.random.permutation(d)
        permuted_weights = score[idx]
        permuted_rank = (-permuted_weights).argsort().argsort() + 1
        rank = permuted_rank[np.argsort(idx)]
        ranks.append(rank)

    return np.array(ranks)

def compute_median_rank(scores, k, datatype_val=None):
    ranks = create_rank(scores, k)
    if datatype_val is None:
        median_ranks = np.median(ranks[:, :k], axis=1)
    else:
        datatype_val = datatype_val[:len(scores)]
        median_ranks1 = np.median(ranks[datatype_val == 'orange_skin', :][:, np.array([0, 1, 2, 3, 9])], axis=1)
        median_ranks2 = np.median(ranks[datatype_val == 'nonlinear_additive', :][:, np.array([4, 5, 6, 7, 9])], axis=1)
        median_ranks = np.concatenate((median_ranks1, median_ranks2), 0)
    return median_ranks

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
        threshold = tf.expand_dims(tf.math.top_k(logits, self.k, sorted=True)[0][:, -1], -1)
        discrete_logits = tf.cast(tf.greater_equal(logits, threshold), tf.float32)

        # Use `training` to switch between training and inference mode
        return samples if training else discrete_logits

    def compute_output_shape(self, input_shape):
        return input_shape

def L2X(datatype, train=True):
    x_train, y_train, x_val, y_val, datatype_val, input_shape = create_data(datatype, n=int(1e6))

    activation = 'relu' if datatype in ['orange_skin', 'XOR'] else 'selu'
    k = ks[datatype]
    tau = 0.1

    model_input = Input(shape=(input_shape,), dtype='float32')

    # P(S|X) network
    net = Dense(100, activation=activation, name='s_dense1',
                kernel_regularizer=regularizers.l2(1e-3))(model_input)
    net = Dense(100, activation=activation, name='s_dense2',
                kernel_regularizer=regularizers.l2(1e-3))(net)
    logits = Dense(input_shape)(net)

    samples = Sample_Concrete(tau, k, name='sample')(logits)

    # q(X_S) network
    new_model_input = Multiply()([model_input, samples])
    net = Dense(200, activation=activation, name='dense1',
                kernel_regularizer=regularizers.l2(1e-3))(new_model_input)
    net = BatchNormalization()(net)
    net = Dense(200, activation=activation, name='dense2',
                kernel_regularizer=regularizers.l2(1e-3))(net)
    net = BatchNormalization()(net)
    preds = Dense(2, activation='softmax', name='dense4',
                  kernel_regularizer=regularizers.l2(1e-3))(net)

    model = Model(model_input, preds)
    current_dir = os.path.dirname(os.path.abspath(__file__))  # Get the directory where the script is located
    filepath = os.path.join(current_dir, 'models', datatype, 'L2X.keras')
    print(filepath)  # Verify the path

    if train:
        adam = optimizers.Adam(learning_rate=1e-3)
        model.compile(loss='categorical_crossentropy',
                      optimizer=adam,
                      metrics=['acc'])
        checkpoint = ModelCheckpoint(filepath, monitor='val_acc',
                                     verbose=1, save_best_only=True, mode='max')
        callbacks_list = [checkpoint]
        model.fit(x_train, y_train, validation_data=(x_val, y_val), callbacks=callbacks_list, epochs=1, batch_size=BATCH_SIZE)
    else:
        model.load_weights(filepath, by_name=True)

    pred_model = Model(model_input, samples)
    pred_model.compile(loss=None, optimizer='rmsprop')

    scores = pred_model.predict(x_val, verbose=1, batch_size=BATCH_SIZE)

    median_ranks = compute_median_rank(scores, k=ks[datatype], datatype_val=datatype_val)

    return median_ranks

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--datatype', type=str, choices=['orange_skin', 'XOR', 'nonlinear_additive', 'switch'], default='orange_skin')
    parser.add_argument('--train', action='store_true')
    args = parser.parse_args()

    args.train=True
    median_ranks = L2X(datatype=args.datatype, train=args.train)
    print(f'Median Ranks: {median_ranks}')
