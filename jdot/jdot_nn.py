# -*- coding: utf-8 -*-
#
#
#
# Author: Remi Flamary <remi.flamary@unice.fr>
#         Nicolas Courty <ncourty@irisa.fr>
#
# License: MIT License

import numpy as np
from models import nn
from keras.utils import np_utils
import keras


class JDOT_NN(object):

    def __init__(self, n_inputs, n_classes):
        self.n_classes = n_classes
        self.n_inputs = n_inputs
        self.model = None

    def build_model(self):
        # simple 1D nn
        net=keras.models.Sequential()
        net.add(keras.layers.Dense(80,activation='tanh', input_dim=self.n_inputs))
        net.add(keras.layers.Dense(80,activation='tanh'))
        net.add(keras.layers.Dense(self.n_classes,activation='softmax'))
        if self.n_classes > 2:
            net.compile(optimizer='sgd',loss='categorical_crossentropy', metrics=['categorical_accuracy'])
        else:
            net.compile(optimizer='sgd', loss='categorical_crossentropy', metrics=['accuracy'])
        return net

    def fit(self, X_src, y_src, X_tgt):
        y_src = np_utils.to_categorical(y_src)
        self.model=self.build_model()
        fit_params = {'epochs': 2}
        self.model, loss = nn.jdot_nn_l2(self.build_model, X_src, y_src, Xtest=X_tgt, ytest=[], fit_params=fit_params,
                                         reset_model=True, numIterBCD=10, alpha=1, method='emd', reg=1, nb_epoch=2,
                                         batch_size=64)


    def predict(self, X, y=None):
        y = np_utils.to_categorical(y)
        pred = self.model.predict(X, batch_size=64)

        if y is not None:
            res = self.model.evaluate(x=X, y=y, batch_size=64)
            return pred, res[1]
        return pred

