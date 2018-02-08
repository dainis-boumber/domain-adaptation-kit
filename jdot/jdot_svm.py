# -*- coding: utf-8 -*-
"""
Classification example for JDOT
"""

# Author: Remi Flamary <remi.flamary@unice.fr>
#         Nicolas Courty <ncourty@irisa.fr>
#
# License: MIT License

from models import classif
from sklearn.svm import LinearSVC
from sklearn.metrics import accuracy_score


class JDOT_SVM(object):

    def __init__(self):
        self.model = LinearSVC()
        self.X_test = None

    def fit(self, X_src, y_src, verbose=False):
        Y,Yb= classif.get_label_matrix(y_src)
        self.model.fit(X=X_src,y=y_src)

    def predict(self, X, y = None):
        ypred=self.model.predict(X)
        if y is not None:
            return ypred, accuracy_score(y_true=y, y_pred=ypred)
        return ypred


