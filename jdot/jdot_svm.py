# -*- coding: utf-8 -*-
"""
Classification example for JDOT
"""

# Author: Remi Flamary <remi.flamary@unice.fr>
#         Nicolas Courty <ncourty@irisa.fr>
#
# License: MIT License

from models import classif
from models import svm
from sklearn.svm import LinearSVC
from sklearn.metrics import accuracy_score


class JDOT_SVM(object):

    def __init__(self):
        self.model = classif.SVMClassifier(1e-1)
        self.X_test = None

    def fit(self, X_src, y_src, X_test, y_test, verbose=False):
        Y,Yb= classif.get_label_matrix(y_src)
        self.model.fit(X=X_src,y=Yb)
        clf_jdot, dic = svm.jdot_svm(X=X_src, y=Y, Xtest=X_test, ytest=y_test, gamma_g=.1, numIterBCD=10, alpha=.1, lambd=reg,
                                     ktype='rbf')  # ,method='sinkhorn',reg=0.01)

    def predict_test(clf, gamma, Xapp, Xtest):
        Kx = classif.linear_kernel(Xtest, Xapp, gamma=gamma)
        return clf.predict(Kx)

    def predict(self, X, y = None):
        ypred=self.model.predict(X)

        if y is not None:
            return ypred, accuracy_score(y_true=y, y_pred=ypred)
        return ypred


