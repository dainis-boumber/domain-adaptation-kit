import numpy as np
from rlscore.learner import LeaveOneOutRLS
from rlscore.measure import ova_accuracy
from rlscore.utilities.multiclass import to_one_vs_all
from rlscore.learner import RLS
from rlscore.measure import accuracy


class LooRLS(object):

    def __init__(self):
        self.learner = None
        self.y_src = None
        self.measure = None

    def fit(self, X_src, y_src, X_tgt, verbose=False):
        # Map labels from set {1,2,3} to one-vs-all encoding

        if np.count_nonzero(y_src) >= len(y_src):
            zerolabels = False
        else:
            zerolabels = True

        y_src = to_one_vs_all(y_src, zerolabels)

        regparams = [2. ** i for i in range(-15, 16)]
        if len(np.unique(y_src)) > 2:
            self.measure = ova_accuracy
        else:
            self.measure = accuracy

        self.learner = LeaveOneOutRLS(X_src, y_src, regparams=regparams, measure=self.measure)
        p_tgt = self.learner.predict(X_tgt)
        # ova_accuracy computes one-vs-all classification accuracy directly between transformed
        # class label matrix, and a matrix of predictions, where each column corresponds to a class
        self.learner = RLS(X_src, y_src)
        best_regparam = None
        best_accuracy = 0.
        # exponential grid of possible regparam values
        log_regparams = range(-15, 16)
        for log_regparam in log_regparams:
            regparam = 2.**log_regparam
            # RLS is re-trained with the new regparam, this
            # is very fast due to computational short-cut
            self.learner.solve(regparam)
            # Leave-one-out cross-validation predictions, this is fast due to
            # computational short-cut
            P_loo = self.learner.leave_one_out()
            acc = self.measure(y_src, P_loo)
            if verbose == True:
                print("LooRLS regparam 2**%d, loo-accuracy %f" % (log_regparam, acc))
            if acc > best_accuracy:
                best_accuracy = acc
                best_regparam = regparam
        self.learner.solve(best_regparam)
        if verbose == True:
            print("LooRLS best regparam %f with loo-accuracy %f" % (best_regparam, best_accuracy))

    def predict(self, X, y=None):
        ypred = self.learner.predict(X)
        if y is not None:
            if np.count_nonzero(y) >= len(y):
                zerolabels = False
            else:
                zerolabels = True
            y = to_one_vs_all(y, zerolabels)
            return ypred, self.measure(y, ypred)
        return ypred


