import numpy as np
from scipy.spatial.distance import cdist
from models import classif
import sklearn
import ot

def jdot_svm(X, y, Xtest,
             ytest=[], gamma_g=1, numIterBCD=10, alpha=1,
             lambd=1e1, method='emd', reg_sink=1, ktype='linear'):
    # Initializations
    n = X.shape[0]
    ntest = Xtest.shape[0]
    wa = np.ones((n,)) / n
    wb = np.ones((ntest,)) / ntest

    # original loss
    C0 = cdist(X, Xtest, metric='sqeuclidean')

    # classifier
    g = classif.SVMClassifier(lambd)

    # compute kernels
    if ktype == 'rbf':
        Kt = sklearn.metrics.pairwise.rbf_kernel(Xtest, gamma=gamma_g)
        # Ks=sklearn.metrics.pairwise.rbf_kernel(X,gamma=gamma_g)
    else:
        Kt = sklearn.metrics.pairwise.linear_kernel(Xtest)
        # Ks=sklearn.metrics.pairwise.linear_kernel(X)

    TBR = []
    sav_fcost = []
    sav_totalcost = []

    results = {}
    ypred = np.zeros(y.shape)

    Chinge = np.zeros(C0.shape)
    C = alpha * C0 + Chinge

    # do it only if the final labels were given
    if len(ytest):
        TBR.append(np.mean(ytest == np.argmax(ypred, 1) + 1))

    k = 0
    while (k < numIterBCD):
        k = k + 1
        if method == 'sinkhorn':
            G = ot.sinkhorn(wa, wb, C, reg_sink)
        if method == 'emd':
            G = ot.emd(wa, wb, C)

        if k > 1:
            sav_fcost.append(np.sum(G * Chinge))
            sav_totalcost.append(np.sum(G * (alpha * C0 + Chinge)))

        Yst = ntest * G.T.dot((y + 1) / 2.)
        # Yst=ntest*G.T.dot(y_f)
        g.fit(Kt, Yst)
        ypred = g.predict(Kt)

        Chinge = classif.loss_hinge(y, ypred)
        # Chinge=SVMclassifier.loss_hinge(y_f*2-1,ypred*2-1)

        C = alpha * C0 + Chinge

        if len(ytest):
            TBR1 = np.mean(ytest == np.argmax(ypred, 1) + 1)
            TBR.append(TBR1)

    results['ypred'] = np.argmax(ypred, 1) + 1
    if len(ytest):
        results['TBR'] = TBR

    results['clf'] = g
    results['G'] = G
    results['fcost'] = sav_fcost
    results['totalcost'] = sav_totalcost
