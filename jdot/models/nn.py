
import numpy as np
from scipy.spatial.distance import cdist
import ot


def jdot_nn_l2(get_model, X, Y, Xtest, ytest=[], fit_params={}, reset_model=True, numIterBCD=10, alpha=1, method='emd',
               reg=1, nb_epoch=100, batch_size=10):
    # get model should return a new model compiled with l2 loss

    # Initializations
    n = X.shape[0]
    ntest = Xtest.shape[0]
    wa = np.ones((n,)) / n
    wb = np.ones((ntest,)) / ntest

    # original loss
    C0 = cdist(X, Xtest, metric='sqeuclidean')
    C0 = C0 / np.max(C0)

    # classifier
    g = get_model()

    TBR = []
    sav_fcost = []
    sav_totalcost = []

    results = {}

    # Init initial g(.)
    g.fit(X, Y, **fit_params)
    ypred = g.predict(Xtest)

    C = alpha * C0 + cdist(Y, ypred, metric='sqeuclidean')

    # do it only if the final labels were given
    if len(ytest):
        ydec = np.argmax(ypred, 1) + 1
        TBR1 = np.mean(ytest == ydec)
        TBR.append(TBR1)

    k = 0
    changeLabels = False
    while (k < numIterBCD):  # and not changeLabels:
        k = k + 1
        if method == 'sinkhorn':
            G = ot.sinkhorn(wa, wb, C, reg)
        if method == 'emd':
            G = ot.emd(wa, wb, C)

        Yst = ntest * G.T.dot(Y)

        if reset_model:
            g = get_model()

        g.fit(Xtest, Yst, **fit_params)
        ypred = g.predict(Xtest)

        # function cost
        fcost = cdist(Y, ypred, metric='sqeuclidean')
        # pl.figure()
        # pl.imshow(fcost)
        # pl.show()

        C = alpha * C0 + fcost

        ydec_tmp = np.argmax(ypred, 1) + 1
        if k > 1:
            changeLabels = np.all(ydec_tmp == ydec)
            sav_fcost.append(np.sum(G * fcost))
            sav_totalcost.append(np.sum(G * (alpha * C0 + fcost)))

        ydec = ydec_tmp
        if len(ytest):
            TBR1 = np.mean((ytest - ypred) ** 2)
            TBR.append(TBR1)

    results['ypred0'] = ypred
    results['ypred'] = np.argmax(ypred, 1) + 1
    if len(ytest):
        results['mse'] = TBR
    results['clf'] = g
    results['fcost'] = sav_fcost
    results['totalcost'] = sav_totalcost
    return g, results