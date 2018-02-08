# -*- coding: utf-8 -*-
"""
Created on Thu Sep 21 16:59:10 2017

@author: rflamary
"""

# Author: Remi Flamary <remi.flamary@unice.fr>
#         Nicolas Courty <ncourty@irisa.fr>
#
# License: MIT License

import numpy as np
from scipy.spatial.distance import cdist
from models import classif
import sklearn
import ot

#from sklearn import datasets


# X: source domain
# y: source labeks
# Xtest: target domain
# ytest is optionnal, just to measure performances of the method along iterations
# gamma: RBF kernel param (default=1)
# numIterBCD: number of Iterations for BCD (default=10)
# alpha: ponderation between ground cost + function cost
# method: choice of algorithm for transport computation (default: emd)


def jdot_krr(X,y,Xtest,gamma_g=1, numIterBCD = 10, alpha=1,lambd=1e1, 
             method='emd',reg=1,ktype='linear'):
    # Initializations
    n = X.shape[0]
    ntest = Xtest.shape[0]
    wa=np.ones((n,))/n
    wb=np.ones((ntest,))/ntest

    # original loss
    C0=cdist(X,Xtest,metric='sqeuclidean')
    #print np.max(C0)
    C0=C0/np.median(C0)

    # classifier    
    g = classif.KRRClassifier(lambd)

    # compute kernels
    if ktype=='rbf':
        Kt=sklearn.metrics.pairwise.rbf_kernel(Xtest,Xtest,gamma=gamma_g)
    else:
        Kt=sklearn.metrics.pairwise.linear_kernel(Xtest,Xtest)

    C = alpha*C0#+ cdist(y,ypred,metric='sqeuclidean')
    k=0
    while (k<numIterBCD):# and not changeLabels:
        k=k+1
        if method=='sinkhorn':
            G = ot.sinkhorn(wa,wb,C,reg)
        if method=='emd':
            G=  ot.emd(wa,wb,C)

        Yst=ntest*G.T.dot(y)

        g.fit(Kt,Yst)
        ypred=g.predict(Kt)
       
        # function cost
        fcost = cdist(y,ypred,metric='sqeuclidean')

        C=alpha*C0+fcost
            
    return g,np.sum(G*(fcost))    


    

