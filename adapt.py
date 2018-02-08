import numpy as np
from utils import ds_loader
from jdot.jdot_nn import JDOT_NN

def adapt(clf, dataset='mars', verbose=False):
    acc_list = []
    X_src, y_src, X_tgt, y_tgt = ds_loader.get_data(dataset)
    subsamples = ds_loader.subsample(X_src, y_src, X_tgt, y_tgt)

    for subsample in subsamples:
        # subsample = (X_src, y_src, X_tgt, y_tgt)
        X_src, y_src, X_tgt, y_tgt = subsample
        if clf is 'JDOT_NN':
            clf = JDOT_NN(X_src.shape[1], 2)
        clf.fit(X_src, y_src, X_tgt)
        _, acc = clf.predict(X_tgt, y=y_tgt)
        acc_list.append(acc)
    if verbose:
        print(dataset +' average accuracy: ', str(np.mean(acc_list)))
        print(dataset +' variance: ', str(np.var(acc_list)))
    return acc_list

def main():
    # adapt(LooRLS(), verbose=True)
    # adapt(JDOT_SVM(), verbose=True)
    # adapt('JDOT_NN', 'supernova', verbose=True)
    pass


if __name__=="__main__":
    main()