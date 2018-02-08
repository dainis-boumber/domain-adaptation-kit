import numpy as np
from sklearn.utils import resample

def get_data(dataset):
    src_fpath = 'data/' + dataset + '/sources/' + dataset + '-src.csv'
    tgt_fpath = 'data/' + dataset + '/targets/' + dataset + '-tgt.csv'
    src_mat = np.loadtxt(open(src_fpath, "rb"), delimiter=",")
    tgt_mat = np.loadtxt(open(tgt_fpath, "rb"), delimiter=",")
    X_src = src_mat[:,:-1]
    y_src = src_mat[:,-1]
    X_tgt = tgt_mat[:,:-1]
    y_tgt = tgt_mat[:,-1]
    return X_src, y_src, X_tgt, y_tgt

def subsample(X_src, y_src, X_tgt, y_tgt, subsample_sz=0.5, n_subsamples=10):
    subsamples = []

    for _ in range(n_subsamples):
        X_src_sample, y_src_sample = resample(X_src, y_src, n_samples=int(len(y_src) * subsample_sz))
        X_tgt_sample, y_tgt_sample = resample(X_tgt, y_tgt, n_samples=int(len(y_tgt) * subsample_sz))
        subsamples.append((X_src_sample, y_src_sample, X_tgt_sample, y_tgt_sample))

    return subsamples

