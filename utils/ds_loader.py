import numpy as np
import model_selection
from definitions import PROJECT_ROOT

def get_data(dataset, n_subsamples=10):
    src_fpath = PROJECT_ROOT + '/data/' + dataset + '/sources/' + dataset + '-src.csv'
    tgt_fpath = PROJECT_ROOT + '/data/' + dataset + '/targets/' + dataset + '-tgt.csv'
    src_mat = np.loadtxt(open(src_fpath, "rb"), delimiter=",")
    tgt_mat = np.loadtxt(open(tgt_fpath, "rb"), delimiter=",")
    subsamples = []
    for _ in range(n_subsamples):
        X_tgt_known, X_tgt_unknown = model_selection.active_da_train_test_split(tgt_mat)
        X_src = src_mat[:,:-1]
        y_src = src_mat[:,-1]
        X_tgt_known = X_tgt_known[:,:-1]
        y_tgt_known = X_tgt_known[:,-1]
        X_tgt_unknown= X_tgt_unknown[:, :-1]
        y_tgt_unknown = X_tgt_unknown[:, -1]
        subsamples.append((X_src, y_src, X_tgt_known, y_tgt_known, X_tgt_unknown, y_tgt_unknown))

    return subsamples


