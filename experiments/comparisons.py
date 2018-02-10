import numpy as np
from utils import ds_loader


def compare_classifiers(classifiers, datasets, verbose=True):
    with open("results.txt", 'a+') as f:
        for ds in datasets:
            subsamples = ds_loader.get_data(ds)
            accuracies = []

            for s in subsamples:
                X_src, y_src, X_tgt_known, y_tgt_known, X_tgt_unknown, y_tgt_unknown = s

                for clf in classifiers:
                    f.write('\n\n' + str(clf.__class__.__name__) + '\n')
                    clf.fit(X_src, y_src, X_tgt_known, y_tgt_known, X_tgt_unknown, y_tgt_unknown, verbose)
                    _, acc = clf.predict(X_tgt_unknown, y_tgt_unknown)
                    accuracies.append(acc)
                if verbose:
                    print(ds + ' average accuracy: ', str(np.mean(accuracies)))
                    print(ds + ' variance: ', str(np.var(accuracies)))
                f.write(ds + ' average accuracy: ' + str(np.mean(accuracies)) + '\n')
                f.write(ds + ' variance: ', str(np.var(accuracies)) + '\n')
