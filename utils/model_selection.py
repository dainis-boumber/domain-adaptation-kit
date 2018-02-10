from random import shuffle

def active_da_train_test_split(dataset, test_portion=0.5, n_known_samples=100):
    len_test = int(len(dataset) * test_portion)
    indices = list(range(len(dataset)))
    shuffle(indices)
    test_indices = indices[:len_test]
    known_indices = indices[len_test:len_test+n_known_samples]
    return dataset[known_indices], dataset[test_indices]