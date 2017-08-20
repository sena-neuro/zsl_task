import h5py
import tensorflow as tf
import numpy as np

def batch(iterable, n=1):
    l = len(iterable)
    for ndx in range(0, l, n):
        yield iterable[ndx:min(ndx + n, l)]

def get_training_data(filename):
    f = h5py.File(filename)
    dset_list = {}
    for dsetname, dset in f.iteritems():
        if dsetname == 'Xtr' or dsetname == 'Str_gt' or dsetname == 'Ltr_oh':
            dset_list[dsetname] = np.transpose(dset[:])     # ????????
    return dset_list


dataset = get_training_data('gbu_CUB_data.mat')

import numpy as np
"""
 here we need to batch only Ltr_oh and Xtr 
"""
def next_batch(num, data, labels):
    """
    Return a total of `num` random samples and labels.
    """
    idx = np.arange(0, len(data[0]))
    np.random.shuffle(idx)
    idx = idx[:num]
    data_shuffle = [data[i][:] for i in idx]
    labels_shuffle = [labels[i][:] for i in idx]

    return np.asarray(data_shuffle), np.asarray(labels_shuffle)