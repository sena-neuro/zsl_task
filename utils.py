import h5py
import numpy as np
import os


def get_data(filename):
    f = h5py.File(filename,"r")
    dset_list = {}
    for dsetname, dset in f.iteritems():
        dset_list[dsetname] = dset.value.astype(np.float32).T
    return dset_list

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
