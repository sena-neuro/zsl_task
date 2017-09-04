import h5py
import numpy as np
import os
import tensorflow as tf
import math

batch_numbers = []
batch__is_used = []

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


def next_batch_2(num, data, labels):

    print len(data), len(labels)
    # For first creation of batch and resetting batches
    if not False in batch__is_used:

        # Finds how many batches possible
        number_of_batches = int(math.ceil(float(len(data)) / num))
        print " First time or resetting batches"

        # Create a list that holds batch numbers and shuffle
        global batch_numbers
        batch_numbers = [i for i in xrange(number_of_batches)]
        np.random.shuffle(batch_numbers)

        # A list to know if the batch is used
        global batch__is_used
        batch__is_used = [False] * number_of_batches

    for batch_no in batch_numbers:

        # If the batch_validity[batch_no] = True flag is as used, return it
        if not batch__is_used[batch_no]:

            # Flag it as used
            batch__is_used[batch_no] = True

            # Find the indicies and return batch TODO: FIX here
            return np.asarray(data[batch_no*num:(batch_no + 1)*num][:]), np.asarray(labels[batch_no*num:(batch_no + 1)*num][:])

Xtr, Ytr = np.arange(0, 20), np.arange(0, 400).reshape(20, 20)
print(Xtr)
print(Ytr)
for i in xrange(20):
    Xtr_batch, Ytr_batch = next_batch_2(2, Xtr, Ytr)
    print('\n5 random samples')
    print(Xtr_batch)
    print(Ytr_batch)