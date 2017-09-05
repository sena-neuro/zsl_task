import h5py
import numpy as np
import os
import tensorflow as tf
import math

# Global variables for creating batches
batch_numbers = []
batch_is_used = []


def get_data(filename):
    f = h5py.File(filename,"r")
    dset_list = {}
    for dsetname, dset in f.iteritems():
        dset_list[dsetname] = dset.value.astype(np.float32).T
    return dset_list


def next_batch(num, data, labels):

    # For first creation of batch and resetting batches
    if not False in batch_is_used:

        # Finds how many batches possible
        number_of_batches = int(math.ceil(float(len(data)) / num))

        # Create a list that holds batch numbers and shuffle
        global batch_numbers
        batch_numbers = [i for i in xrange(number_of_batches)]
        np.random.shuffle(batch_numbers)

        # A list to know if the batch is used
        global batch_is_used
        batch_is_used = [False] * number_of_batches

    for batch_no in batch_numbers:

        # If the batch_validity[batch_no] = True flag is as used, return it
        if not batch_is_used[batch_no]:

            # Flag it as used
            batch_is_used[batch_no] = True

            # Find the indicies and return batch
            return np.asarray(data[batch_no*num:(batch_no + 1)*num][:]), np.asarray(labels[batch_no*num:(batch_no + 1)*num][:])
