import h5py

def batch(iterable, n=1):
    l = len(iterable)
    for ndx in range(0, l, n):
        yield iterable[ndx:min(ndx + n, l)]

def get_training_data(filename):
    f = h5py.File(filename)
    dset_list = {}
    for dsetname, dset in f.iteritems():
        if dsetname == 'Xtr' or dsetname == 'Str_gt' or dsetname == 'Ltr_oh':
            dset_list[dsetname] = dset[:]
    return dset_list


dataset = get_training_data('gbu_CUB_data.mat')

print dataset

def getBAATCHED(key):
    for x in batch(dataset[key], 2):
        print x
        print "batch end"
