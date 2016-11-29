import numpy as np

def batch_stream(arrays, batch_size=8):
    size = arrays[0].shape[0]
    n_batches = size / batch_size

    indx = np.random.permutation(size)

    for batch in xrange(n_batches):
        from_i = batch * batch_size
        to_i = from_i + batch_size

        batch_indx = indx[from_i:to_i]

        yield [ arr[batch_indx] for arr in arrays ]

def onehot(y):
    n_classes = np.max(y) + 1
    
    encoded = np.zeros(shape=(y.shape[0], n_classes), dtype='float32')
    encoded[np.arange(y.shape[0]), y] = 1
    
    return encoded