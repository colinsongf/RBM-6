import numpy as np

class DataFeeder(object):
    def __init__(self, N, N_tst = 1000, N_val=1000, batch_s=0.1):
        from mnist import mnist_read
        X_trn = mnist_read('../data/train-images-idx3-ubyte.gz', N+N_val)
        X_tst = mnist_read('../data/t10k-images-idx3-ubyte.gz', N_tst)
        y_tst = mnist_read('../data/t10k-labels-idx1-ubyte.gz',N_tst)
        self.X_trn = np.asarray(X_trn[:-N_val]).reshape((-1,784))/255.
        self.X_val = np.asarray(X_trn[-N_val:]).reshape((-1,784))/255.
        self.X_tst = np.asarray(X_tst).reshape((-1,784))/255.
        self.y_tst = np.asarray(y_tst)

        self.batch_s = batch_s
        if type(batch_s) is float:
            self.batch_s = int(N*batch_s)
        self.n_batches = N/self.batch_s
        self.i = 0

    def get_batch(self):
        ind = self.i
        batch = self.X_trn[ind*self.batch_s:(ind+1)*self.batch_s]
        self.i +=1
        if self.i >= self.n_batches:
            self.i = 0
        return batch

    def get_test_set(self):
        return (self.X_tst, self.y_tst)

    def get_valid_set(self):
        return self.X_val
