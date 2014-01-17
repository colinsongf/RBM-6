import numpy as np

import theano
import theano.tensor as T

from RBM import RBM

class DBN(object):
    def __init__(self, shapes):
        self.layers = []
        
        self.shapes = shapes

        (nv,_) = shapes[0]
        X = T.matrix('X')
        self.inputs = X
        output = self.inputs

        for s in shapes[1:]:
            self.layers.append(RBM(nv, s[0], output, v_unit=s[1]))
            nv = s[0]
            output = self.layers[-1].sample_h_given_v(output)

        self.output = output
    
        self.N = len(self.layers)
        recstr = self.output
        for i in range(1, self.N+1):
            recstr = self.layers[self.N-i].sample_v_given_h(recstr)

        self.recstr = recstr

    def train_dbn(self, X_trn):
        for i in range(self.N):
            print 'training layer {} / {}'.format(i+1, self.N)
            self.train_layer(X_trn, k=i, epochs=30, lr=0.1)

    
    def get_update(self, cost, params, lr=0.1):
        lr = T.cast(lr, dtype=theano.config.floatX)
        updates = []
        for p in params:
            gp = T.grad(cost, p)
            updates += [(p, p- lr*gp)]
        return updates

    def train_layer(self, X_trn, k=0, epochs=1000, batch_s=0.1, lr=0.1/25):
        cost, up = self.layers[k].step(lr)
       
        if type(batch_s) is float:
            batch_s = int(X_trn.shape[0]*batch_s)
        n_batches = X_trn.shape[0]/batch_s
 
        #updates = self.get_update(cost, self.layers[k].params)
        f_trn = theano.function([self.inputs], cost, updates=up)#dates)
        
        for epoch in range(epochs):
            m_cost = []
            for ind in range(n_batches):
                m_cost += [f_trn(X_trn[ind*batch_s: batch_s*(ind+1)])]
            print 'Done epoch {}... cout moyen is {:5}'.format(
                                    epoch, np.mean(m_cost))

    def sample(self, X):
        
        f_sample = theano.function([self.inputs], self.recstr)

        N = X.shape[0]
        X_rec = []
        for i in range(N):
            X_s = f_sample(X[None,i])
            X_rec += [X_s ]
        return X_rec

DEBUG = True

if __name__=='__main__':
    from mnist import *

    images = mnist_read("../data/train-images-idx3-ubyte.gz")
    classes = mnist_read("../data/train-labels-idx1-ubyte.gz")

    model = DBN([(784, None), (1000, None), (500, None), 
                 (250, 'LOG'), (30, None), (10, 'BIN')])

    X_trn = images.reshape((-1,784))/255

    if DEBUG:
        X_trn = X_trn[:1000]

    #model.train_layer(X_trn, epochs=100, batch_s=0.1)
    #model.train_layer(X_trn, k=1, epochs=100)

    model.train_dbn(X_trn)
    test = model.sample(X_trn)

    imshow(test[0].reshape(28,28))
