import numpy as np

import theano
import theano.tensor as T
from theano.tensor.shared_randomstreams import RandomStreams

from RBM import RBM

PARAMS = '../data/params'

class DBN(object):
    def __init__(self, shapes, queue):
        self.queue = queue
        
        self.layers = []
        self.shapes = shapes

        (nv,_,_) = shapes[0]
        X = T.matrix('X')
        self.inputs = X
        self.params = []
        self.params_ft = []
        output = X

        np_rng = np.random.RandomState(1234)
        theano_rng = RandomStreams(np_rng.randint(2**30))

        for s in shapes[1:]:
            self.layers.append(RBM(nv, s[0], output, v_unit=s[2],
                                   unit_type=s[1]))
            nv = s[0]
            output = self.layers[-1].up(output)
            self.params += self.layers[-1].params
            self.params_ft += self.layers[-1].params_ft

        self.code = output
    
        self.N = len(self.layers)
        recstr = output
        cl = T.scalar(dtype=theano.config.floatX,
                       name= 'corruption level')
        fine_tune = T.clip(output * \
                    theano_rng.binomial(size=output.shape,
                                        n=1, p=1-cl),
                    #theano_rng.normal(size=output.shape,
                    #                  std=cl),
                    0.,1.) 
        decode = X
        for i in range(1, self.N+1):
            recstr = self.layers[self.N-i].down(recstr)
            decode = self.layers[self.N-i].down(decode)
            fine_tune = self.layers[self.N-i].down(fine_tune)
        self.recstr = recstr
        self.decode = decode
        self.cl = cl
        self.ft = fine_tune
        self.sample = theano.function([self.inputs], self.recstr)
        self.compile()
    
    def pretrain(self, X_trn, epochs=30, lr=0.1, batch_s=0.1):
        X = X_trn

        if type(batch_s) is float:
            self.batch_s = int(X_trn.shape[0]*batch_s)
        self.n_batches = X.shape[0]/self.batch_s

        for i in range(self.N):
            print 'training layer {} / {}'.format(i+1, self.N)
            self.train_layer(X, epochs=epochs, lr=lr, k=i)
            X = self.layers[i].hid(X)


    def fine_tune(self, X_trn, X_tst, epochs=30, lr=0.1):
        cost = T.sum((self.ft-self.inputs)**2)/self.batch_s
        params =[]
        l_r = T.scalar('lr', dtype=theano.config.floatX)
        updates = self.get_update(cost, self.params_ft, l_r)
        
        f_cost = theano.function(inputs=[self.inputs, self.cl, l_r], 
                                outputs=cost, 
                                updates=updates)
        n_batches = self.n_batches
        batch_s = self.batch_s

        cl = 0.05
        for epoch in range(epochs):
            m_cost = []
            for ind in range(n_batches):
                m_cost += [f_cost(X_trn[ind*batch_s: (ind+1)*batch_s], 
                            cl, lr)]
            if epoch % 10 == 0:
                cl = min(0.25,cl+ 0.05)
            m_cost = np.mean(m_cost)
            self.anim(X_tst)
            print 'Done epoch {}... cout moyen is {:5}'.format(
                                    epoch, m_cost)
    
    def get_update(self, cost, params, lr=0.1):
        lr = T.cast(lr, dtype=theano.config.floatX)
        updates = []
        for p in params:
            gp = T.grad(cost, p)
            updates += [(p, p- lr*gp)]
        return updates

    def train_layer(self, X_trn, epochs=10, lr=0.1, k=0):
        learning_rate = T.scalar('lr', dtype=theano.config.floatX)
        cost, up = self.layers[k].step(learning_rate)

        n_batches = self.n_batches
        batch_s = self.batch_s

        f_trn = theano.function([self.layers[k].inputs, learning_rate], cost, updates=up)#dates)
        prev = []
        for epoch in range(epochs):
            m_cost = []
            for ind in range(n_batches):
                m_cost += [f_trn(X_trn[ind*batch_s: batch_s*(ind+1)], lr)]
            m_cost = np.mean(m_cost)
            print 'Done epoch {}... cout moyen is {:5}'.format(
                                    epoch, m_cost)
            if False & epoch != 0:
                if m_cost > prev[-1]:
                    lr *= max(lr*0.7, 0.005)
                    print 'reduce the learning rate', lr
                elif m_cost/(prev[-1]+(prev[-1] == 0)) > 0.995:
                    lr *= max(lr*1.1, 0.5)
                    print 'raise the learning rate', lr
            prev.append(m_cost)

    def compile(self):
        self.f_code = theano.function(inputs=[self.inputs],
                                      outputs=self.code)
        self.f_decode = theano.function(inputs=[self.inputs],
                                        outputs=self.decode)

    def load(self, fname=None):
        import os
        if fname is None:
            num=1
            l_files = os.listdir(PARAMS)
            while 'exp%i_params.npy'%num in l_files:
                num += 1
            num -=1
            fname = 'exp%i'%num
        if not os.path.exists(j(PARAMS, '%s_params.npy'%fname)):
            return
        from os.path import join as j
        params = np.load(j(PARAMS, '%s_params.npy'%fname))
        params_ft = np.load(j(PARAMS, '%s_params_ft.npy'%fname))
        for k in range(self.N):
            self.layers[k].W.set_value(params[3*k].get_value())
            self.layers[k].vbias.set_value(params[3*k+1].get_value())
            self.layers[k].hbias.set_value(params[3*k+2].get_value())

            self.layers[k].eps_up.set_value(params_ft[2*k].get_value())
            self.layers[k].eps_down.set_value(params_ft[2*k+1].get_value())
            
    def save(self, fname=None):
        if fname is None:
            import os
            l_files = os.listdir(PARAMS)
            num = 1
            while 'exp%i_params.npy'%num in l_files:
                num +=1
            fname = 'exp%i'%num
        from os.path import join as j
        np.save(j(PARAMS, '%s_params.npy'%fname), self.params)
        np.save(j(PARAMS, '%s_params_ft.npy'%fname), self.params_ft)

DEBUG = True 

if __name__=='__main__':
    from mnist import *

    images = mnist_read("../data/train-images-idx3-ubyte.gz", 100)
    classes = mnist_read("../data/train-labels-idx1-ubyte.gz",100)

    model = DBN([(784, None, ''), (1000, None, 'GAUSS'), (500, None, ''), 
                 (250, 'LOG', ''), (30, None, '')])#, (10, 'LOG', '')])

    X_trn = images.reshape((-1,784))/255
    if DEBUG:
        X_trn = X_trn[:1000]

    model.pretrain(X_trn, epochs=50, lr=0.05)
    rec = model.sample(X_trn)

    print 'Pretraining done\n'

    model.fine_tune(X_trn)
    test = model.sample(X_trn)

    import matplotlib.pyplot as plt
    plt.imshow(test[0].reshape(28,28))

    off = 0
    for k in range(6):
        plt.subplot(6, 3, 3*k+1)
        plt.imshow(X_trn[off+k].reshape(28,28))
        plt.axis('off')
        plt.subplot(6, 3, 3*k+2)
        plt.imshow(rec[off+k].reshape(28,28))
        plt.axis('off')
        plt.subplot(6, 3, 3*k+3)
        plt.imshow(test[off+k].reshape(28,28))
        plt.axis('off')

    plt.subplots_adjust(0,0,1.,1.,0,0)
    plt.show()


