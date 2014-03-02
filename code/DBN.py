import numpy as np

import theano
import theano.tensor as T
from theano.tensor.shared_randomstreams import RandomStreams

from RBM import RBM

PARAMS = '../data/params'

class DBN(object):
    def __init__(self, shapes, queue, noise=None):

        # Communication queue for the log
        self.queue = queue

        # Semantic variables for the input, corruption level
        # and learning rate
        X = T.matrix('X')
        self.inputs = X
        cl = T.scalar(dtype=theano.config.floatX,
                    name='corruption level')
        self.cl = cl
        lr = T.scalar(dtype=theano.config.floatX,
                    name='learning rate')
        self.lr = lr
 
        # Random number generators used for the noise
        np_rng = np.random.RandomState()
        theano_rng = RandomStreams(np_rng.randint(2**30))

        # Layers initialisation, cast the shape
        # and fill the layers list.
        self.layers = []
        
        self.shapes = shapes
        (nv,_,_) = shapes[0]

        self.params = []
        self.params_ft = []
        
        output = X
        sample_up = X

        # Compute the droupout training function
        p_do = 0.5
        self.p_do = p_do
        dropout_out =X
        rec_do = X

        # Build the layers,  linking each one to the next
        # Fill the param list
        for i, s in enumerate(shapes[1:]):
            lay = RBM(nv, s[0], output, v_unit=s[2],
                                   unit_type=s[1])
            self.layers.append(lay)

            self.params += lay.params
            self.params_ft += lay.params_ft
            nv = s[0]

            output = lay.up(output)
            sample_up = lay.sample_h_given_v(sample_up)
            if i != 0:
                dropout_out *= theano_rng.binomial(size=dropout_out.shape,
                                        n=1, p=p_do)
                rec_do *= p_do
            dropout_out = lay.up(dropout_out)
            rec_do = lay.up(rec_do)
            

        # Define the up functions
        self.code = output
        self.sample_up = sample_up
    
        # Prepare the variables to decode
        self.N = len(self.layers)
        recstr = output
        decode = X
        sample_down = X
        sample = sample_up

        # Add noise to the output for the fine tuning part
        self.noise = noise
        if self.noise == 'MASK':
            fine_tune = T.clip(output * \
                    theano_rng.binomial(
                        size=output.shape,
                        n=1, p=1-cl),0.,1.)
        elif self.noise == 'GAUSS':
            fine_tune = T.clip(output +\
                    theano_rng.normal(
                        size=output.shape,
                        std=cl), 0.,1.) 
        else:
            fine_tune = output

        # Down sample every variable
        for i in range(1, self.N+1):
            lay = self.layers[self.N-i]
            recstr = lay.down(recstr)
            decode = lay.down(decode)
            fine_tune = lay.down(fine_tune)
            sample_dowm = lay.sample_v_given_h(sample_down)
            sample = lay.sample_v_given_h(sample)
            if i!= self.N:
                rec_do *= p_do
                dropout_out *= theano_rng.binomial(size=dropout_out.shape,
                                    n=1, p=p_do)
            dropout_out = lay.down(dropout_out)
            rec_do = lay.down(rec_do)

        #define the sampeling and decoding functions
        self.recstr = recstr
        self.decode = decode
        self.ft = fine_tune
        self.do = dropout_out
        self.sample_down = sample_down
        self.sample = sample

        self.compile()
    
    def pretrain(self, X_trn, epochs, lr, batch_s=0.1):
        X = X_trn
        
        self.batch_s = batch_s
        if type(batch_s) is float:
            self.batch_s = int(X_trn.shape[0]*batch_s)
        self.n_batches = X.shape[0]/self.batch_s

        for i in range(self.N):
            print 'training layer {} / {}'.format(i+1, self.N)
            self.train_layer(i, X, epochs=epochs, lr=lr)
            X = self.layers[i].hid(X)


    def fine_tune(self, data, epochs, lr, cl,
                 lcost=False, dropout=False):
        if dropout:
            out = self.do
            for l in self.layers[1:]:
                l.W.set_value(l.W.get_value()/self.p_do)
        else:
            out = self.ft

        if not lcost:
            cost = T.mean(T.sum((out-self.inputs)**2, axis=1))
            cost_val = T.mean(T.sum((self.recstr-self.inputs)**2,axis=1))
        else:
            cost = T.mean(T.sum(-self.inputs*T.log(out) -
                         (1-self.inputs)*T.log(1-out), axis=1))
            cost_val = T.mean(T.sum(-self.inputs*T.log(self.recstr)-
                        (1-self.inputs)*T.log(1-self.recstr), axis=1))
        params = []
        updates = self.get_update(cost, self.params_ft, self.lr)
        
        f_train = theano.function(inputs=[self.inputs, self.cl, self.lr], 
                                outputs=cost, 
                                updates=updates,
                                on_unused_input='ignore')
        f_cost = theano.function(inputs=[self.inputs], outputs=cost_val)
        n_batches = data.n_batches
        batch_s = data.batch_s
        

        X_val = data.get_valid_set()
        for epoch in range(epochs):
            m_cost = f_train(data.get_batch(), 
                           cl(epoch), lr(epoch))
            if epoch % 10 == 0 or epoch == epochs-1:
                val_cost = f_cost(X_val)
                recstr = self.f_recstr(X_val[:10])
                self.queue.put(('im', [X_val[:10], recstr])) 
                self.queue.put(('cost', (epoch, m_cost, val_cost)))
        if dropout:
            for l in self.layers[1:]:
                l.W.set_value(l.W.get_value()*self.p_do)
    
    def get_update(self, cost, params, lr):
        updates = []
        for p in params:
            gp = T.grad(cost, p)
            updates += [(p, p- lr*gp)]
        return updates

    def train_layer(self, k, X_trn, epochs, lr):
        '''Train the layer k for a certain number of epochs on the 
        X_trn dataset.
        '''

        #Compute the cost function and the updates
        cost, up = self.layers[k].step(self.lr)
        f_trn = theano.function([self.layers[k].inputs, self.lr], cost, updates=up)

        # Run the RBM for a certain number of epochs
        # with a batch script
        n_batches = self.n_batches
        batch_s = self.batch_s
        for epoch in range(epochs):
            m_cost = []
            for ind in range(n_batches):
                m_cost += [f_trn(X_trn[ind*batch_s: batch_s*(ind+1)], lr(epoch))]
            m_cost = np.mean(m_cost)
            print 'Done epoch {}... cout moyen is {:5}'.format(
                                    epoch, m_cost)

    def compile(self):
        '''Compile the different fuctions that we can use to 
        code, decode and sample from our model

        '''
        X = self.inputs
        self.f_code = theano.function(inputs=[X],
                            outputs=self.code)
        self.f_decode = theano.function(inputs=[X],
                            outputs=self.decode)
        self.f_sample = theano.function(inputs=[X],
                            outputs=self.sample)
        self.f_sample_up = theano.function(inputs=[X],
                            outputs=self.sample_up)
        self.f_sample_down = theano.function(inputs=[X],
                            outputs=self.sample_down)
        self.f_recstr = theano.function(inputs=[X],
                            outputs=self.recstr)

    def load(self, fname=None):
        import os
        if fname is None:
            num=1
            l_files = os.listdir(PARAMS)
            while 'exp%i_params.npy'%num in l_files:
                num += 1
            num -=1
            fname = 'exp%i'%num
        from os.path import join as j
        if not os.path.exists(j(PARAMS, '%s_params.npy'%fname)):
            ename = ''.join(c for c in fname if c.isalpha())
            fname = '%s_pretrain'%ename
            if not os.path.exists(j(PARAMS, '%s_params.npy'%fname)):
                return False

        params = np.load(j(PARAMS, '%s_params.npy'%fname))
        params_ft = np.load(j(PARAMS, '%s_params_ft.npy'%fname))
        for k in range(self.N):
            self.layers[k].W.set_value(params[3*k].get_value())
            self.layers[k].vbias.set_value(params[3*k+1].get_value())
            self.layers[k].hbias.set_value(params[3*k+2].get_value())

            self.layers[k].eps_up.set_value(params_ft[2*k].get_value())
            self.layers[k].eps_down.set_value(params_ft[2*k+1].get_value())
        return True
            
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


