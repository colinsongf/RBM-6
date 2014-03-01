import numpy as np

import theano
import theano.tensor as T

from theano.tensor.shared_randomstreams import RandomStreams


class RBM:
    '''Class to represent a Restricted Bolzmann Machine layer
    implement a training method
    got an input and output theano attribut
    '''
    def __init__(self, n_v, n_h, inputs, vbias=None,
                 hbias=None, initial_W=None, v_unit='BIN',
                 unit_type='LOG'):
        '''
        v_unit: str, optional, default: 'BIN'
        This variable control the output unit of our RBM
        The possible value are ['BIN', 'LOG', 'GAUSS']

        unit_type: str, optional, default:'LOG'
        This variable control the activation function of the unit,
        'LIN' -> W.h +b
        'LOG' -> sig(W.h +b)
        '''
        self.type = unit_type
        
        if initial_W is None:
            initial_W = np.asarray(np.random.uniform(
                            low=-4*np.sqrt(6. / (n_v + n_h)),
                            high=4*np.sqrt(6. / (n_v + n_h)),
                            size=(n_v, n_h)),
                            dtype=theano.config.floatX)
        if hbias is None:
            hbias = theano.shared(value=np.zeros(n_h,
                                dtype=theano.config.floatX), name='hbias')
        if vbias is None:
            vbias = theano.shared(value=np.zeros(n_v,
                                dtype=theano.config.floatX), name='vbias')

        e1 = np.zeros((n_v, n_h), dtype=theano.config.floatX)
        e2 = np.zeros((n_h, n_v), dtype=theano.config.floatX)

        self.inputs = inputs
        self.shape = (n_v, n_h)

        self.W = theano.shared(value=initial_W, name='W')
        self.eps_up = theano.shared(value=e1, name='eps_u')
        self.eps_down = theano.shared(value=e2, name='eps_d')
        self.vbias = vbias
        self.hbias = hbias

        np_rng = np.random.RandomState()
        theano_rng = RandomStreams(np_rng.randint(2**30))

        self.v_type = v_unit
        if v_unit is 'LOG':
            theano_rng.v_unit = self.log_sample
        elif v_unit is 'GAUSS':
            theano_rng.v_unit = self.gauss_sample
        else:
            theano_rng.v_unit = theano_rng.binomial
        self.theano_rng = theano_rng

        self.params = [self.W, self.vbias, self.hbias]
        self.params_ft = [self.eps_up, self.eps_down]

        self.hid = theano.function([self.inputs], self.up(self.inputs))

    def log_sample(self, size, n, p, dtype):
        rv_u = self.theano_rng.uniform(size=size, dtype=dtype)
        return 1./(1+T.exp(-T.log(rv_u/(1-rv_u))+p))

    def gauss_sample(self, size, n, p, dtype):
        return self.theano_rng.normal(size=size, avg=p, std=1./128, dtype=dtype)

    def up(self, vis):
        activ = T.dot(vis, self.W+ self.eps_up) + self.hbias
        if self.type is 'LIN':
            h_mean = activ
        elif self.type is 'ReLU':
            h_mean = T.max(0, activ)
        elif self.type is 'SoftPlus':
            h_mean = T.log(1+T.exp(activ))
        else:
            h_mean = T.nnet.sigmoid(activ)
        return h_mean

    def down(self, hid):
        activ = T.dot(hid, self.W.T + self.eps_down) + self.vbias
        if self.type is 'LIN':
            v_mean = activ
        elif self.type is 'ReLU':
            h_mean = T.max(0, activ)
        elif self.type is 'SoftPlus':
            h_mean = T.log(1+T.exp(activ))
        else:
            v_mean = T.nnet.sigmoid(activ)
        return v_mean

    def sample_h_given_v(self, v_sample):
        h_mean = self.up(v_sample)
        h_sample = self.theano_rng.binomial(size=h_mean.shape, n=1, 
                                            p=h_mean,
                                            dtype=theano.config.floatX)
        return h_sample

    def sample_v_given_h(self, h_sample):
        v_mean = self.down(h_sample)
        v_sample = self.theano_rng.v_unit(size=v_mean.shape, n=1,
                                          p=v_mean, 
                                          dtype=theano.config.floatX)
        return v_sample

    def gibbs_hvh(self, v_sample):
        h_sample = self.sample_h_given_v(v_sample)
        v1_sample = self.sample_v_given_h(h_sample)
        h1_sample = self.sample_h_given_v(v1_sample)

        return [h_sample, v1_sample, h1_sample]

    def free_energy(self, v):
        wx_b = T.dot(v, self.W) + self.hbias
        bias = T.dot(v, self.vbias)
        hidden = T.sum(T.log(1+T.exp(wx_b)),axis=1)
        return -hidden -bias
        

    def step(self, lr):
        vs = self.inputs
        [hs, v1s, h1s] = self.gibbs_hvh(vs)
        
        cost = T.mean(self.free_energy(vs)) \
             - T.mean(self.free_energy(v1s))
        lr_T = T.cast(lr, dtype=theano.config.floatX)

        updates = []
        gW = T.grad(cost, self.W, consider_constant=[v1s])
        updates.append((self.W, self.W-lr_T*gW))
        ghb = T.grad(cost, self.hbias, consider_constant=[v1s])
        updates.append((self.hbias, self.hbias-lr_T*ghb))
        gvb = T.grad(cost, self.vbias, consider_constant=[v1s]) 
        updates.append((self.vbias, self.vbias-lr_T*gvb))

        return [cost, updates]


if __name__=='__main__':
    X = T.matrix('x')
    model = RBM(100,4,X, v_unit='LOG')

    X_trn = np.random.normal(size=(25,100))

    cost, updates = model.step()
    step = theano.function([X], cost, updates=updates)

    m_cost=[]
    for i in range(100):
        m_cost += [step(X_trn)]
        print m_cost[-1]

    import matplotlib.pyplot as plt
    plt.figure(1)
    plt.plot(m_cost)
    plt.show()
