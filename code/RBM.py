import numpy as np

import theano
import theano.tensor as T


class RBM:
    '''Class to represent a Restricted Bolzmann Machine layer
    implement a training method
    got an input and output theano attribut
    '''
    def __init__(self, n_v, n_h, inputs, vbias=None,
                 hbias=None, initial_W):
        
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

        self.input = inputs
        self.layer = (n_v, n_h)

        self.W = theano.shared(value=initial_W, name='W')
        self.vbias = vbias
        self.hbias = hbias

        np_rng = np.random.RandomState(1234)
        theano_rng = RandomStreams(np_rng.randint(2**30))

        self.theano_rng = theano_rng

        self.params = [self.W, self.vbias, self.hbias]

    def sample_h_given_v(self, v_sample):
        activ = T.dot(v_sample, self.W) + self.hbias
        h_mean = T.nnet.sigmoid(activ)
        h_sample = self.theano_rng.binomial(size=h_mean.shape, n=1, p=h_mean,
                                            dtype=theano.config.floatX)
        return [activ, h_mean, h_sample]

    def prop
