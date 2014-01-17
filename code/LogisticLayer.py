import numpy as np

import theano
import theano.T as tensor

class LogisticLayer(object):
    def __init__(self, n_in, n_out, inputs=None):
        self.shape = (n_in, n_out)

        if inputs is None:
            inputs = T.matrix('X_log')

        self.inputs = inputs
        self.W = theano.shared(value=np.zeros(self.shape,
                              dtype=theano.config.floatX),
                              name = 'log_W')
        self.b = theano.shared(value=np.zeros(n_out,
                              dtype=theano.config.floatX),
                              name='log_b')
        self.activ = T.dot(inputs, self.W) + self.b
        self.output = 1/(1+T.exp(activ))

    def cost(self, y, lr=0.1):
       cost = self.output
