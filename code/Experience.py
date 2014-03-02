import numpy as np
import os.path as path

from multiprocessing import Queue

from Observer import Observer
from DBN import DBN

EXP_DIR = '../data/experiences'

class Experience(object):
    def __init__(self, N, name=None, disp=False, noise=None):
        # If no name, we create an experience
        if name is None:
            num = 1
            fname = path.join(EXP_DIR,'exp%i.data'%num)
            while path.exists(fname):
                num += 1
                fname = path.join(EXP_DIR, 'exp%i.data'%num)
            name = 'exp%i'%num
        self.name = name

        # If there exist no previous file,
        # We create one from the default experience
        fname = path.join(EXP_DIR, '%s.data'%name)
        if not path.exists(fname):
            e_name = ''.join(c for c in name if c.isalpha())
            e_name = path.join(EXP_DIR, '%s.data'%e_name)
            from shutil import copy
            if not path.exists(e_name):
                copy(path.join(EXP_DIR, 'default.data'), fname)
            else:
                copy(e_name, fname)
        
        #We load the experience parameters
        import json
        with open(fname) as f:
            params_exp = json.load(f)
        self.fname = fname
        self.params_exp = params_exp

        # Create the instance of the experience
        self.queue = Queue()
        self.obs = lambda: Observer(name, self.queue, disp)
        self.dbn = DBN(params_exp['lay_shape'], self.queue, noise)

        #Load the model weights if it exists
        self.exists = self.dbn.load(name)

        #Load the dataset and split between train and test set
        from DataFeeder import DataFeeder
        self.data = DataFeeder(N, batch_s=1000)

    def pretrain(self, epochs=30, lr=0.1):
        l_r = lr
        if type(l_r) == float:
            l_r = lambda e: lr
        self.dbn.pretrain(self.data.X_trn, epochs=epochs, lr=l_r)
        for i in range(self.params_exp['N_layer']):
            self.params_exp['epochs_lay'][i]  += epochs

    def fine_tune(self, epochs=30, lr=.1, cl=.1, lcost=False ,dropout=False):
        l_r = lr
        c_l = cl
        if type(l_r) == float:
            l_r = lambda e: lr
        if type(c_l) == float:
            c_l= lambda e: cl
        p = self.obs()
        p.start()
        self.dbn.fine_tune(self.data, epochs=epochs, lr=l_r, 
                           cl=c_l, lcost=lcost, dropout=dropout)
        self.params_exp['epochs_ft'] += epochs
        self.queue.put(('end',))
        p.join()

    def eval_perf(self):
        X_tst, y_tst = self.data.get_test_set()
        code = self.dbn.f_code(X_tst)

        from sklearn import metrics
        sil_c = metrics.silhouette_score(code, y_tst)
        sil_X = metrics.silhouette_score(X_tst, y_tst)
        
        print 'Silhouette code y', sil_c
        print 'Silhouette X y', sil_X


    def save(self):
        '''Save the experience so we can reload it later
        '''

        #Save the experience parameters
        import json
        with open(self.fname, 'w') as f:
            json.dump(self.params_exp, f)
        
        # Save the model weights
        self.dbn.save(self.name)
        
