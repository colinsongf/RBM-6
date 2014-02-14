import numpy as np
import os.path as path

from multiprocessing import Queue

from Observer import Observer

EXP_DIR = '../data/experiences'

class Experience(object):
    def __init__(self, name=None):
        if name is None:
            num = 1
            fname = path.join(EXP_DIR,'exp%i.data'%num)
            while path.exists(fname):
                num += 1
                fname = path.join(EXP_DIR, 'exp%i.data'%num)
            name = 'exp%i'%num
        self.name = name

        fname = path.join(EXP_DIR, '%s.data'%name)
        if not os.path.exists(fname)
            from shutils import copy
            copy(path.join(EXP_DIR, 'default.data'), fname)
        
        import json
        with open(fname) as f:
            param_exp = json.load(f)
        self.fname = fname
        q = Queue()
        self.obs = Observer(name, q)
        self.dbn = DBN(param_exp['lay_shape'], q)
        self.dbn.load(name)

    def save(self):
        import json
        with open(self.fname, 'w') as f:
            json.dump(param_exp, f)

    def pretrain(self, X_trn, epochs=30, lr=0.1):
        self.dbn(X_trn, epochs=epochs, lr=lr)
        for i in range(self.params_exp['N_layer']):
            self.params_exp['epochs_lay'][i]  += 40

    def fine_tune(self, X_trn, X_tst, epochs=30):
        self.dbn.fine_tune(X_trn, X_tst, epochs)



    def save(self):
        import json
        with open(self.fname, 'w') as f:
            json.dump(param_exp, f)



    def save(self):
        import json
        with open(self.fname, 'w') as f:
            json.dump(param_exp, f)


