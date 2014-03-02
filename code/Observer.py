import numpy as np

import matplotlib.pyplot as plt        
        
from multiprocessing import Process

RESULT_DIR = '../data/results'

class Observer(Process):
    def __init__(self, exp_name, queue, disp=True):
        self.exp_name = exp_name
        self.queue = queue
        self.disp = disp
        self.cost = []
        self.im = []
        super(Observer, self).__init__()
        self.load()

    def display_im(self, l_im):
        n = len(l_im)
        n_p = l_im[0].shape[0]
        if n==1:
            col = row = int(np.sqrt(n_p))
        elif n==2:
            row = (n_p+1)/2
            col = 4
        else:
            row = n_p
            col = n
        for k in range(n_p):
            for i in range(n):
                plt.subplot(row, col, n*k + i+1)
                plt.imshow(l_im[i][k].reshape((28,28)))
                plt.axis('off')
        plt.subplots_adjust(0,0,1.,1.,0,0)
        plt.draw()

    def run(self):
        plt.ion()
        run = True
        while run:
            item = self.queue.get()
            if item[0] == 'im':
                if self.disp:
                    self.display_im(item[1])
                else:
                    self.im.append(item[1])
            elif item[0] == 'end':
                run = False
            elif item[0] == 'cost':
                self.cost.append(item[1])
                print ('Epoch {}, train_cost {}, val cost '
                       '{}').format(item[1][0], item[1][1], item[1][2])
            else:
                print item[1]
        plt.ioff()
        self.save()

    def save(self):
        from os.path import join as j
        np.save(j(RESULT_DIR, '%s_im.npy'%self.exp_name), self.im)
        np.save(j(RESULT_DIR, '%s_cost.npy'%self.exp_name), self.cost)

    def load(self):
        from os.path import join as j
        from os.path import exists

        if exists(j(RESULT_DIR, '%s_im.npy'%self.exp_name)):
            self.im = list(np.load(j(RESULT_DIR, 
                            '%s_im.npy'%self.exp_name)))
            self.cost = list(np.load(j(RESULT_DIR, 
                            '%s_cost.npy'%self.exp_name)))
