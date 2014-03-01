import numpy as np

import matplotlib.pyplot as plt        
        
from multiprocessing import Process

class Observer(Process):
    def __init__(self, exp_name, queue, disp=True):
        self.name = exp_name
        self.queue = queue
        self.disp = disp
        super(Observer, self).__init__()

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
                print 'IM'
                if self.disp:
                    self.display_im(item[1])
            elif item[0] == 'end':
                run = False
            else:
                print item[1]
        plt.ioff()
