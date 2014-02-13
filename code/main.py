import numpy as np

from DBN import DBN

from mnist import mnist_read



if __name__=='__main__':
    import argparse

    parser = argparse.ArgumentParser(description='')
    parser.add_argument('-d', dest='debug', action='store_true', 
                    help='Use a small number of training example')
    args = parser.parse_args()
    
    N = 10000
    N_tst = 10
    max_epochs = 100
    if args.debug:
        N = 1500
        max_epochs=10

    X = mnist_read('../data/train-images-idx3-ubyte.gz', nmax=N+N_tst)

    model = DBN([(784, None, ''), (1000, None, 'GAUSS'), (500, None, ''),
                (250, 'LOG', ''), (30, None, '')])

    X_trn = X[:N].reshape((-1,784))/255.
    X_tst = X[N:].reshape((-1,784))/255.

    model.pretrain(X_trn, epochs=max_epochs, lr=0.05)
    rec = model.sample(X_tst)
    print 'Pretraining done\n'

    model.fine_tune(X_trn, X_tst,epochs=max_epochs)
    test = model.sample(X_tst)

    from utils.Observer import Observer
    obs = Observer('test')
    disp = []
    disp.append(X_trn[:6])
    disp.append(rec[:6])
    disp.append(test[:6])
    obs.display_im(disp)

