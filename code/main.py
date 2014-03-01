import numpy as np

from Experience import Experience

if __name__=='__main__':
    import argparse

    parser = argparse.ArgumentParser(description='')
    parser.add_argument('-d', dest='debug', action='store_true', 
                    help='Use a small number of training example')
    parser.add_argument('-n', dest='name', metavar='NAME', default=None,
                    type=str, help='Name of the experiment')
    parser.add_argument('--img', action='store_true',
                    help='Display the reconstruction while training')
    parser.add_argument('--lcost', action='store_true', help=('Fine tune loss function'
                    ', default is MSE, else cross entropy')
    parser.add_argument('--noise', type=str, default=None,
                    help=('Denoising autoencoder, use GAUSS or MASK to get'
                          'representation robust to noise'))
    parser.add_argument('--dropout', action='store_true',
                    help='Enable dropout training')
    parser.add_argument('--epochs', type=int, default=30,
                    help='Number of epochs for fine tune')
    parser.add_argument('--training', type=str, default='10000',
                    help='Number of training exemple used')
    args = parser.parse_args()
    try:
        N = int(args.training)
    except ValueError:
        if N == 'full':
            N = 60000
        else:
            print 'Couldn\'t cast the number of training. Used 10000'
            N = 10000
    max_epochs = 10
    max_epochs_ft = args.epochs
    if args.debug:
        N = N / 10
        max_epochs = 10
        max_epochs_ft = max(10, max_epochs_ft)

    model = Experience(N, name=args.name, disp=args.img, 
                          noise=args.noise)
    if not model.exists:
        model.pretrain(epochs=max_epochs, lr=0.1)
        model.save()
    print 'Pretraining done\n'

    model.fine_tune(epochs=max_epochs_ft, lr=0.05,
                    dropout=args.dropout, lcost=args.lcost)
    
    X_tst = model.X_tst
    X_rcstr = model.dbn.f_recstr(X_tst)
    model.obs.display_im([X_tst[:10], X_recstr[:10]])

    model.eval_perf()
