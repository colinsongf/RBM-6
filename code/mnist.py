import gzip,re,numpy,sys
from numpy import *

def read_int(stream):
    "Read a 32 bit integer from the stream."
    result = 0
    for i in range(4):
        result = (result<<8) + ord(stream.read(1))
    return result

def mnist_read(file,nmax=1000000,verbose=1):
    "Read an MNIST data file and return the contents as a NumPy array."
    if re.search('\.gz$',file):
        stream = gzip.GzipFile(file,"rb")
    else:
        stream = open(file,"rb")
    magic = read_int(stream)
    rank = (magic & 255) - 1
    n = read_int(stream)
    n = min(n,nmax)
    if rank==0:
        data = stream.read(n)
        result = numpy.fromstring(data,"u1",n)
        if verbose:
            sys.stderr.write("read %d scalars from %s\n"%(n,file))
        return array(result,dtype=float)
    elif rank==2:
        w = read_int(stream)
        h = read_int(stream)
        result = []
        for i in range(n):
            data = stream.read(w*h)
            img = numpy.fromstring(data,"u1",w*h)
            img.shape = (w,h)
            result.append(img)
        if verbose:
            sys.stderr.write("read %d %dx%d images from %s\n"%(n,w,h,file))
        return array(result,dtype=float)
