#!/usr/bin/env python
import numpy as np
from math import tanh, pi
from scipy.interpolate import interp1d
from scipy.signal import fftconvolve
from PIL import Image
from os import listdir
from os.path import isfile, join
import array as ar
from matplotlib import pyplot
import brewer2mpl
import time

# my packages for estimating entropy
#from info_theory_functions import *

# path to van Hateren Natural Image Database
#im_path = '/Users/lmcintosh/Dropbox/Stanford University/CS 229 Machine Learning/Project/Natural_Images'
#im_path = '/home/lane/Natural_images/imc'


##### FUNCTIONS #####
# define the nonlinearity
def nonlinearity(x,K=1.,g=1.,theta=0.,mode='sigmoid',normalized=False):
    if normalized:
        if mode == 'sigmoid':
            n = K/(1+np.exp(-g*(x - theta)))
        elif mode == 'tanh':
            if type(x)==list or type(x)==numpy.ndarray:
                n = [g*tanh(a-theta) + K - g for a in x]
            else:
                n = g*tanh(x-theta) + K - g
        return np.array(n)/np.sqrt(np.var(np.array(n)))
    else:
        if mode == 'sigmoid':
            return K/(1+np.exp(-g*(x - theta)))
        elif mode == 'tanh':
            if type(x)==list or type(x)==numpy.ndarray:
                return [g*tanh(a-theta) + K - g for a in x]
            else:
                return g*tanh(x-theta) + K - g

# return the shape of a Gaussian
def gaussian(x=np.linspace(-5,5,50),sigma=1.,mu=0.):
     return np.array([(1./(2.*pi*sigma**2))*np.exp((-(xi-mu)**2.)/(2.*sigma**2)) for xi in x])

# return a 2d difference of Gaussians with zero mean and unit variance
def spatial_filter_2d(c_width=1.,s_width=2.,xs_num=50, alpha=1., beta=1.):
    xs = np.linspace(-5,5,xs_num)
    S = gaussian(x=xs, sigma=s_width)
    C = gaussian(x=xs, sigma=c_width)
    S = S/np.sum(S)
    C = C/np.sum(C)
    S_2d = zeros((len(S),len(S)))
    C_2d = zeros((len(C),len(C)))
    for idx,x in enumerate(xs):
        for idy,y in enumerate(xs):
            S_2d[idx,idy] = S[np.min([int(np.sqrt((idx-np.floor(len(xs)/2.))**2 
                                                     + (idy-np.floor(len(xs)/2.))**2) + np.floor(len(xs)/2.)), len(xs)-1])]
            C_2d[idx,idy] = C[np.min([int(np.sqrt((idx-np.floor(len(xs)/2.))**2 
                                                     + (idy-np.floor(len(xs)/2.))**2) + np.floor(len(xs)/2.)), len(xs)-1])]
    # Make all surrounds have same peak sensitivity of 1.
    S_2d = S_2d/np.max(S_2d)
    # Make center have same integral as surround
    C_2d = (np.sum(S_2d)/np.sum(C_2d))*C_2d
    X = alpha*C_2d - beta*S_2d
    
    return X/np.sqrt(np.var(X))

# plot information
def print_matrix(M, min_x, max_x, min_y, max_y, top_title, x_label='size of filter 2', y_label='size of filter 1', path='', extraInfo=''):
    fig = pyplot.gcf()
    fig.set_size_inches((16,10))
    bmap = brewer2mpl.get_map('RdYlBu','diverging',8,reverse=True).mpl_colormap
    
    pyplot.imshow(M[::-1,:],interpolation='nearest',cmap=bmap, extent=[min_x,max_x,min_y,max_y])
    pyplot.colorbar()
    pyplot.ylabel(y_label,fontsize=14)
    pyplot.xlabel(x_label,fontsize=14)
    pyplot.title(top_title,fontsize=16)
    pyplot.savefig(path + time.strftime("%Y-%m-%d %H-%M-%S") + ' ' + top_title + '. ' + extraInfo + '.png')
    time.sleep(1)
    fig.clf()


# return a linear prediction given the filter and an image
def linear_output(stimulus,sfilter,numSubunits,borderDist=0):
    '''stimulus is space x space
       sfilter is space x space
       numSubunits is number of LN subunits across a single dimension
    '''
    output = fftconvolve(stimulus, sfilter, mode='same')
    return subsample(output, (numSubunits, numSubunits), padding=borderDist)



# return the output of the full LNL model
def lnl_output(stimulus,filter1,filter2=0.,nonlinearThreshold=1.,nonlinearGain=0.25,numSubunits=16,SNR=1000.,signalVariance=0.,key=3):
    ''' key = 0 for L
        key = 1 for L-N
        key = 2 for L-N + L
        key = 3 for L-N-L
    '''
    # NO NEED TO NORMALIZE STIMULUS (happens in load_images())

    # JUST LINEAR FILTER, NO NOISE
    if key == 0:
        return linear_output(stimulus,filter1,1,borderDist=stimulus.shape[0]//2)
    # SINGLE LN SUBUNIT, NO NOISE
    elif key == 1:
        ls = linear_output(stimulus,filter1,1,borderDist=stimulus.shape[0]//2)
        return nonlinearity(ls,g=nonlinearGain,theta=nonlinearThreshold)
    # SINGLE LN SUBUNIT COMBINED LINEARLY WITH A PARALLEL LINEAR STAGE
    elif key == 2:
        ls = linear_output(stimulus,filter1,1,borderDist=stimulus.shape[0]//2)
        ns = nonlinearity(ls,g=nonlinearGain,theta=nonlinearThreshold)
        return ns + linear_output(stimulus,filter2,1,borderDist=stimulus.shape[0]//2)
    # FULL LNL MODEL WITH OR WITHOUT NOISE
    elif key == 3:
        ls = linear_output(stimulus,filter1,numSubunits,borderDist=int(stimulus.shape[0]//10)) # borderDist was 0
        ns = np.array([nonlinearity(l,g=nonlinearGain,theta=nonlinearThreshold) for l in ls]).reshape((numSubunits,numSubunits))
        if SNR == 'Inf':
            return linear_output(ns,filter2,1,borderDist=numSubunits//2)
        else:
            return linear_output(ns,filter2,1,borderDist=numSubunits//2) + np.sqrt(signalVariance/SNR)*randn(1)
    
    if key < 0 or key > 3:
        raise Exception('Key not recognized.  Must be 0-3')
   

def get_threshold(stimuli,filter1):
    return np.sqrt(np.var(np.array([lnl_output(s,filter1,key=0) for s in stimuli]).flat))

def get_signal_variance(stimuli,filter1,filter2,threshold,subunits=16,key=3):
    return np.var([lnl_output(s,filter1,filter2,nonlinearThreshold=threshold,numSubunits=subunits,SNR='Inf',key=key) for s in stimuli])


def readCommand(argv):
  """
  Processes the command used to run lnl_model from the command line.
  """
  from optparse import OptionParser
  usageStr = """
  USAGE:      python lnl_model.py <options>
  EXAMPLES:   (1) python lnl_model.py
                  - starts lnl_model with default parameters
              (2) python pacman.py --numImages 50 --patchSize 64 --key 3
              OR  python pacman.py -n 50 -p 64 -k 3 
  """
  parser = OptionParser(usageStr)

  parser.add_option('-n', '--numImages', dest='numImages', type='int',
                    help='the number of images to use for information computation',
                    metavar='IMAGES', default=20)
  parser.add_option('-p', '--patchSize', dest='patchSize', type='int',
                    help='the patches will be (patchSize,patchSize) pixels',
                    metavar='PATCHES', default=32)
  parser.add_option('-i', '--iterations', dest='iterations', type='int',
                    help='number of iterations in information computation', default=80)
  parser.add_option('-k', '--key', dest='key', type='int',
                    help='int in [0,3] for model type', default=3)
  parser.add_option('-o', '--numPhotoreceptors', dest='numPhotoreceptors', type='int',
                    help='number of photoreceptors per subunit', default=16)
  parser.add_option('-s', '--numSubunits', type='int', dest='numSubunits',
                    help='number of LN subunits', default=16)
  parser.add_option('-g', '--gain', type='float', dest='nonlinearGain',
                    help='gain of sigmoid', default=0.25)
  parser.add_option('-r', '--random', action='store_true', dest='fixRandomSeed', 
                    help='fix random seed', default=False)
  parser.add_option('-e', '--snr', type='float', dest='snr', help='output noise SNR',
                    default=1000.)

  options, otherjunk = parser.parse_args(argv)
  if len(otherjunk) != 0:
    raise Exception('Command line input not understood: ' + str(otherjunk))
  args = dict()

  # Fix the random seed
  if options.fixRandomSeed: random.seed('cs188')

  args['numImages'] = options.numImages
  args['patchSize'] = options.patchSize
  args['iterations'] = options.iterations
  args['key'] = options.key
  args['numPhotoreceptors'] = options.numPhotoreceptors
  args['numSubunits'] = options.numSubunits
  args['nonlinearGain'] = options.nonlinearGain
  args['snr'] = options.snr

  return args
