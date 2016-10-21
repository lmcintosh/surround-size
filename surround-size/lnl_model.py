#!/usr/bin/env python
import sys
import numpy as np
import time
from numpy.random import randn

# my packages for estimating entropy
from info_theory_functions import *
from lnl_model_functions import *

# path to van Hateren Natural Image Database
im_path = '/Users/lmcintosh/Documents/Natural_Images'
fig_path = '/Users/lmcintosh/Dropbox/Stanford University/00 Baccus Lab/Surround size project/figures/'
#im_path = '/home/lane/Natural_images/imc'


# PARAMETERS
args               = readCommand( sys.argv[1:] ) # Get command line input
key                = args['key'] # default 3
numPhotoreceptors  = args['numPhotoreceptors'] # default 16
numSubunits        = args['numSubunits'] # default 16
nonlinearGain      = args['nonlinearGain'] # default 0.25
fixed_center_width = 0.25
s_widths           = np.linspace(0.3,6,10)
SNR                = args['snr'] # default 1000
iterations         = args['iterations'] # default 80
nBins              = 10
numImages          = args['numImages'] # default 20
patchSize          = args['patchSize'] # default 32
eps                = 10e-5

fileIdentifier = str(args).replace('{','').replace('}','').replace(':',',')

# GET DATA
patches   = load_images(path=im_path, numImages=numImages, patchSize=patchSize)

# INITIALIZE INFORMATION ARRAYS
I = zeros((len(s_widths),len(s_widths)))
H_rs = zeros((len(s_widths),len(s_widths)))
H_r_s_avg = zeros((len(s_widths),len(s_widths)))

start = time.time() 
# MAIN LOOP
for ids1,s1 in enumerate(s_widths):
    for ids2,s2 in enumerate(s_widths):
        sys.stdout.write('\r')

        # Make filters
        X_2d1 = spatial_filter_2d(c_width=fixed_center_width,s_width=s1,xs_num=numPhotoreceptors)
        X_2d2 = spatial_filter_2d(c_width=fixed_center_width,s_width=s2,xs_num=numSubunits)

        # Get model threshold and output variance
        threshold = get_threshold(patches, X_2d1)
        #signalVar = get_signal_variance(patches, X_2d1, X_2d2, threshold=threshold, subunits=numSubunits)

        noiselessOutput = [lnl_output(p,filter1=X_2d1,filter2=X_2d2,nonlinearThreshold=threshold,nonlinearGain=nonlinearGain,numSubunits=numSubunits,SNR='Inf',key=key) for p in patches] 
        signalVar = np.var(noiselessOutput)
        
        output     = np.zeros((numImages,iterations))
        for idi in xrange(numImages):
            output[idi,:] = noiselessOutput[idi] + (signalVar/SNR)*randn(iterations)

        # pad binning in off-chance that output has zero entropy
        binning = np.linspace(np.min(output)-eps,np.max(output)+eps,nBins+1)

        H_r = entropy(output.flat,Bins=[binning])
        H_r_s = []
        for s in xrange(int(numImages)):
            H_r_s.append(entropy(output[s,:],Bins=[binning]))
        I[ids1,ids2] = H_r - np.mean(H_r_s)
        H_rs[ids1,ids2] = H_r
        H_r_s_avg[ids1,ids2] = np.mean(H_r_s)

        msg1 = 'Current sweep is (%d,%d) of (%d,%d).' % (ids1,ids2,len(s_widths)-1,len(s_widths)-1)
        msg2 = " It's taking %f seconds." % (time.time()-start)
        msg = msg1+msg2
        sys.stdout.write(msg)
        sys.stdout.flush()
print('')

print_matrix(I, min_x=s_widths[0], max_x=s_widths[-1], min_y=s_widths[0], max_y=s_widths[-1], top_title='Information transmission in LNL model', path=fig_path, extraInfo=fileIdentifier)
print_matrix(H_rs, min_x=s_widths[0], max_x=s_widths[-1], min_y=s_widths[0], max_y=s_widths[-1], top_title='Total response entropy', path=fig_path, extraInfo=fileIdentifier)
print_matrix(H_r_s_avg, min_x=s_widths[0], max_x=s_widths[-1], min_y=s_widths[0], max_y=s_widths[-1], top_title='Average conditional entropy', path=fig_path, extraInfo=fileIdentifier)



