import numpy as np
from os import listdir
from os.path import isfile, join
import array as ar
import scipy.io

def load_images(path, numImages, patchSize=None, acceptedExtensions=['.imc','LUM.mat'], square=False, normalize='subtract', effective_contrast=None):
    ''' Load images and return a list of numpy arrays.'''
    
    allfiles = []
    for ext in acceptedExtensions:
        allfiles = allfiles + [f for f in listdir(path) if isfile(join(path,f)) and ext in f]
    if len(allfiles) < numImages:
        print 'Only ' + str(len(allfiles)) + ' files in directory.'
    names    = allfiles[:numImages]
    patches  = []

    for im in names:
        if '.imc' in im:
            fin = open(path+'/'+im, 'rb')
            s   = fin.read()
            fin.close()
            arr = ar.array('H', s)
            arr.byteswap()
            img = np.array(arr, dtype='uint16').reshape(1024,1536)
        elif '.mat' in im:
            fin = scipy.io.loadmat(path+'/'+im)
            arr = fin['LUM_Image']
            # make 0 to 255
            arr = arr - min(arr.flat)
            arr = 255.*arr/max(arr.flat)
            img = np.array(arr, dtype='uint16')
            
        # make patches
        if patchSize:
            img_center = [np.shape(img)[0]/2,np.shape(img)[1]/2]
            img        = img[img_center[0]-patchSize/2:img_center[0]+patchSize/2,img_center[1]-patchSize/2:img_center[1]+patchSize/2]
            
        if square:
            N = min(np.shape(img))
            N = N - (N % 2) # make even
            img = img[:N,:N]
        
        if normalize == 'subtract':
            # normalize to unit variance and zero mean (originally I did these in opposite order, but this
            # is the official z-scoring order I believe)
            img -= np.mean(img)
            patches.append(img/np.std(img))
        elif normalize == 'divisive':
            # normalize by the mean
            if effective_contrast:
                img = img/np.mean(img) # can't use /= because img.dtype is uint16
                img /= np.std(img)
                img += 1./effective_contrast - np.mean(img)
                patches.append(img)
            else:
                patches.append(img/np.mean(img))

        else:
            patches.append(img)

    return patches

def cart2pol(x, y):
    rho = np.sqrt(x**2 + y**2)
    phi = np.arctan2(y, x)
    return (phi, rho)

def rotavg(im):
    '''im can be square 2d numpy array'''
    imShape = im.shape # NNM
    N       = imShape[0]
    X, Y    = np.meshgrid(range(-N//2,N//2),range(-N//2,N//2))

    theta, rho = cart2pol(X,Y);
    rho = rho.round()
    f   = [np.mean(im[np.where(rho==r)]) for r in range(N//2+1)] 
        
    return f

def spectrum2d(arr, spacing=1.0, frequencyFlag=True):
    amp = abs(np.fft.fftshift(np.fft.fft2(arr)))/np.prod(arr.shape)
    amp = rotavg(amp)
    n   = len(amp)
    frq = np.fft.fftfreq(2*n, spacing)[:n] 

    if frequencyFlag:
        return (frq, amp)
    else:
        return amp
