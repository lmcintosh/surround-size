import numpy as np

##### FUNCTIONS #####
def gaussian(x=np.linspace(-1, 1, 10), sigma=1., mu=0., magnitude=1):
    '''Returns the shape of a Gaussian distribution
       with mean mu, standard deviation sigma, and
       evaluated at x.
    '''
    return magnitude * np.array([(1./(2.*np.math.pi*sigma**2))*np.exp((-(xi-mu)**2.)/(2.*sigma**2)) for xi in x])


def difference_of_gaussians(x=np.linspace(-1, 1, 10), sigma1=1, sigma2=2, strength1=1, strength2=2, mu=0.):
    '''Returns a difference of two Gaussians.
       You specify the standard deviations of the
       two Gaussians, their means, and the magnitudes 
       of each Gaussian.
    '''
    gauss1 = gaussian(x=x, sigma=sigma1, mu=mu, magnitude=strength1)
    gauss2 = gaussian(x=x, sigma=sigma2, mu=mu, magnitude=strength2)
    return gauss1 + gauss2
