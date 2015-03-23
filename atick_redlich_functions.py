import numpy as np
from scipy.optimize import minimize

def getLambda(r0, inputNoise, outputNoise):
    
    N = inputNoise
    def lhs(lam):
        return np.sum(np.log(np.sqrt(r0/(2 * lam * N**2)) + np.sqrt(1 + (r0/(2 * lam * N**2)))))
    
    #R = np.array(R_0) + inputNoise**2
    r0 = np.array(r0)
    r  = r0 + N**2
    rhs = 0.5 * np.sum(np.log((r/(N**2)) * ((N**2 + outputNoise**2)/(r + outputNoise**2))))
    
    def relativeDifference(lam):
        left = lhs(lam)
        return abs(left-rhs)/(abs(left) + abs(rhs))
    
    lam = 5*rand(1)
    res = minimize(relativeDifference, x0=lam, method='nelder-mead')
    
    return res.x, res.fun


def uniqueSoln(r0, inputNoise, outputNoise, verbose=True):
    '''R_0 is input covariance matrix
       R is R_0 + N^2 delta_n,m'''

    #R = np.array(R_0) + inputNoise**2 #* np.eye(*R_0.shape)
    N  = inputNoise
    r0 = np.array(r0)
    r  = r0 + N**2
    
    lam, relErr = getLambda(r0, inputNoise, outputNoise)
    if verbose:
        print 'The relative error from this choice of Lagrangian multiplier is %s.' %(relErr)
        
    f = ((1. / N**2) * (.5 * r0/r * (1 + np.sqrt(1 + (2 * lam * N**2)/r0)) - 1)).astype(complex)
    
    return real(outputNoise * np.sqrt(f))
