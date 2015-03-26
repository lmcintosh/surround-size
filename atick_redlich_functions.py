import numpy as np
from scipy.optimize import minimize
from scipy.optimize import curve_fit
from scipy.interpolate import interp1d
import matplotlib.pyplot as plt

# my functions
from lnl_model_functions import *
from get_davids_data import *

def get_lambda(r0, inputNoise, outputNoise):

    assert np.min(r0) >= 0, 'Uhoh, the power spectra is negative!'
    
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


def unique_soln(r0, inputNoise, outputNoise, verbose=True):
    '''R_0 is input covariance matrix
       R is R_0 + N^2 delta_n,m'''

    #R = np.array(R_0) + inputNoise**2 #* np.eye(*R_0.shape)
    N  = inputNoise
    r0 = np.array(r0)
    r  = r0 + N**2
    
    lam, relErr = get_lambda(r0, inputNoise, outputNoise)
    if verbose:
        print 'The relative error from this choice of Lagrangian multiplier is %s.' %(relErr)
        
    f = ((1. / N**2) * (.5 * r0/r * (1 + np.sqrt(1 + (2 * lam * N**2)/r0)) - 1)).astype(complex)
    
    return real(outputNoise * np.sqrt(f))

def compare_to_experiment(frequencies, spectra, space_h=None, proj_h=None, space_a=None, proj_a=None, 
        inputNoise=0.1, outputNoise=0.4, center_weighting=2.1, surround_weighting=0.1,
        horz_weighting=0.5, ama_weighting=0.5, center_width=.5, interpolation='fit', 
        numPoints=1000, returnFlag=False, plotFlag='aggregate', verbose=True, xlimit=None):
    ''' Compare ideal infomax filter to experimental projective fields.
    INPUTS:
    frequencies: np array of spatial frequencies corresponding to spectra
    spectra: np array of original power spectra
    space_h: space in degrees corresponding to horizontal cell projective field
    proj_h: horizontal cell projective field
    space_a: space in degrees corresponding to amacrine cell projective field
    proj_a: amacrine cell projective field
    
    RETURNS:
    frequencies_ideal, ideal_filter, frequencies_expt, expt_filter
    '''

    ###### EXPERIMENTAL ######
    # make surround
    if space_h is None:
        horz_pfs = get_horizontal_projective_field()
        ama_pfs  = get_amacrine_projective_field()

        # for now just pick one of each
        space_h, horz_pf = horz_pfs[0]
        space_a, ama_pf  = ama_pfs[0]
        space            = space_h

        assert len(space_h) == len(space_a), 'Horizontal and amacrine space must be the same.'
        
        Fs_h = space_h[-1] - space_h[-2]
        Fs_a = space_a[-1] - space_a[-2]
        Fs   = Fs_h

        assert Fs_h == Fs_a, 'Horizontal and amacrine spacing must be the same.'
    
    # make surround
    surround       = horz_weighting * horz_pf + ama_weighting * ama_pf

    # make center
    if center_width is None:
        center = center_weighting * np.where(abs(surround)==np.max(abs(surround)), 1, 0) # delta function
    else:
        center = gaussian(x=space, sigma=center_width, mu=space[abs(surround)==np.max(abs(surround))]) # gaussian

    # put them together
    if len(center.shape) > 1:
        center = center.squeeze()
    rf = center_weighting * center + surround_weighting * surround
    
    # FFT of RF
    rf_f_two_sided = abs(np.fft.fftshift(np.fft.fft(rf)))
    rf_f_one_sided = rf_f_two_sided[len(rf_f_two_sided)/2:]

    rf_freqs_two_sided = np.fft.fftshift(np.fft.fftfreq(len(rf_f_two_sided),Fs))
    rf_freqs_one_sided = rf_freqs_two_sided[len(rf_freqs_two_sided)/2:]
    
    
    ###### IDEAL ######
    if interpolation == 'linear':
        naturalInterpolation = interp1d(frequencies, spectra, kind='linear')
        moreFreqs   = np.linspace(0., frequencies[-1], numPoints)
        moreSpectra = naturalInterpolation(moreFreqs)
        idealFilter = unique_soln(moreSpectra, inputNoise, outputNoise, verbose=verbose) # spectra, input_noise, output_noise
    elif interpolation == 'cubic':
        naturalInterpolation = interp1d(frequencies, spectra, kind='cubic')
        moreFreqs   = np.linspace(0., frequencies[-1], numPoints)
        moreSpectra = naturalInterpolation(moreFreqs)
        idealFilter = unique_soln(moreSpectra, inputNoise, outputNoise, verbose=verbose) # spectra, input_noise, output_noise
    elif interpolation == 'fit':
        # power law
        def func(x, a, b):
            return a/(x**b) # adding a constant tends to make moreSpectra have negative numbers

        popt, pcov  = curve_fit(func, frequencies[1:], spectra[1:])
        moreFreqs   = np.linspace(0., frequencies[-1], numPoints)[1:]
        moreSpectra = func(moreFreqs, *popt)
        idealFilter = unique_soln(moreSpectra, inputNoise, outputNoise, verbose=verbose) # spectra, input_noise, output_noise
    elif interpolation is None:
        idealFilter = unique_soln(spectra, inputNoise, outputNoise, verbose=verbose)
        
    if plotFlag == 'aggregate':
        plt.plot(moreFreqs, idealFilter, 'r', linewidth=3, alpha=0.8)
        plt.plot(rf_freqs_one_sided, rf_f_one_sided, 'k', linewidth=3, alpha=0.8)
        if xlimit:
            plt.xlim(xlimit)
    elif plotFlag == 'separate':
        horz_ffts = get_horizontal_projective_fft()
        ama_ffts  = get_amacrine_projective_fft()

        # for now just pick one
        freq_h, horz_fft = horz_ffts[0]
        freq_a, ama_fft  = ama_ffts[0]

        flag = 'exp'

        if flag == 'interp':
            # interpolate horizontal and amacrine projective field filters in frequency space
            horz_interpolate = interp1d(freq_h, 1-horz_fft/np.max(horz_fft), kind='slinear')
            ama_interpolate  = interp1d(freq_a, 1-ama_fft/np.max(ama_fft), kind='slinear')

            horz_interpolate_x = np.linspace(freq_h[0], freq_h[-1], len(moreFreqs))
            ama_interpolate_x  = np.linspace(freq_a[0], freq_a[-1], len(moreFreqs))

            horz_fit = horz_interpolate(horz_interpolate_x)
            ama_fit  = ama_interpolate(ama_interpolate_x)

        elif flag == 'exp':
            def func(x, a, b, c):
                # exponential
                y = a * np.exp(-b * x) + c

                # power law
                #y = a / x**b + c
                #y[x==0] = np.max(y[np.isfinite(y)])

                return 1 - y/np.max(y)

            popt_horz, pcov_horz = curve_fit(func, freq_h, 1-horz_fft/np.max(horz_fft))
            popt_ama, pcov_ama   = curve_fit(func, freq_a, 1-ama_fft/np.max(ama_fft))
            
            horz_interpolate_x = np.linspace(freq_h[0], freq_h[-1], len(moreFreqs))
            ama_interpolate_x  = np.linspace(freq_a[0], freq_a[-1], len(moreFreqs))
            
            horz_fit = func(horz_interpolate_x, *popt_horz)
            ama_fit  = func(ama_interpolate_x, *popt_ama)

                
        plt.plot(moreFreqs, idealFilter/np.nanmax(idealFilter), 'b-', linewidth=3, alpha=0.5) 
        plt.plot(horz_interpolate_x, horz_fit, 'g', linewidth=2, alpha=0.8)
        plt.plot(ama_interpolate_x, ama_fit, 'c', linewidth=2, alpha=0.8)
        plt.plot(freq_h, 1 - horz_fft/np.max(horz_fft), 'g.', alpha=0.8, markersize=10)
        plt.plot(freq_a, 1 - ama_fft/np.max(ama_fft), 'c.', alpha=0.8, markersize=10)
        if xlimit:
            plt.xlim(xlimit)

    
    if returnFlag:
        try:
            return moreFreqs, idealFilter, rf_freqs_one_sided, rf_f_one_sided
        except:
            return frequencies, idealFilter, rf_freqs_one_sided, rf_f_one_sided
