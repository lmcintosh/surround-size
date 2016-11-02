import numpy as np
from scipy.optimize import minimize
from scipy.optimize import curve_fit
from scipy.interpolate import interp1d
import matplotlib.pyplot as plt

# my functions
from surround.modeling import *
from surround.data_handling import *

def get_lambda(r0, inputNoise, outputNoise):
    '''Numerical minimization of Lagrangian in Atick & Redlich 1990. 
       Returns lambda and the error.
    '''

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
    
    lam = 5*np.random.rand(1)
    res = minimize(relativeDifference, x0=lam, method='nelder-mead')
    
    return res.x, res.fun


def unique_soln(r0, inputNoise, outputNoise, verbose=True):
    '''R_0 is input covariance matrix
       R is R_0 + N^2 delta_n,m
       Since variance at each frequency is the power spectrum,
       r0 should be power spectrum, NOT amplitude spectrum.
    '''

    #R = np.array(R_0) + inputNoise**2 #* np.eye(*R_0.shape)
    N  = inputNoise
    r0 = np.array(r0)
    r  = r0 + N**2
    
    lam, relErr = get_lambda(r0, inputNoise, outputNoise)
    if verbose:
        print('The relative error from this choice of Lagrangian multiplier is %s.' %(relErr))
        
    f = ((1. / N**2) * (.5 * r0/r * (1 + np.sqrt(1 + (2 * lam * N**2)/r0)) - 1)).astype(complex)
    
    return np.real(outputNoise * np.sqrt(f))


def spectrum_fit(frequencies, spectrum, num_points=5000):
    def func(x, a, b):
            return a/(x**b) # adding a constant tends to make moreSpectra have negative numbers

    popt, pcov  = curve_fit(func, frequencies[1:], spectrum[1:])
    moreFreqs   = np.linspace(0., frequencies[-1], num_points)[1:]
    moreSpectra = func(moreFreqs, *popt)

    return (moreFreqs, moreSpectra)


def fit_ideal(freqs, amplitude, center_width=None, returnFlag='array'):
    '''Fit a linear combination of horizontal + amacrine + Gaussian center
    to the amplitude spectrum of the ideal infomax filter.
    
    if returnFlag='array',
        OUTPUT: matching amplitude spectrum
    elif returnFlag='interp',
        OUTPUT: interp_function, popt
    '''

    # Get original data
    horz_pfs = get_horizontal_projective_field()
    ama_pfs  = get_amacrine_projective_field()

    # get means of the projective fields
    space_h, horz_pf, horz_sem = get_mean(horz_pfs)
    space_a, ama_pf, ama_sem   = get_mean(ama_pfs)

    # interpolate horz and ama to get a unified space; mode='valid'
    horz_interp = interp1d(space_h, horz_pf, kind='slinear')
    ama_interp  = interp1d(space_a, ama_pf, kind='slinear')
    space       = np.linspace(np.max([np.min(space_h), np.min(space_a)]), np.min([np.max(space_h), np.max(space_a)]), 100)

    # project interpolations on unified space
    horz_pf     = horz_interp(space)
    ama_pf      = ama_interp(space)

    # set sampling rate
    spacing = space[-1] - space[-2]
    
    def rf_fft(freqs, horz_weighting, ama_weighting, center_weighting, surround_weighting, center_width=center_width):
        # make surround
        surround = horz_weighting * horz_pf + ama_weighting * ama_pf

        # make center
        if center_width is 0.0:
            center = np.where(abs(surround)==np.max(abs(surround)), 1, 0) # delta function
        else:
            center = gaussian(x=space, sigma=center_width, mu=space[abs(surround)==np.max(abs(surround))]) # gaussian

        # put them together
        if len(center.shape) > 1:
            center = center.squeeze()
        rf = center_weighting * center + surround_weighting * surround

        # Amplitude Spectrum of RF
        rf_f_two_sided = abs(np.fft.fft(rf)) / np.prod(rf.shape)
        n = len(rf_f_two_sided)
        if n % 2 == 0:
            rf_f_one_sided = rf_f_two_sided[:(int(n/2) + 1)]
        else:
            rf_f_one_sided = rf_f_two_sided[:(int((n-1)/2) + 1)]
        rf_freqs_one_sided = np.linspace(0, 1./(2*spacing), len(rf_f_one_sided))

        rf_interp = interp1d(rf_freqs_one_sided, rf_f_one_sided, kind='slinear', fill_value=[0.0], bounds_error=False)
        return rf_interp(freqs)/np.nanmax(rf_interp(freqs))

    # fit least-squares
    if center_width is None:
        popt, pcov = curve_fit(rf_fft, freqs, amplitude, p0=[0.5, 0.75, 6., 0.45, 1.7])
    else:
        popt, pcov = curve_fit(rf_fft, freqs, amplitude, p0=[0.5, 0.75, 6., 0.45])

    if returnFlag == 'array':
        return rf_fft(freqs, *popt)
    if returnFlag == 'interp':
        return rf_fft, popt
    if returnFlag == 'all':
        if center_width is None:
            horz_weight, ama_weight, center_weight, surround_weight, center_width = popt
        else:
            horz_weight, ama_weight, center_weight, surround_weight = popt
        return freqs, rf_fft(freqs, *popt), horz_weight, ama_weight, center_weight, surround_weight, center_width

def fit_ideal_horz_only(freqs, amplitude, center_width=None, returnFlag='array'):
    '''Fit a linear combination of horizontal + amacrine + Gaussian center
    to the amplitude spectrum of the ideal infomax filter'''

    # Get original data
    horz_pfs = get_horizontal_projective_field()

    # get means of the projective fields
    space_h, horz_pf, horz_sem = get_mean(horz_pfs)

    # interpolate horz and ama to get a unified space; mode='valid'
    horz_interp = interp1d(space_h, horz_pf, kind='slinear')
    space       = np.linspace(np.min(space_h), np.max(space_h), 100)

    # project interpolations on unified space
    horz_pf     = horz_interp(space)

    # set sampling rate
    spacing = space[-1] - space[-2]
    
    def rf_fft(freqs, center_weighting, surround_weighting, center_width=center_width):
        # make surround
        surround       = horz_pf

        # make center
        if center_width is 0.0:
            center = np.where(abs(surround)==np.max(abs(surround)), 1, 0) # delta function
        else:
            center = gaussian(x=space, sigma=center_width, mu=space[abs(surround)==np.max(abs(surround))]) # gaussian

        # put them together
        if len(center.shape) > 1:
            center = center.squeeze()
        rf = center_weighting * center + surround_weighting * surround

        # Amplitude Spectrum of RF
        rf_f_two_sided = abs(np.fft.fft(rf)) / np.prod(rf.shape)
        n = len(rf_f_two_sided)
        if n % 2 == 0:
            rf_f_one_sided = rf_f_two_sided[:(int(n/2) + 1)]
        else:
            rf_f_one_sided = rf_f_two_sided[:(int((n-1)/2) + 1)]
        rf_freqs_one_sided = np.linspace(0, 1./(2*spacing), len(rf_f_one_sided))

        rf_interp = interp1d(rf_freqs_one_sided, rf_f_one_sided, kind='slinear', bounds_error=False,
                fill_value=[0.0])
        return rf_interp(freqs)/np.nanmax(rf_interp(freqs))

    # fit least-squares
    if center_width is None:
        popt, pcov = curve_fit(rf_fft, freqs, amplitude, p0=[6., 0.45, 1.7])
    else: 
        popt, pcov = curve_fit(rf_fft, freqs, amplitude, p0=[6., 0.45])

    if returnFlag == 'array':
        return rf_fft(freqs, *popt)
    if returnFlag == 'interp':
        return rf_fft, popt
    if returnFlag == 'all':
        if center_width is None:
            center_weight, surround_weight, center_width = popt
        else:
            center_weight, surround_weight = popt
        return freqs, rf_fft(freqs, *popt), center_weight, surround_weight, center_width

def fit_ideal_ama_only(freqs, amplitude, center_width=None, returnFlag='array'):
    '''Fit a linear combination of horizontal + amacrine + Gaussian center
    to the amplitude spectrum of the ideal infomax filter'''

    # Get original data
    ama_pfs = get_amacrine_projective_field()

    # get means of the projective fields
    space_a, ama_pf, ama_sem = get_mean(ama_pfs)

    # interpolate horz and ama to get a unified space; mode='valid'
    ama_interp = interp1d(space_a, ama_pf, kind='slinear')
    space      = np.linspace(np.min(space_a), np.max(space_a), 100)

    # project interpolations on unified space
    ama_pf     = ama_interp(space)

    # set sampling rate
    spacing = space[-1] - space[-2]
    
    def rf_fft(freqs, center_weighting, surround_weighting, center_width=center_width):
        # make surround
        surround       = ama_pf

        # make center
        if center_width is 0.0:
            center = np.where(abs(surround)==np.max(abs(surround)), 1, 0) # delta function
        else:
            center = gaussian(x=space, sigma=center_width, mu=space[abs(surround)==np.max(abs(surround))]) # gaussian

        # put them together
        if len(center.shape) > 1:
            center = center.squeeze()
        rf = center_weighting * center + surround_weighting * surround

        # Amplitude Spectrum of RF
        rf_f_two_sided = abs(np.fft.fft(rf)) / np.prod(rf.shape)
        n = len(rf_f_two_sided)
        if n % 2 == 0:
            rf_f_one_sided = rf_f_two_sided[:(int(n/2) + 1)]
        else:
            rf_f_one_sided = rf_f_two_sided[:(int((n-1)/2) + 1)]
        rf_freqs_one_sided = np.linspace(0, 1./(2*spacing), len(rf_f_one_sided))

        rf_interp = interp1d(rf_freqs_one_sided, rf_f_one_sided, kind='slinear', bounds_error=False,
                fill_value=[0.0])
        return rf_interp(freqs)/np.nanmax(rf_interp(freqs))

    # fit least-squares
    if center_width is None:
        popt, pcov = curve_fit(rf_fft, freqs, amplitude, p0=[6., 0.45, 1.7])
    else:
        popt, pcov = curve_fit(rf_fft, freqs, amplitude, p0=[6., 0.45])

    if returnFlag == 'array':
        return rf_fft(freqs, *popt)
    if returnFlag == 'interp':
        return rf_fft, popt
    if returnFlag == 'all':
        if center_width is None:
            center_weight, surround_weight, center_width = popt
        else:
            center_weight, surround_weight = popt
        return freqs, rf_fft(freqs, *popt), center_weight, surround_weight, center_width


def get_snr(input_noise, output_noise, signal_spectrum, filter_spectrum, signal_freqs=None, filter_freqs=None):
    ''' Returns the SNR as a function of the standard deviation of input and 
    output noise, and the signal amplitude spectrum.

    Input noise is the amplitude spectrum of input noise. Output noise is amplitude
    spectrum of output noise.

    Note: amplitude spectra must be normalized by 1/len(x) so that spectra[0]
    is the mean of x.

    Signal and filter spectra must be one-sided amplitude spectra.
    
    SNR is (Signal Variance)/(Noise Variance). We can compute this from
    the amplitude spectra because Var(X) = 2*integral(power spectrum).
    '''
    if len(signal_spectrum) != len(filter_spectrum):
        print('Warning: signal spectrum is length %d but filter spectrum is length %d.' %(len(signal_spectrum),
                    len(filter_spectrum)))
        assert signal_freqs is not None, 'You must specify signal frequencies.'
        assert filter_freqs is not None, 'You must specify filter frequencies.'

        # Just interpolate filter spectrum to be over the same range as the signal
        filt_interp = interp1d(filter_freqs, filter_spectrum, kind='slinear', bounds_error=False, fill_value=[0])
        filter_spectrum = filt_interp(signal_freqs)


    signal_var = 2.0*np.sum((filter_spectrum[1:] * signal_spectrum[1:])**2)
    noise_var  = 2.0*np.sum((filter_spectrum[1:] * input_noise)**2 + output_noise**2)

    return signal_var / noise_var


def corresponding_ideal(frequencies, spectrum, expt_freq, expt_amplitude_spectrum, snr):
    '''Takes the power spectra of the signal and the amplitude spectrum of a ganglion cell,
    and returns the input and output noise for the best match ideal filter.

    Returns the std of the input and output noises.
    '''
    def objective(noises):
        input_noise, output_noise = noises
        
        signal_interp = interp1d(frequencies, spectrum, kind='slinear')
        signal_pwr    = signal_interp(expt_freq)
        ideal_filt    = unique_soln(signal_pwr, input_noise, output_noise, verbose=False)

        return np.mean((ideal_filt - expt_amplitude_spectrum)**2)


    def constraint(noises):
        input_noise, output_noise = noises

        # SNR is a function of input and output noise stds, signal_spect, filter_spect
        return (snr - get_snr(input_noise, output_noise, spectrum, expt_amplitude_spectrum))**2


    # minimize the difference between the ideal filter and the ganglion cell spectrum
    res = minimize(fun=objective, x0=np.array([0.1, 0.4]), method='L-BFGS-B')
#            constraints={'type':'eq', 'fun':constraint}, options={'disp':True})

    return res.x




def fig5_each_cell(frequencies, spectra, snr, plot_style='same'):
    ''' Generates Figure 5 with an infomax fit per ganglion cell.
    Steps:
    1) Get FFT of receptive field for each ganglion cell
    2) Fit each ganglion RF FFT to an ideal infomax curve, varying input + output noise
    3) Fit FFT of a linear combination of Ama., Horz., and Center. to the ideal infomax curve
    4) Plot all of the ideal infomax power spectra and the ffts of the models together
    
    INPUTS:
    frequencies: np array of spatial frequencies corresponding to spectra
    spectra: np array of original power spectra
    
    RETURNS:
    frequencies_ideal, ideal_filter, frequencies_expt, expt_filter
    '''

    ###### GET FFT OF GANGLION RFS #######
    ganglion_rfs  = load_ganglion_cells(micronsPerDeg=50., pca_mode='space')
    ganglion_ffts = get_fft(ganglion_rfs, mode='amplitude')

    for spatial_freq, ganglion_amp_spect in ganglion_ffts:
        # get the input and output noise corresponding to the matching ideal condition
        input_noise, output_noise = corresponding_ideal(frequencies, spectra, spatial_freq, ganglion_amp_spect, snr)

        print (input_noise, output_noise)
        
        # get the ideal filter
        freq_ideal, filt_ideal, tmp1, tmp2 = compare_to_experiment(frequencies, spectra, 
                inputNoise=abs(input_noise), outputNoise=abs(output_noise), returnFlag=True, plotFlag=False)

        model_freqs     = np.linspace(np.max([np.min(freq_ideal), np.min(spatial_freq)]),
                np.min([np.max(freq_ideal), np.max(spatial_freq)]), len(freq_ideal))
        ideal_interp    = interp1d(freq_ideal, filt_ideal, kind='slinear')
        resampled_ideal = ideal_interp(model_freqs)

        filt_model = fit_ideal(model_freqs, resampled_ideal/np.nanmax(resampled_ideal), returnFlag='array')

        if plot_style == 'random':
            r            = lambda: np.random.randint(0,255)
            random_color = '#%02X%02X%02X' % (r(),r(),r())
            plt.plot(freq_ideal, filt_ideal/np.nanmax(filt_ideal), color=random_color, alpha=0.7,
                    linestyle='.', marker='o')
            plt.plot(model_freqs, filt_model, color=random_color, alpha=0.7, linestyle='--',
                    marker='')
            plt.plot(spatial_freq, ganglion_amp_spect/np.nanmax(ganglion_amp_spect), color=random_color, 
                    alpha=0.5, linewidth=2)
        elif plot_style == 'same':
            plt.plot(freq_ideal, filt_ideal/np.nanmax(filt_ideal), color='c', alpha=0.7, linewidth=3)
            plt.plot(model_freqs, filt_model, color='b', alpha=0.7, linewidth=3)
            plt.plot(spatial_freq, ganglion_amp_spect/np.nanmax(ganglion_amp_spect), color='k', alpha=0.6, linewidth=2)

        plt.tick_params(axis='y', direction='out')
        plt.tick_params(axis='x', direction='out')
        adjust_spines(plt.gca(), ['left', 'bottom'])
        plt.xlim([0,1])
        plt.ylim([0,1.1])
