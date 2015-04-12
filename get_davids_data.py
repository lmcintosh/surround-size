import numpy as np
from sklearn.decomposition import PCA
from os.path import expanduser
from scipy.interpolate import interp1d
from scipy.stats import sem

def get_space(rf, spatial_delta, microns_per_deg, scale=True):
    '''Returns a spatial vector for each point in 1d vector rf,
    with zero degrees aligned to the max(abs(rf)).
    INPUT:
    rf              is a 1d numpy array
    spatial_delta   is a float in mm
    microns_per_deg is a float in microns/deg
    
    RETURNS:
    space           is a 1d numpy array in degrees
    '''
    peak  = np.argmax(abs(rf))
    space = np.linspace(-spatial_delta*peak, spatial_delta*(len(rf)-peak), len(rf))
    if scale:
        space *= 1000 # mm to microns
        space /= microns_per_deg
    return space

def get_interp(data, interpolation='slinear', nPoints=200):
    '''Aligns and interpolates list of (x,y) data. 
       Returns list of (aligned_x, aligned_y) tuples.
    '''
    interp_data = []

    max_x     = np.max([np.max(x) for x, y in data])
    min_x     = np.min([np.min(x) for x, y in data])
    if nPoints:
        aligned_x = np.linspace(min_x, max_x, nPoints)
    else:
        # otherwise just use the first x, which should be the same for all data
        aligned_x = data[0][0]

    for x, y in data:
        if interpolation:
            # fill out of bounds with nans
            y_interp = interp1d(x, y, kind=interpolation, bounds_error=False)
            interp_data.append((aligned_x, y_interp(aligned_x)))
        else:
            interp_data.append((aligned_x, y))
            assert len(y) == len(aligned_x), 'x and y must have same length'

    return interp_data


def get_mean(data, interpolation='slinear', nPoints=200):
    '''Return (space, mean_rf, error) tuple. Input should be list of (x, y) tuples (e.g. (space, rf)).
    '''
    max_x     = np.max([np.max(x) for x, y in data])
    min_x     = np.min([np.min(x) for x, y in data])
    if nPoints:
        aligned_x = np.linspace(min_x, max_x, nPoints)
    else:
        # otherwise just use the first x, which should be the same for all data
        aligned_x = data[0][0]
    aligned_y = []

    for x, y in data:
        if interpolation:
            # fill out of bounds with nans
            y_interp = interp1d(x, y, kind=interpolation, bounds_error=False)
            aligned_y.append(y_interp(aligned_x))
        else:
            aligned_y.append(y)
            assert len(y) == len(aligned_x), 'x and y must have same length'

    mean_y = np.nanmean(np.vstack(aligned_y), axis=0)
    errors = sem(np.vstack(aligned_y), axis=0)

    return (aligned_x, mean_y, errors)

def get_fft(data, mode='fourier'):
    '''Return (frq, fft) tuple.
       Mode can be 'fourier', 'amplitude', or 'power'
    '''
    ffts = []

    for space, y in data:
        if mode == 'fourier':
            fft_two_sided = np.fft.fft(y)
        if mode == 'amplitude':
            fft_two_sided = abs(np.fft.fft(y))
        if mode == 'power':
            fft_two_sided = abs(np.fft.fft(y))**2
        n = len(fft_two_sided)
        if n % 2 == 0:
            fft_one_sided = fft_two_sided[:n/2 + 1]
        else:
            fft_one_sided = fft_two_sided[:(n-1)/2 + 1]

        spacing = space[-1] - space[-2]
        freqs_one_sided = np.linspace(0, 1./(2*spacing), len(fft_one_sided))

        #freqs_two_sided = np.fft.fftshift(np.fft.fftfreq(len(fft_two_sided), Fs))
        #freqs_one_sided = freqs_two_sided[len(freqs_two_sided)/2:]

        ffts.append((freqs_one_sided, fft_one_sided))

    return ffts


def load_ganglion_cells(micronsPerDeg=50., pca_mode='space'):
    ''' Returns list of tuples (space, spatial receptive field)
    '''

    data_path = expanduser('~/Documents/Stanford/00 Baccus Lab/Surround size project/david_receptive_fields')
    file_name = data_path + '/allGC.txt'
    data_gc   = np.loadtxt(file_name, delimiter="\t")
    data_gc   = data_gc.reshape((100,80,28))
    nCells    = data_gc.shape[2]

    # get spacing for spatial receptive fields
    spatialDelta = 0.027 # mm

    # since receptive fields are noisy, use PCA
    spatial_rfs = []
    for n in range(nCells):
        pca = PCA(n_components=2)
        if pca_mode == 'space':
            pca.fit(data_gc[:,:,n])
            g_pca = pca.components_[0]
        elif pca_mode == 'time':
            pca.fit(data_gc[:,:,n].T)
            g_pca = np.dot(data_gc[:,:,n].T, pca.components_[0])


        sign_of_pc = -1 * np.sign(g_pca[abs(g_pca) == np.max(abs(g_pca))])
        space      = get_space(g_pca, spatialDelta, micronsPerDeg)

        spatial_rfs.append((space, sign_of_pc * g_pca))

    return spatial_rfs

def load_bipolar_cells(micronsPerDeg=50.):
    ''' Returns list of tuples (space, spatial receptive field)
    '''

    data_path  = expanduser('~/Documents/Stanford/00 Baccus Lab/Surround size project/david_receptive_fields')
    file_name1 = data_path + '/B1.txt'
    file_name2 = data_path + '/B2.txt'
    data_b1    = np.loadtxt(file_name1, delimiter="\t") # 50 time x 100 space
    data_b2    = np.loadtxt(file_name2, delimiter="\t") # 50 time x 100 space
    data_b     = [data_b1, data_b2]

    # get spacing for all bipolar spatial receptive fields
    spatialDelta = 0.022 # mm

    # since receptive fields are noisy, use PCA
    spatial_rfs = []
    for b in data_b:
        pca = PCA(n_components=2)
        pca.fit(b)

        b_pca      = pca.components_[0]
        sign_of_pc = -1 * np.sign(b_pca[abs(b_pca) == np.max(abs(b_pca))])
        space      = get_space(b_pca, spatialDelta, micronsPerDeg)

        spatial_rfs.append((space, sign_of_pc * b_pca))

    return spatial_rfs


def load_amacrine_cells(micronsPerDeg=50.):
    ''' Returns list of tuples (space, spatial receptive field)
    '''

    data_path = expanduser('~/Documents/Stanford/00 Baccus Lab/Surround size project/david_receptive_fields')
    file_name = data_path + '/H1A2.txt'
    data_ha   = np.loadtxt(file_name, delimiter="\t")
    data_ha   = data_ha.reshape((3,50,80))
    data_a    = data_ha[1:, :, :] # first cell in this file is a horizontal cell
    nCells    = data_a.shape[0]

    # get spacing for spatial receptive fields
    spatialDelta = 0.027 # mm

    # since receptive fields are noisy, use PCA
    spatial_rfs = []
    for n in range(nCells):
        pca = PCA(n_components=2)
        pca.fit(data_a[n,:,:])

        spatial_rfs.append((get_space(pca.components_[0], spatialDelta, micronsPerDeg), pca.components_[0]))

    return spatial_rfs


def load_horizontal_cells(micronsPerDeg=50.):
    ''' Returns list of tuples (space, spatial receptive field)
    '''

    data_path = expanduser('~/Documents/Stanford/00 Baccus Lab/Surround size project/david_receptive_fields')
    
    ###### CELL 1 ######
    file_name = data_path + '/H1A2.txt'
    data_ha   = np.loadtxt(file_name, delimiter="\t")
    data_ha   = data_ha.reshape((3,50,80))
    data_h    = data_ha[0, :, :] # first cell in this file is a horizontal cell

    # get spacing for spatial receptive fields
    spatialDelta = 0.027 # mm

    # since receptive fields are noisy, use PCA
    spatial_rfs = []
    pca = PCA(n_components=2)
    pca.fit(data_h)
    spatial_rfs.append((get_space(pca.components_[0], spatialDelta, micronsPerDeg), pca.components_[0]))


    ###### CELL 2 ######
    file_name = data_path + '/H2.txt'
    data_h    = np.loadtxt(file_name, delimiter="\t") # shape is (50, 100) or (time, space)

    # get spacing for spatial receptive fields
    spatialDelta = 0.022 # mm

    # since receptive fields are noisy, use PCA
    pca = PCA(n_components=2)
    pca.fit(data_h)
    spatial_rfs.append((get_space(pca.components_[0], spatialDelta, micronsPerDeg), pca.components_[0]))

    return spatial_rfs


def get_amacrine_projective_field(micronsPerDeg=50.):
    spatial_rfs = load_amacrine_cells(micronsPerDeg=micronsPerDeg)
    spatial_pfs = []
    time_const  = 78./22.

    for space, rf in spatial_rfs:
        proj_range = [np.exp(-abs(t)/time_const) for t in np.linspace(np.min(space),np.max(space),len(space))]
        proj_field = np.convolve(proj_range, rf, mode='same')
        proj_field *= np.mean(rf) / np.mean(proj_field)

        spatial_pfs.append((get_space(proj_field, space[1]-space[0], micronsPerDeg, scale=False), proj_field))

    return spatial_pfs


def get_horizontal_projective_field(micronsPerDeg=50.):
    spatial_rfs = load_horizontal_cells(micronsPerDeg=micronsPerDeg)
    spatial_pfs = []
    time_const  = 118./22.

    for space, rf in spatial_rfs:
        proj_range = [np.exp(-abs(t)/time_const) for t in np.linspace(np.min(space),np.max(space),len(space))]
        proj_field = np.convolve(proj_range, rf, mode='same')
        proj_field *= np.mean(rf) / np.mean(proj_field)

        spatial_pfs.append((get_space(proj_field, space[1]-space[0], micronsPerDeg, scale=False), proj_field))

    return spatial_pfs


def get_amacrine_projective_fft(micronsPerDeg=50., fft_mode='amplitude'):
    '''Returns tuples of (frequencies, amplitude spectrum)
    for amacrine projective field.
    '''
    spatial_pfs = get_amacrine_projective_field(micronsPerDeg=micronsPerDeg)

    return get_fft(spatial_pfs, mode=fft_mode)

def get_horizontal_projective_fft(micronsPerDeg=50., fft_mode='amplitude'):
    '''Returns tuples of (frequencies, amplitude spectrum)
    for horizontal projective field.
    '''
    spatial_pfs = get_horizontal_projective_field(micronsPerDeg=micronsPerDeg)

    return get_fft(spatial_pfs, mode=fft_mode)
