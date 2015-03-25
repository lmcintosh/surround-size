import numpy as np
from sklearn.decomposition import PCA
from os.path import expanduser

def load_ganglion_cells(micronsPerDeg=50.):
    ''' Returns list of tuples (space, spatial receptive field)
    '''

    data_path = expanduser('~/Documents/Stanford/00 Baccus Lab/Surround size project/david_receptive_fields')
    file_name = data_path + '/allGC.txt'
    data_gc   = np.loadtxt(file_name, delimiter="\t")
    data_gc   = data_gc.reshape((100,80,28))
    nCells    = data_gc.shape[2]

    # get spacing for spatial receptive fields
    spatialDelta = 0.027 # mm
    space = np.linspace(-spatialDelta*data_gc.shape[1]/2., spatialDelta*data_gc.shape[1]/2., data_gc.shape[1])
    space *= 1000 # mm to microns
    space /= micronsPerDeg

    # since receptive fields are noisy, use PCA
    spatial_rfs = []
    for n in range(nCells):
        pca = PCA(n_components=2)
        pca.fit(data_gc[:,:,n])

        spatial_rfs.append((space, pca.components_[0]))

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
    space = np.linspace(-spatialDelta*data_a.shape[2]/2., spatialDelta*data_a.shape[2]/2., data_a.shape[2])
    space *= 1000 # mm to microns
    space /= micronsPerDeg

    # since receptive fields are noisy, use PCA
    spatial_rfs = []
    for n in range(nCells):
        pca = PCA(n_components=2)
        pca.fit(data_a[n,:,:])

        spatial_rfs.append((space, pca.components_[0]))

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
    space = np.linspace(-spatialDelta*data_h.shape[1]/2., spatialDelta*data_h.shape[1]/2., data_h.shape[1])
    space *= 1000 # mm to microns
    space /= micronsPerDeg

    # since receptive fields are noisy, use PCA
    spatial_rfs = []
    pca = PCA(n_components=2)
    pca.fit(data_h)
    spatial_rfs.append((space, pca.components_[0]))


    ###### CELL 2 ######
    file_name = data_path + '/H2.txt'
    data_h    = np.loadtxt(file_name, delimiter="\t") # shape is (50, 100) or (time, space)

    # get spacing for spatial receptive fields
    spatialDelta = 0.022 # mm
    space = np.linspace(-spatialDelta*data_h.shape[1]/2., spatialDelta*data_h.shape[1]/2., data_h.shape[1])
    space *= 1000 # mm to microns
    space /= micronsPerDeg

    # since receptive fields are noisy, use PCA
    pca = PCA(n_components=2)
    pca.fit(data_h)
    spatial_rfs.append((space, pca.components_[0]))

    return spatial_rfs


def get_amacrine_projective_field(micronsPerDeg=50.):
    spatial_rfs = load_amacrine_cells(micronsPerDeg=micronsPerDeg)
    spatial_pfs = []
    time_const  = 78./22.

    for space, rf in spatial_rfs:
        proj_range = [np.exp(-abs(t)/time_const) for t in np.linspace(np.min(space),np.max(space),len(space))]
        proj_field = np.convolve(proj_range, rf, mode='same')
        scaling    = np.mean(rf) / np.mean(proj_field)
        proj_field *= scaling

        spatial_pfs.append((space, proj_field))

    return spatial_pfs


def get_horizontal_projective_field(micronsPerDeg=50.):
    spatial_rfs = load_horizontal_cells(micronsPerDeg=micronsPerDeg)
    spatial_pfs = []
    time_const  = 118./22.

    for space, rf in spatial_rfs:
        proj_range = [np.exp(-abs(t)/time_const) for t in np.linspace(np.min(space),np.max(space),len(space))]
        proj_field = np.convolve(proj_range, rf, mode='same')
        scaling    = np.mean(rf) / np.mean(proj_field)
        proj_field *= scaling

        spatial_pfs.append((space, proj_field))

    return spatial_pfs


def get_amacrine_projective_fft(micronsPerDeg=50.):
    '''Returns tuples of (frequencies, amplitude spectrum)
    for amacrine projective field.
    '''
    spatial_pfs = get_amacrine_projective_field(micronsPerDeg=micronsPerDeg)
    spatial_fft = []

    for space, pf in spatial_pfs:
        pf_f_two_sided = abs(np.fft.fftshift(np.fft.fft(pf)))
        pf_f_one_sided = pf_f_two_sided[len(pf_f_two_sided)/2:]

        Fs = space[-1] - space[-2]
        pf_freqs_two_sided = np.fft.fftshift(np.fft.fftfreq(len(pf_f_two_sided), Fs))
        pf_freqs_one_sided = pf_freqs_two_sided[len(pf_freqs_two_sided)/2:]

        spatial_fft.append((pf_freqs_one_sided, pf_f_one_sided))
    
    return spatial_fft

def get_horizontal_projective_fft(micronsPerDeg=50.):
    '''Returns tuples of (frequencies, amplitude spectrum)
    for horizontal projective field.
    '''
    spatial_pfs = get_horizontal_projective_field(micronsPerDeg=micronsPerDeg)
    spatial_fft = []

    for space, pf in spatial_pfs:
        pf_f_two_sided = abs(np.fft.fftshift(np.fft.fft(pf)))
        pf_f_one_sided = pf_f_two_sided[len(pf_f_two_sided)/2:]

        Fs = space[-1] - space[-2]
        pf_freqs_two_sided = np.fft.fftshift(np.fft.fftfreq(len(pf_f_two_sided), Fs))
        pf_freqs_one_sided = pf_freqs_two_sided[len(pf_freqs_two_sided)/2:]

        spatial_fft.append((pf_freqs_one_sided, pf_f_one_sided))
    
    return spatial_fft

