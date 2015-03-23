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
    space = np.linspace(-spatialDelta*data_h.shape[2]/2., spatialDelta*data_h.shape[2]/2., data_h.shape[2])
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


