# +
import rasterio
import numpy as np

def sma(image, endmembers, nodata = -99999):
    
    '''
    The SMA assumes that the energy received within the field of vision of the remote sensor 
    can be considered as the sum of the energies received from each dominant endmember. 
    This function addresses a Linear Mixing Model.
    
    A regression analysis is used to obtain the fractions. In least squares inversion algorithms, 
    the common objective is to estimate abundances that minimize the squared error between the 
    actual spectrum and the estimated spectrum. The values of the fractions will be between 0 and 1.
    
    Parameters:
    
        image: Optical images. It must be rasterio.io.DatasetReader with 3d.
        
        endmembers: Endmembers must be a matrix (numpy.ndarray) and with more than one endmember. 
                    Rows represent the endmembers and columns represent the spectral bands.
                    The number of bands must be greater than the number of endmembers.
                    E.g. an image with 6 bands, endmembers dimension should be $n*6$, where $n$ 
                    is rows with the number of endmembers and 6 is the number of bands 
                    (should be equal).
                    
        nodata: The NoData value to replace with -99999.
    
    Return:
    
        numpy.ndarray with 2d.
        
    References:
    
        Adams, J. B., Smith, M. O., & Gillespie, A. R. (1993). Imaging spectroscopy:
        Interpretation based on spectral mixture analysis. In C. M. Pieters & P.
        Englert (Eds.), Remote geochemical analysis: Elements and mineralogical
        composition. NY: Cambridge Univ. Press 145-166 pp.
    
        Shimabukuro, Y.E. and Smith, J., (1991). The least squares mixing models to
        generate fraction images derived from remote sensing multispectral data.
        IEEE Transactions on Geoscience and Remote Sensing, 29, pp. 16-21.
    
    Note:
    A regression analysis is used to obtain the fractions. In least squares
    inversion algorithms, the common objective is to estimate abundances that
    minimize the squared error between the actual spectrum and the estimated spectrum.
    The values of the fractions will be between 0 and 1.
    '''
    
    if not isinstance(image, (rasterio.io.DatasetReader)):
        raise TypeError('"image" must be raster read by rasterio.open().')
        
    if not isinstance (endmembers, (np.ndarray)):
        raise ErrorType('"endmembers" must be numpy.ndarray.')
    
    bands = image.count
        
    rows = image.height
        
    cols = image.width
    
    # number of endmembers
    n_endm = endmembers.shape[0]
    
    # number of bands extracted for each endmember
    b_endm = endmembers.shape[1]
        
    st = image.read()
        
    # data in [rows, cols, bands]
    st_reorder = np.moveaxis(st, 0, -1) 
    # data in [rows*cols, bands]
    arr = st_reorder.reshape((rows*cols, bands))
    
    # nodata
    if np.isnan(np.sum(arr)):
        arr[np.isnan(arr)] = self.nodata
    
    if not arr.shape[1] > n_endm:
        raise ValueError('The number of bands must be greater than the number of endmembers.')
    
    if not arr.shape[1] == b_endm:
        raise ValueError('The number of values extracted in band should be equal.')
    
    M = np.transpose(endmembers)
    
    mat_oper = np.dot(np.linalg.inv(np.dot(np.transpose(M), M)), np.transpose(M)) 
    
    frac = np.zeros((rows*cols, n_endm))
                
    for i in np.arange(0, n_endm, 1):
        for j in np.arange(0, rows*cols, 1):
            f = np.dot(mat_oper, arr[j,:])
            frac[j,i] = f[i,]
    
    sma_img = frac.reshape((rows, cols, n_endm))
    
    return sma_img
