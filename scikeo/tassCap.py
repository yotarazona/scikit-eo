# +
import numpy as np
import rasterio

def tassCap(image, sat = "Landsat8OLI", nodata = -99999, scale = None):
    
    '''
    The Tasseled-Cap Transformation is a linear transformation method for various 
    remote sensing data. Not only can it perform volume data compression, but it
    can also provide parameters associated with the physical characteristics, 
    such as brightness, greenness and wetness indices.
    
    Parameters:
    
        image: Optical images. It must be rasterio.io.DatasetReader with 3d.
        
        sat: Specify satellite and sensor type (Landsat5TM, Landsat7ETM or Landsat8OLI).
                    
        nodata: The NoData value to replace with -99999.
        
        scale: Conversion of coefficients values
    
    Return:
        numpy.ndarray with 3d containing brightness, greenness and wetness indices.
    
    References:
    
        Crist, E.P., R. Laurin, and R.C. Cicone. 1986. Vegetation and soils information 
        contained in transformed Thematic Mapper data. Pages 1465-1470 Ref. ESA SP-254. 
        European Space Agency, Paris, France. http://www.ciesin.org/docs/005-419/005-419.html.

        Baig, M.H.A., Shuai, T., Tong, Q., 2014. Derivation of a tasseled cap transformation 
        based on Landsat 8 at-satellite reflectance. Remote Sensing Letters, 5(5), 423-431. 
    
        Li, B., Ti, C., Zhao, Y., Yan, X., 2016. Estimating Soil Moisture with Landsat Data 
        and Its Application in Extracting the Spatial Distribution of Winter Flooded Paddies. 
        Remote Sensing, 8(1), 38.
    
    Note:
    Currently implemented for satellites such as Landsat-4 TM, Landsat-5 TM, Landsat-7 ETM+, 
    Landsat-8 OLI and Sentinel2. The input data must be in top of atmosphere reflectance (toa). 
    Bands required as input must be ordered as:
    
    Consider using the following satellite bands:
    ===============   ================================
    Type of Sensor     Name of bands
    ===============   ================================
    Landsat4TM         :blue, green, red, nir, swir1, swir2
    Landsat5TM         :blue, green, red, nir, swir1, swir2
    Landsat7ETM+       :blue, green, red, nir, swir1, swir2
    Landsat8OLI        :blue, green, red, nir, swir1, swir2
    Landsat8OLI-Li2016 :coastal, blue, green, red, nir, swir1, swir2
    Sentinel2MSI       :coastal, blue, green, red, nir-1, mir-1, mir-2
    '''
    
    if not isinstance(image, (rasterio.io.DatasetReader)):
        raise TypeError('"image" must be raster read by rasterio.open().')
        
    if sat == 'Landsat4TM':
        coefc = np.array([[0.3037, 0.2793, 0.4743, 0.5585, 0.5082, 0.1863], # brightness
                         [-0.2848, -0.2435, -0.5436, 0.7243, 0.0840, -0.1800], # greenness
                         [0.1509, 0.1973, 0.3279, 0.3406, -0.7112, -0.4572]]) # wetness
    
    elif sat == 'Landsat5TM':
        coefc = np.array([[0.2909, 0.2493, 0.4806, 0.5568, 0.4438, 0.1706],
                         [-0.2728, -0.2174, -0.5508, 0.7221, 0.0733, -0.1648],
                         [0.1446, 0.1761, 0.3322, 0.3396, -0.6210, -0.4186]])
    
    elif sat == 'Landsat7ETM':
        coefc = np.array([[0.3561, 0.3972, 0.3904, 0.6966, 0.2286, 0.1596],
                         [-0.3344, -0.3544, -0.4556, 0.6966, -0.0242, -0.2630],
                         [0.2626, 0.2141, 0.0926, 0.0656, -0.7629, 0.5388]])
    
    elif sat == 'Landsat8OLI':
        coefc = np.array([[0.3029, 0.2786, 0.4733, 0.5599, 0.5080, 0.1872],
                         [-0.2941, -0.2430, -0.5424, 0.7276, 0.0713, -0.1608],
                         [0.1511, 0.1973, 0.3283, 0.3407, -0.7117, -0.4559]])
    
    elif sat == 'Landsat8OLI-Li2016':
        coefc = np.array([[0.2540, 0.3037, 0.3608, 0.3564, 0.7084, 0.2358, 0.1691],
                         [-0.2578, -0.3064, -0.3300, -0.4325, 0.6860, -0.0383, -0.2674],
                         [0.1877, 0.2097, 0.2038, 0.1017, 0.0685, -0.7460, -0.5548]])
        
    elif sat == 'Sentinel2MSI':
        coefc = np.array([[0.2381, 0.2569, 0.2934, 0.3020, 0.3580, 0.0896, 0.0780],
                         [-0.2266, -0.2818, -0.3020, -0.4283, 0.3138, -0.1341, -0.2538],
                         [0.1825, 0.1763, 0.1615, 0.0486, -0.0755, -0.7701, -0.5293]])
    else:
        raise ValueError('Satellite not supported. Please see the list of satellites mentioned'
                         'in docstrings.')
        
    if not scale is None:
        coefc = coefc*scale
    
    bands = image.count
        
    rows = image.height
        
    cols = image.width
        
    st = image.read()
        
    # data in [rows, cols, bands]
    st_reorder = np.moveaxis(st, 0, -1) 
    # data in [rows*cols, bands]
    arr = st_reorder.reshape((rows*cols, bands))
    
    # nodata
    if np.isnan(np.sum(arr)):
        arr[np.isnan(arr)] = self.nodata
        
    if bands != coefc.shape[1]:
        raise ValueError('The number of bands must be equal to the number of coefficients in bands.')
    
    bgw = np.dot(arr, np.transpose(coefc))
    
    bgw = bgw.reshape((rows, cols, coefc.shape[0]))

    return bgw
