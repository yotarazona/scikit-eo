# +
import rasterio
import numpy as np
import os

def writeRaster(arr, image, filename = None, filepath = None, n = 1):
    
    '''
    This algorithm allows to save array images to raster format (.tif).
    
    Parameters:
    
        arr: Array object with 2d (rows and cols) or 3d (rows, cols, bands).
        
        image: Optical images. It must be read by rasterio.open().
        
        filename: The image name to be saved.
        
        filepath: The path which the image will be stored.
        
        n: Number of images to be saved.
    
    Return:
        A raster in your filepath.
    '''
    
    if not isinstance(arr, (np.ndarray)):
        raise TypeError(f'{arr} must be an array with 2d (rows and cols).')
    
    if not isinstance(image, (rasterio.io.DatasetReader)): 
        raise TypeError(f'The image must be rasterio.io.DatasetReader. image = {type(image)}')
    
    if not arr.ndim >= 2:
        raise ValueError(f'Input mus have two or three dimensions. ndim = {arr.ndim}')
    
    if filename is None:
        
        filename = 'raster'
    
    if filepath is None:
        
        current_path = os.getcwd()
        
        path = current_path.replace(os.sep, '/') + '/' + filename + '.tif'
        
    else:
        path = filepath.replace(os.sep, '/') + '/' + filename + '.tif'
    
    path_name = path
    
    if arr.ndim == 2:
        n = 1
    else:
        n = arr.shape[2] # number of bands >= 2
        
    kwargs = image.meta
    
    if arr.dtype == 'int64':
        arr = arr.astype(np.int32)
    
    dtype = str(arr.dtype)
    
    kwargs.update({      
        'dtype': dtype,
        'count': n
    })
    
    if arr.ndim == 2:
        with rasterio.open(path_name, 'w', compress='lzw', **kwargs) as img:    
            img.write_band(1, arr)
    else:
        with rasterio.open(path_name, 'w', compress='lzw', **kwargs) as img:
            for num, name in enumerate([arr[:,:,i] for i in range(n)], start=1):
                img.write_band(num, name)
