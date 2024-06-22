# +
from sklearn.cluster import KMeans
import rasterio
import numpy as np

def rkmeans(image, k, nodata = -99999, **kwargs):
    
    '''
    This function allows to classify satellite images using k-means
    
    In principle, this function allows to classify satellite images specifying
    a ```k``` value (clusters), however it is recommended to find the optimal value of ```k``` using
    the ```calkmeans``` function embedded in this package.
        
    Parameters:
    
        image: Optical images. It must be rasterio.io.DatasetReader with 3d.
        
        k: The number of clusters to be detected.
    
        nodata: The NoData value to replace with -99999.
                     
        **kwargs: These will be passed to scikit-learn KMeans, please see full lists at:
                  https://scikit-learn.org/stable/modules/generated/sklearn.cluster.KMeans.html
    
    Return:
        
        Labels of classification as numpy object with 2d.
    '''
    
    if not isinstance(image, (rasterio.io.DatasetReader)):
        raise TypeError('"image" must be raster read by rasterio.open().')
        
    bands = image.count
        
    rows = image.height
        
    cols = image.width
        
    # stack images
    #l = []
    #for i in np.arange(1, bands+1): l.append(image.read(int(i)))
    #st = np.stack(l)
    st = image.read()
    
    # data in [rows, cols, bands]
    st_reorder = np.moveaxis(st, 0, -1) 
    # data in [rows*cols, bands]
    arr = st_reorder.reshape((rows*cols, bands))
    
    # nodata
    if np.isnan(np.sum(arr)):
        arr[np.isnan(arr)] = self.nodata
    
    kmeans = KMeans(**kwargs) # max_iter=300 by default
    kmeans.fit(arr)
    
    labels_km = kmeans.labels_
    classKM = labels_km.reshape((rows, cols))
    
    return classKM
