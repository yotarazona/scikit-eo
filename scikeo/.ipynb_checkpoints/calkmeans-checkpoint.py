# -*- coding: utf-8 -*-
# +
from sklearn.cluster import KMeans
import rasterio
import numpy as np

def calkmeans(image, k = None, algo = ("auto", "elkan"), max_iter = 300, n_iter = 10, nodata = -99999, **kwargs):
    
    '''
    Calibrating kmeans
    
    This function allows to calibrate the kmeans algorithm. It is possible to obtain
    the best 'k' value and the best embedded algorithm in KMmeans.
        
    Parameters:
        image: Optical images. It must be rasterio.io.DatasetReader with 3d.
            
        k: k This argument is None when the objective is to obtain the best 'k' value. 
           f the objective is to select the best algorithm embedded in kmeans, please specify a 'k' value.
    
        max_iter: The maximum number of iterations allowed. Strictly related to KMeans. Please see
                  https://scikit-learn.org/stable/modules/generated/sklearn.cluster.KMeans.html
            
        algo: It can be "auto" and 'elkan'. "auto" and "full" are deprecated and they will be 
              removed in Scikit-Learn 1.3. They are both aliases for "lloyd".
            
        Changed in version 1.1: Renamed “full” to “lloyd”, and deprecated “auto” and “full”. 
                               Changed “auto” to use “lloyd” instead of “elkan”.
            
        n_iter: Iterations number to obtain the best 'k' value. 'n_iter' must be greater than the 
                number of classes expected to be obtained in the classification. Default is 10.
            
        nodata: The NoData value to replace with -99999. 
                     
        **kwargs: These will be passed to scikit-learn KMeans, please see full lists at:
                  https://scikit-learn.org/stable/modules/generated/sklearn.cluster.KMeans.html.
    
        Return:
            Labels of classification as numpy object with 2d.
            
        Note:
        If the idea is to find the optimal value of 'k' (clusters or classes), k = None as an 
        argument of the function must be put, because the function find 'k' for which the intra-class 
        inertia is stabilized. If the 'k' value is known and the idea is to find the best algorithm 
        embedded in kmeans (that maximizes inter-class distances), k = n, which 'n' is a specific 
        class number, must be put. It can be greater than or equal to 0.  
        
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
    
    if k is None:
        
        if 'auto' in algo:
            
            inertias_intra_lloyd_k = []
    
            for i in range(1, n_iter + 1):
    
                # Building and fitting the model
                kmeanModel = KMeans(n_clusters = i, max_iter = max_iter, algorithm = 'auto', **kwargs)
    
                kmeanModel.fit(arr)

                inertias_intra_lloyd_k.append(kmeanModel.inertia_)
        
        if 'elkan' in algo:
            
            inertias_intra_elkan_k = []
    
            for i in range(1, n_iter + 1):
    
                # Building and fitting the model
                kmeanModel = KMeans(n_clusters = i, max_iter = max_iter, algorithm = 'elkan', **kwargs)
    
                kmeanModel.fit(arr)

                inertias_intra_elkan_k.append(kmeanModel.inertia_)
        
        dic = {'inertias_intra_lloyd_k': inertias_intra_lloyd_k,
               'inertias_intra_elkan_k': inertias_intra_elkan_k}

        all_algorithms = ("auto", "elkan")

        for model in all_algorithms:
            if model not in algo:
                del dic[model]
                    
        find_k = dic
        
        return find_k
    
    # nc = range(1,10)
    # kmeanlist = [KMeans(n_clusters = i) for i in Nc]
    # variance = [kmeanlist[i].fit(datos).inertia_ for i in range(len(kmeanlist))]
    
    elif isinstance(k, (int)):
        
        if 'auto' in algo:
            
            inertias_inter_lloyd = []
    
            for i in range(n_iter):
    
                kmeanModel = KMeans(n_clusters = k, max_iter = max_iter, algorithm = 'auto', **kwargs)
    
                kmeanModel.fit(arr)

                inertias_inter_lloyd.append(kmeanModel.score(arr))
        
        if 'elkan' in algo:
            
            inertias_inter_elkan = []
    
            for i in range(n_iter):
    
                kmeanModel = KMeans(n_clusters = k, max_iter = max_iter, algorithm = 'elkan', **kwargs)
    
                kmeanModel.fit(arr)

                inertias_inter_elkan.append(kmeanModel.inertia_)
        
        dic = {'inertias_inter_lloyd': inertias_inter_lloyd,
               'inertias_inter_elkan': inertias_inter_elkan}

        all_algorithms = ("auto", "elkan")

        for model in all_algorithms:
            if model not in algo:
                del dic[model]
                    
        best_algorithm = dic
        
        return best_algorithm
    
    else:
        raise TypeError(f'{k} must be None or positive integer. type = {type(k)}')
