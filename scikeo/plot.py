# -*- coding: utf-8 -*-
# +
import rasterio
import numpy as np
import matplotlib.pyplot as plt

def plotHist(image, bands = 1, bins = 128, alpha = 0.8, title = None, xlabel = None, ylabel = None,
             label = None, ax = None, density = True, **kwargs):
    
    '''
    This function allows to plot satellite images histogram.
    
    Parameters:
        image: Optical images. It must be rasterio.io.DatasetReader with 3d or 2d.
            
        bands: Must be specified as a number of a list.
            
        bins: By default is 128. 
            
        alpha: Percentage (%) of transparency between 0 and 1. 0 indicates 0% and 1 indicates
               100%. By default is 80%.
            
        title: Assigned title.
        
        xlabel: X axis title.
        
        ylabel: Y axis title.
            
        label: Labeling the histogram.
        
        ax: current axes
        
        **kwargs: These will be passed to the matplotlib imshow(), please see full lists at:
                https://matplotlib.org/stable/api/_as_gen/matplotlib.axes.Axes.hist.html
    
    Return:
        ax: A histogram of an image.
        
    '''
    
    if not isinstance(image, (rasterio.io.DatasetReader)):
        raise TypeError('"image" must be raster read by rasterio.open().')
    
    st = image.read()
    
    # data in [rows, cols, bands]
    arr = np.moveaxis(st, 0, -1)
    
    if ax is None:
        ax = plt.gca() # get current axes
    
    nbands = np.array([bands]).flatten()
            
    # xlabel
    if label is None:
        label = {i:f'band {i}' for i in nbands}
    else:
        nlabel = np.array([label]).flatten()
        
        if not len(nbands) == len(nlabel):
            raise TypeError('Length of "bands" must be equal to "label".')
        
        label = {b:l for b,l in zip(nbands, nlabel)}
        
    for i in nbands:
        
        # Draw the plot
        arr_1d = arr[:, :, i].flatten()
        
        ax.hist(arr_1d, bins = bins, alpha = alpha, label = label[i], density = density, **kwargs)
    
    # title
    if title is None: title = "Histogram"
    
    # ylabel
    if ylabel is None: ylabel = "Frequency"
    
    # xlabel
    if xlabel is None: xlabel = "Band values"
        
    # Title and labels
    ax.set(title = title, xlabel = xlabel, ylabel = ylabel)

    plt.legend()
    plt.tight_layout()
    
    return ax


def plotRGB(image, bands = [3,2,1], stretch = 'std', title = None, xlabel = None, ylabel = None, 
            ax = None, **kwargs):
    
    '''
    Plotting an image in RGB.
    
    This function allows to plot an satellite image in RGB channels.
        
    Parameters:
            
        image: Optical images. It must be rasterio.io.DatasetReader with 3d.
            
        bands: A list contain the order of bands to be used in order to plot in RGB. For example,
                   for six bands (blue, green, red, nir, swir1 and swir2), number four (4) indicates 
                   the swir1 band, number three (3) indicates the nir band and the number two (2) indicates
                   the red band.
                   
        stretch: Contrast enhancement using the histogram. There are two options here: i) using
                     standard deviation ('std') and ii) using percentiles ('per'). For default is 'std', which means
                     standard deviation.
            
        title: Assigned title.
        
        xlabel: X axis title.
        
        ylabel: Y axis title.
        
        ax: current axes
        
        **kwargs: These will be passed to the matplotlib imshow(), please see full lists at:
                https://matplotlib.org/stable/api/_as_gen/matplotlib.pyplot.imshow.html
    
    Return:
        ax: Graphic of an image in RGB.
        
    '''
    
    if not isinstance(image, (rasterio.io.DatasetReader)):
        raise TypeError('"image" must be raster read by rasterio.open().')
        
    st = image.read()
    
    if st.shape[0] < 3:
        raise TypeError('"image" must be a raster with at least three bands.')
    
    # data in [rows, cols, bands]
    st = np.moveaxis(st, 0, -1) 
    
    bands = bands
    
    arr_rgb = np.dstack([st[:, :, (bands[0]-1)], st[:, :, (bands[1]-1)], st[:, :, (bands[2]-1)]])
    
    if stretch == 'std':
        
        mean = np.nanmean(arr_rgb)
        
        std = np.nanstd(arr_rgb)*1.5
        
        min_val = np.nanmax([mean - std, np.nanmin(arr_rgb)])
        
        max_val = np.nanmin([mean + std, np.nanmax(arr_rgb)])
        
        clipped_arr = np.clip(arr_rgb, min_val, max_val)
        
        arr_rgb_norm = (clipped_arr - min_val)/(max_val - min_val)

    elif stretch == 'per':
        
        p10 = np.nanpercentile(arr_rgb, 10) # percentile10
        
        p90 = np.nanpercentile(arr_rgb, 90) # percentile90
        
        clipped_arr = np.clip(arr_rgb, p10, p90)
        
        arr_rgb_norm = (clipped_arr - p10)/(p90 - p10)
        
    else:
        raise TypeError('Stretch type is not implemented. Please select either "std" or "per".')

    if ax is None:
        ax = plt.gca()

    ax.imshow(arr_rgb_norm, **kwargs)
    ax.grid(False)
    
    # title
    if title is not None:
        ax.set_title(title)
    
    # ylabel
    if ylabel is not None:
        ax.set_ylabel(ylabel)
    
    # xlabel
    if xlabel is not None:
        ax.set_xlabel(xlabel)
    
    #ax.set_axis_off()

    return ax
