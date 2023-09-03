# +
import numpy as np
from scipy import stats
import rasterio
import pandas as pd
import statsmodels.api as sm

class linearTrend(object):
    
    "Linear Trend in Remote Sensing"
    
    def __init__(self, image, nodata = -99999):
        
        '''
        Parameters:
    
            image: Optical images. It must be rasterio.io.DatasetReader with 3d.
            
            nodata: The NoData value to replace with -99999.
            
        '''
        
        self.image = image
        self.nodata = nodata
    
    def LN(self, **kwargs):
        
        '''
        Linear trend is useful for mapping forest degradation, land degradation, etc.
        This algorithm is capable of obtaining the slope of an ordinary least-squares 
        linear regression and its reliability (p-value).
    
        Parameters:
        
            **kwargs: These will be passed to LN, please see full lists at:
                  https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.linregress.html
                  
        Return:
            a dictionary with slope, intercept and p-value obtained. All of them in numpy.ndarray 
            with 2d.
        
        References:
        
            Tarazona, Y., Maria, Miyasiro-Lopez. (2020). Monitoring tropical forest degradation using
            remote sensing. Challenges and opportunities in the Madre de Dios region, Peru. Remote
            Sensing Applications: Society and Environment, 19, 100337.
        
            Wilkinson, G.N., Rogers, C.E., 1973. Symbolic descriptions of factorial models for
            analysis of variance. Appl. Stat. 22, 392-399.
        
            Chambers, J.M., 1992. Statistical Models in S. CRS Press.
        
        Note:
        Linear regression is widely used to analyze forest degradation or land degradation.
        Specifically, the slope and its reliability are used as main parameters and they
        can be obtained with this function. On the other hand, logistic regression allows
        obtaining a degradation risk map, in other words, it is a probability map.
        
        '''
        
        if not isinstance(self.image, (rasterio.io.DatasetReader)):
            raise TypeError(f'"image" must be raster read by rasterio.open(). {type(self.image)}')
        
        if not self.image.count >= 2:
            raise ValueError(f'The number of bands must be greater than 2. shape = {self.image.count}')
            
        bands = self.image.count
        
        rows = self.image.height
        
        cols = self.image.width
        
        st = self.image.read()
        
        st_reorder = np.moveaxis(st, 0, -1)  # rows, cols and bands
        
        arr = st_reorder.reshape((rows*cols, bands))
        
        # nodata
        if np.isnan(np.sum(arr)):
            arr[np.isnan(arr)] = self.nodata
        
        sequences = range(0, rows*cols)
        
        x = range(0, bands)
        
        lr = map(lambda i: stats.linregress(x, arr[i, :], **kwargs), sequences)
        
        lr_coef = np.stack(list(lr))
        
        slope = lr_coef[:,0].reshape((rows, cols))
        
        intercept = lr_coef[:,1].reshape((rows, cols))
        
        pvalue = lr_coef[:,3].reshape((rows, cols))
        
        result = {'slope': slope,
                 'intercept': intercept,
                 'pvalue': pvalue
                 }
        
        return result
    
    def LR(self, col_pos = 0, **kwargs):
        
        '''
        Logistic Regression is a statistical analysis technique that can measure 
        statistically the relative influence of several factors and explain objectively how values 
        depend on predictor variables. This method is applied to remotely sensed data.
    
        Parameters:
        
            **kwargs: These will be passed to MLN, please see full lists at:
                  https://www.statsmodels.org/dev/generated/statsmodels.discrete.discrete_model.Logit.html
                  
        Return:
            a dictionary with the summary of logistic regression and an array of probability with 2d.
            
        References:
            Tarazona, Y., Maria, Miyasiro-Lopez. (2020). Monitoring tropical forest degradation using
            remote sensing. Challenges and opportunities in the Madre de Dios region, Peru. Remote
            Sensing Applications: Society and Environment, 19, 100337.
        
            Chambers, J.M., 1992. Statistical Models in S. CRS Press.
        
        Note:
        Logistic regression allows obtaining a degradation risk map (for instance), in other words, 
        it is a probability map.
        '''
        
        if not isinstance(self.image, (rasterio.io.DatasetReader)):
            raise TypeError(f'"image" must be raster read by rasterio.open(). {type(self.image)}')
        
        if not self.image.count >= 2:
            raise ValueError(f'The number of bands must be greater than 2. shape = {self.image.count}')
            
        bands = self.image.count
        
        rows = self.image.height
        
        cols = self.image.width
        
        st = self.image.read()
        
        st_reorder = np.moveaxis(st, 0, -1)  # rows, cols and bands
        
        arr = st_reorder.reshape((rows*cols, bands))
        
        # nodata
        if np.isnan(np.sum(arr)):
            arr[np.isnan(arr)] = self.nodata
        
        st_reorder = np.moveaxis(st, 0, -1)  # rows, cols and bands
        
        arr = st_reorder.reshape((rows*cols, bands))

        DF = pd.DataFrame(arr, columns = [f'var{i}' for i in range(1, bands + 1)])

        list_ind = [i for i in range(bands)]

        list_ind.remove(col_pos)

        Xtrain = DF.iloc[:, list_ind]

        ytrain = DF.iloc[:, col_pos]

        log_reg = sm.Logit(ytrain, Xtrain).fit()

        summary_logit = log_reg.summary()

        pred_prob = log_reg.predict(Xtrain)

        pred_prob = pred_prob.to_numpy().reshape((rows, cols))

        result = {'summary_logit': summary_logit,
                  'arr_prob': pred_prob,
                 }
        
        return result
