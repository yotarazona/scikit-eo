<!-- markdownlint-disable -->

<a href="..\scikeo\sma.py#L0"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

# <kbd>module</kbd> `sma`





---

<a href="..\scikeo\sma.py#L5"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>function</kbd> `sma`

```python
sma(image, endmembers, nodata=-99999)
```

The SMA assumes that the energy received within the field of vision of the remote sensor  can be considered as the sum of the energies received from each dominant endmember.  This function addresses a Linear Mixing Model. 

A regression analysis is used to obtain the fractions. In least squares inversion algorithms,  the common objective is to estimate abundances that minimize the squared error between the  actual spectrum and the estimated spectrum. The values of the fractions will be between 0 and 1. 



**Parameters:**
 


 - <b>`image`</b>:  Optical images. It must be rasterio.io.DatasetReader with 3d. 


 - <b>`endmembers`</b>:  Endmembers must be a matrix (numpy.ndarray) and with more than one endmember.   Rows represent the endmembers and columns represent the spectral bands.  The number of bands must be greater than the number of endmembers.  E.g. an image with 6 bands, endmembers dimension should be $n*6$, where $n$   is rows with the number of endmembers and 6 is the number of bands   (should be equal).  


 - <b>`nodata`</b>:  The NoData value to replace with -99999. 

Return: 

numpy.ndarray with 2d. 

References: 

Adams, J. B., Smith, M. O., & Gillespie, A. R. (1993). Imaging spectroscopy: Interpretation based on spectral mixture analysis. In C. M. Pieters & P. 
 - <b>`Englert (Eds.), Remote geochemical analysis`</b>:  Elements and mineralogical 
 - <b>`composition. NY`</b>:  Cambridge Univ. Press 145-166 pp. 

Shimabukuro, Y.E. and Smith, J., (1991). The least squares mixing models to generate fraction images derived from remote sensing multispectral data. IEEE Transactions on Geoscience and Remote Sensing, 29, pp. 16-21. 



**Note:**

> A regression analysis is used to obtain the fractions. In least squares inversion algorithms, the common objective is to estimate abundances that minimize the squared error between the actual spectrum and the estimated spectrum. The values of the fractions will be between 0 and 1. 




---

_This file was automatically generated via [lazydocs](https://github.com/ml-tooling/lazydocs)._
