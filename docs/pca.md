<!-- markdownlint-disable -->

<a href="..\scikeo\pca.py#L0"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

# <kbd>module</kbd> `pca`





---

<a href="..\scikeo\pca.py#L9"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>function</kbd> `PCA`

```python
PCA(image, stand_varb=True, nodata=-99999, **kwargs)
```

Runing Principal Component Analysis (PCA) with satellite images. 

This algorithm allows to obtain Principal Components from images either radar or optical coming from different spectral sensors. It is also possible to obtain the contribution (%)  of each variable. 



**Parameters:**
 


 - <b>`images`</b>:  Optical or radar image, it must be rasterio.io.DatasetReader with 3d. 


 - <b>`stand_varb`</b>:  Logical. If ``stand.varb = True``, the PCA is calculated using the correlation   matrix (standardized variables) instead of the covariance matrix   (non-standardized variables).  


 - <b>`nodata`</b>:  The NoData value to replace with -99999. 


 - <b>`**kwargs`</b>:  These will be passed to scikit-learn PCA, please see full lists at: 
 - <b>`https`</b>: //scikit-learn.org/stable/modules/generated/sklearn.decomposition.PCA.html 

Return: 

A dictionary. 



**Note:**

> The contributions of variables in accounting for the variability in a given principal component are expressed in percentage. Variables that are correlated with PC1 (i.e., Dim.1) and PC2 (i.e., Dim.2) are the most important in explaining the variability in the data set. Variables that do not correlated with any PC or correlated with the last dimensions are variables with low contribution and might be removed to simplify the overall analysis. The contribution is a scaled version of the squared correlation between variables and component axes (or the cosine, from a geometrical point of view) --- this is used to assess the quality of the representation of the variables of the principal component, and it is computed as (cos(variable,axis)^2/total cos2 of the component)×100. 




---

_This file was automatically generated via [lazydocs](https://github.com/ml-tooling/lazydocs)._
