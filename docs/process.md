<!-- markdownlint-disable -->

<a href="..\scikeo\process.py#L0"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

# <kbd>module</kbd> `process`





---

<a href="..\scikeo\process.py#L12"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>function</kbd> `crop`

```python
crop(image, shp, filename=None, filepath=None)
```

This algorithm allows to clip a raster (.tif) including a satellite image using a shapefile. 



**Parameters:**
  


 - <b>`image`</b>:  This parameter can be a string with the raster path (e.g., r'/home/image/b3.tif') or it can be a rasterio.io.DatasetReader type. 


 - <b>`shp`</b>:  Vector file, tipically shapefile. 


 - <b>`filename`</b>:  The image name to be saved. 


 - <b>`filepath`</b>:  The path which the image will be stored. 

Return: 

A raster in your filepath. 


---

<a href="..\scikeo\process.py#L112"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>function</kbd> `extract`

```python
extract(image, shp)
```

This algorithm allows to extract raster values using a shapefile. 



**Parameters:**
  


 - <b>`image`</b>:  Optical images. It must be rasterio.io.DatasetReader with 3d or 2d. 


 - <b>`shp`</b>:  Vector file, tipically shapefile. 

Return: 

A dataframe with raster values obtained. 



**Note:**

> This function is usually used to extract raster values to be used on machine learning algorithms. 


---

<a href="..\scikeo\process.py#L158"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>function</kbd> `confintervalML`

```python
confintervalML(matrix, image_pred, pixel_size=10, conf=1.96, nodata=None)
```

The error matrix is a simple cross-tabulation of the class labels allocated by the classification of the remotely  sensed data against the reference data for the sample sites. The error matrix organizes the acquired sample data  in a way that summarizes key results and aids the quantification of accuracy and area. The main diagonal of the error  matrix highlights correct classifications while the off-diagonal elements show omission and commission errors.  The cell entries and marginal values of the error matrix are fundamental to both accuracy assessment and area  estimation. The cell entries of the population error matrix and the parameters derived from it must be estimated  from a sample. This function shows how to obtain a confusion matrix by estimated proportions of area with a confidence interval at 95% (1.96). 



**Parameters:**
 


 - <b>`matrix`</b>:  confusion matrix or error matrix in numpy.ndarray.  


 - <b>`image_pred`</b>:  Could be an array with 2d (rows, cols). This array should be the image classified   with predicted classes. Or, could be a list with number of pixels for each class.  


 - <b>`pixel_size`</b>:  Pixel size of the image classified. By default is 10m of Sentinel-2.  


 - <b>`conf`</b>:  Confidence interval. By default is 95%. 

Return: 

Information of confusion matrix by proportions of area, overall accuracy, user's accuracy with confidence interval  and estimated area with confidence interval as well.  

Reference: 

Olofsson, P., Foody, G.M., Herold, M., Stehman, S.V., Woodcock, C.E., and Wulder, M.A. 2014. “Good practices  
 - <b>`for estimating area and assessing accuracy of land change.” Remote Sensing of Environment, Vol. 148`</b>:  42–57.  
 - <b>`doi`</b>: https://doi.org/10.1016/j.rse.2014.02.015. 



**Note:**

> Columns and rows in a confusion matrix indicate reference and prediction respectively. 


---

<a href="..\scikeo\process.py#L315"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>function</kbd> `print_info`

```python
print_info(params)
```

Information: Confusion Matrix by Estimated Proportions of area an uncertainty 



**Parameters:**
 


 - <b>`params`</b>:  ```confintervalML``` result. See the function: https://github.com/ytarazona/scikit-eo/blob/main/scikeo/process.py 

Return: 

Information of confusion matrix by proportions of area, overall accuracy, user's accuracy with confidence interval  and estimated area with confidence interval as well. 



**Note:**

> This function was tested using ground-truth values obtained by Olofsson et al. (2014). 
>




---

_This file was automatically generated via [lazydocs](https://github.com/ml-tooling/lazydocs)._
