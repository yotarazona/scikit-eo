<!-- markdownlint-disable -->

<a href="..\scikeo\rkmeans.py#L0"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

# <kbd>module</kbd> `rkmeans`





---

<a href="..\scikeo\rkmeans.py#L6"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>function</kbd> `rkmeans`

```python
rkmeans(image, k, nodata=-99999, **kwargs)
```

This function allows to classify satellite images using k-means 

In principle, this function allows to classify satellite images specifying a ```k``` value (clusters), however it is recommended to find the optimal value of ```k``` using the ```calkmeans``` function embedded in this package.  



**Parameters:**
 


 - <b>`image`</b>:  Optical images. It must be rasterio.io.DatasetReader with 3d. 


 - <b>`k`</b>:  The number of clusters to be detected. 


 - <b>`nodata`</b>:  The NoData value to replace with -99999.  


 - <b>`**kwargs`</b>:  These will be passed to scikit-learn KMeans, please see full lists at: 
 - <b>`https`</b>: //scikit-learn.org/stable/modules/generated/sklearn.cluster.KMeans.html 

Return: 

Labels of classification as numpy object with 2d. 




---

_This file was automatically generated via [lazydocs](https://github.com/ml-tooling/lazydocs)._
