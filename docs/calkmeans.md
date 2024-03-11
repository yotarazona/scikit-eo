<!-- markdownlint-disable -->

<a href="..\scikeo\calkmeans.py#L0"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

# <kbd>module</kbd> `calkmeans`





---

<a href="..\scikeo\calkmeans.py#L7"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>function</kbd> `calkmeans`

```python
calkmeans(
    image,
    k=None,
    algo=('auto', 'elkan'),
    max_iter=300,
    n_iter=10,
    nodata=-99999,
    **kwargs
)
```

Calibrating kmeans 

This function allows to calibrate the kmeans algorithm. It is possible to obtain the best 'k' value and the best embedded algorithm in KMmeans.  



**Parameters:**
 
 - <b>`image`</b>:  Optical images. It must be rasterio.io.DatasetReader with 3d.  


 - <b>`k`</b>:  k This argument is None when the objective is to obtain the best 'k' value.   f the objective is to select the best algorithm embedded in kmeans, please specify a 'k' value. 


 - <b>`max_iter`</b>:  The maximum number of iterations allowed. Strictly related to KMeans. Please see 
 - <b>`https`</b>: //scikit-learn.org/stable/modules/generated/sklearn.cluster.KMeans.html 


 - <b>`algo`</b>:  It can be "auto" and 'elkan'. "auto" and "full" are deprecated and they will be   removed in Scikit-Learn 1.3. They are both aliases for "lloyd".  


 - <b>`Changed in version 1.1`</b>:  Renamed “full” to “lloyd”, and deprecated “auto” and “full”.   Changed “auto” to use “lloyd” instead of “elkan”.  


 - <b>`n_iter`</b>:  Iterations number to obtain the best 'k' value. 'n_iter' must be greater than the   number of classes expected to be obtained in the classification. Default is 10.  


 - <b>`nodata`</b>:  The NoData value to replace with -99999.   


 - <b>`**kwargs`</b>:  These will be passed to scikit-learn KMeans, please see full lists at: 
 - <b>`https`</b>: //scikit-learn.org/stable/modules/generated/sklearn.cluster.KMeans.html. 

Return: Labels of classification as numpy object with 2d. 



**Note:**

> If the idea is to find the optimal value of 'k' (clusters or classes), k = None as an argument of the function must be put, because the function find 'k' for which the intra-class inertia is stabilized. If the 'k' value is known and the idea is to find the best algorithm embedded in kmeans (that maximizes inter-class distances), k = n, which 'n' is a specific class number, must be put. It can be greater than or equal to 0. 
>




---

_This file was automatically generated via [lazydocs](https://github.com/ml-tooling/lazydocs)._
