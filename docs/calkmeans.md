<!-- markdownlint-disable -->

<a href="..\scikeo\calkmeans.py#L0"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

# <kbd>module</kbd> `calkmeans`





---

<a href="..\scikeo\calkmeans.py#L8"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>function</kbd> `calkmeans`

```python
calkmeans(
    image: DatasetReader,
    k: Optional[int] = None,
    algo: Tuple[str, ] = ('lloyd', 'elkan'),
    max_iter: int = 300,
    n_iter: int = 10,
    nodata: float = -99999.0,
    **kwargs
) → Dict[str, List[float]]
```

Calibrating KMeans Clustering Algorithm 

This function calibrates the KMeans algorithm for satellite image classification. It can either find the optimal number of clusters (k) by evaluating the inertia over a range of cluster numbers or determine the best algorithm ('lloyd' or 'elkan') for a given k. 



**Parameters:**
 
 - <b>`image`</b> (rasterio.io.DatasetReader):  Optical image with 3D data. 
 - <b>`k`</b> (Optional[int]):  The number of clusters. If None, the function finds the optimal k. 
 - <b>`algo`</b> (Tuple[str, ...]):  Algorithms to evaluate ('lloyd', 'elkan'). 
 - <b>`max_iter`</b> (int):  Maximum iterations for KMeans (default 300). 
 - <b>`n_iter`</b> (int):  Number of iterations or clusters to evaluate (default 10). 
 - <b>`nodata`</b> (float):  The NoData value to identify and handle in the data. 
 - <b>`**kwargs`</b>:  Additional arguments passed to sklearn.cluster.KMeans. 



**Returns:**
 
 - <b>`Dict[str, List[float]]`</b>:  A dictionary with algorithm names as keys and lists of  inertia values as values. 



**Notes:**

> - If k is None, the function evaluates inertia for cluster numbers from 1 to n_iter. - If k is provided, the function runs KMeans n_iter times for each algorithm to evaluate their performance. - The function handles NoData values using fuzzy matching to account for floating-point precision. 




---

_This file was automatically generated via [lazydocs](https://github.com/ml-tooling/lazydocs)._
