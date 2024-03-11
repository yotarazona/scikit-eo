<!-- markdownlint-disable -->

<a href="..\scikeo\linearTrend.py#L0"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

# <kbd>module</kbd> `linearTrend`






---

<a href="..\scikeo\linearTrend.py#L8"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>class</kbd> `linearTrend`
Linear Trend in Remote Sensing 

<a href="..\scikeo\linearTrend.py#L12"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `__init__`

```python
__init__(image, nodata=-99999)
```



**Parameters:**
 


 - <b>`image`</b>:  Optical images. It must be rasterio.io.DatasetReader with 3d. 


 - <b>`nodata`</b>:  The NoData value to replace with -99999. 






---

<a href="..\scikeo\linearTrend.py#L26"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `LN`

```python
LN(**kwargs)
```

Linear trend is useful for mapping forest degradation, land degradation, etc. This algorithm is capable of obtaining the slope of an ordinary least-squares  linear regression and its reliability (p-value). 



**Parameters:**
 


 - <b>`**kwargs`</b>:  These will be passed to LN, please see full lists at: 
 - <b>`https`</b>: //docs.scipy.org/doc/scipy/reference/generated/scipy.stats.linregress.html 

Return: a dictionary with slope, intercept and p-value obtained. All of them in numpy.ndarray  with 2d. 

References: 

Tarazona, Y., Maria, Miyasiro-Lopez. (2020). Monitoring tropical forest degradation using remote sensing. Challenges and opportunities in the Madre de Dios region, Peru. Remote 
 - <b>`Sensing Applications`</b>:  Society and Environment, 19, 100337. 

Wilkinson, G.N., Rogers, C.E., 1973. Symbolic descriptions of factorial models for analysis of variance. Appl. Stat. 22, 392-399. 

Chambers, J.M., 1992. Statistical Models in S. CRS Press. 



**Note:**

> Linear regression is widely used to analyze forest degradation or land degradation. Specifically, the slope and its reliability are used as main parameters and they can be obtained with this function. On the other hand, logistic regression allows obtaining a degradation risk map, in other words, it is a probability map. 

---

<a href="..\scikeo\linearTrend.py#L104"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `LR`

```python
LR(col_pos=0, **kwargs)
```

Logistic Regression is a statistical analysis technique that can measure  statistically the relative influence of several factors and explain objectively how values  depend on predictor variables. This method is applied to remotely sensed data. 



**Parameters:**
 


 - <b>`**kwargs`</b>:  These will be passed to MLN, please see full lists at: 
 - <b>`https`</b>: //www.statsmodels.org/dev/generated/statsmodels.discrete.discrete_model.Logit.html 

Return: a dictionary with the summary of logistic regression and an array of probability with 2d. 

References: Tarazona, Y., Maria, Miyasiro-Lopez. (2020). Monitoring tropical forest degradation using remote sensing. Challenges and opportunities in the Madre de Dios region, Peru. Remote 
 - <b>`Sensing Applications`</b>:  Society and Environment, 19, 100337. 

Chambers, J.M., 1992. Statistical Models in S. CRS Press. 



**Note:**

> Logistic regression allows obtaining a degradation risk map (for instance), in other words, it is a probability map. 




---

_This file was automatically generated via [lazydocs](https://github.com/ml-tooling/lazydocs)._
