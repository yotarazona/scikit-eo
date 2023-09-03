<!-- markdownlint-disable -->

<a href="..\scikeo\plot.py#L0"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

# <kbd>module</kbd> `plot`





---

<a href="..\scikeo\plot.py#L7"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>function</kbd> `plotHist`

```python
plotHist(
    image,
    bands=1,
    bins=128,
    alpha=0.8,
    title=None,
    xlabel=None,
    ylabel=None,
    label=None,
    ax=None,
    density=True,
    **kwargs
)
```

This function allows to plot satellite images histogram. 



**Parameters:**
 
 - <b>`image`</b>:  Optical images. It must be rasterio.io.DatasetReader with 3d or 2d.  


 - <b>`bands`</b>:  Must be specified as a number of a list.  


 - <b>`bins`</b>:  By default is 128.   


 - <b>`alpha`</b>:  Percentage (%) of transparency between 0 and 1. 0 indicates 0% and 1 indicates  100%. By default is 80%.  


 - <b>`title`</b>:  Assigned title. 


 - <b>`xlabel`</b>:  X axis title. 


 - <b>`ylabel`</b>:  Y axis title.  


 - <b>`label`</b>:  Labeling the histogram. 


 - <b>`ax`</b>:  current axes 


 - <b>`**kwargs`</b>:  These will be passed to the matplotlib imshow(), please see full lists at: 
 - <b>`https`</b>: //matplotlib.org/stable/api/_as_gen/matplotlib.axes.Axes.hist.html 

Return: 
 - <b>`ax`</b>:  A histogram of an image. 




---

<a href="..\scikeo\plot.py#L90"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>function</kbd> `plotRGB`

```python
plotRGB(
    image,
    bands=[3, 2, 1],
    stretch='std',
    title=None,
    xlabel=None,
    ylabel=None,
    ax=None,
    **kwargs
)
```

Plotting an image in RGB. 

This function allows to plot an satellite image in RGB channels.  



**Parameters:**
  


 - <b>`image`</b>:  Optical images. It must be rasterio.io.DatasetReader with 3d.  


 - <b>`bands`</b>:  A list contain the order of bands to be used in order to plot in RGB. For example,  for six bands (blue, green, red, nir, swir1 and swir2), number four (4) indicates   the swir1 band, number three (3) indicates the nir band and the number two (2) indicates  the red band.  


 - <b>`stretch`</b>:  Contrast enhancement using the histogram. There are two options here: i) using  standard deviation ('std') and ii) using percentiles ('per'). For default is 'std', which means  standard deviation.  


 - <b>`title`</b>:  Assigned title. 


 - <b>`xlabel`</b>:  X axis title. 


 - <b>`ylabel`</b>:  Y axis title. 


 - <b>`ax`</b>:  current axes 


 - <b>`**kwargs`</b>:  These will be passed to the matplotlib imshow(), please see full lists at: 
 - <b>`https`</b>: //matplotlib.org/stable/api/_as_gen/matplotlib.pyplot.imshow.html 

Return: 
 - <b>`ax`</b>:  Graphic of an image in RGB. 






---

_This file was automatically generated via [lazydocs](https://github.com/ml-tooling/lazydocs)._
