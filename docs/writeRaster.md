<!-- markdownlint-disable -->

<a href="..\scikeo\writeRaster.py#L0"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

# <kbd>module</kbd> `writeRaster`





---

<a href="..\scikeo\writeRaster.py#L6"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>function</kbd> `writeRaster`

```python
writeRaster(arr, image, filename=None, filepath=None, n=1)
```

This algorithm allows to save array images to raster format (.tif). 



**Parameters:**
 


 - <b>`arr`</b>:  Array object with 2d (rows and cols) or 3d (rows, cols, bands). 


 - <b>`image`</b>:  Optical images. It must be read by rasterio.open(). 


 - <b>`filename`</b>:  The image name to be saved. 


 - <b>`filepath`</b>:  The path which the image will be stored. 


 - <b>`n`</b>:  Number of images to be saved. 

Return: A raster in your filepath. 




---

_This file was automatically generated via [lazydocs](https://github.com/ml-tooling/lazydocs)._
