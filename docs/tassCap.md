<!-- markdownlint-disable -->

<a href="..\scikeo\tassCap.py#L0"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

# <kbd>module</kbd> `tassCap`





---

<a href="..\scikeo\tassCap.py#L5"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>function</kbd> `tassCap`

```python
tassCap(image, sat='Landsat8OLI', nodata=-99999, scale=None)
```

The Tasseled-Cap Transformation is a linear transformation method for various  remote sensing data. Not only can it perform volume data compression, but it can also provide parameters associated with the physical characteristics,  such as brightness, greenness and wetness indices. 



**Parameters:**
 


 - <b>`image`</b>:  Optical images. It must be rasterio.io.DatasetReader with 3d. 


 - <b>`sat`</b>:  Specify satellite and sensor type (Landsat5TM, Landsat7ETM or Landsat8OLI).  


 - <b>`nodata`</b>:  The NoData value to replace with -99999. 


 - <b>`scale`</b>:  Conversion of coefficients values 

Return: numpy.ndarray with 3d containing brightness, greenness and wetness indices. 

References: 

Crist, E.P., R. Laurin, and R.C. Cicone. 1986. Vegetation and soils information  contained in transformed Thematic Mapper data. Pages 1465-1470 Ref. ESA SP-254.  
 - <b>`European Space Agency, Paris, France. http`</b>: //www.ciesin.org/docs/005-419/005-419.html. 

Baig, M.H.A., Shuai, T., Tong, Q., 2014. Derivation of a tasseled cap transformation  based on Landsat 8 at-satellite reflectance. Remote Sensing Letters, 5(5), 423-431.  

Li, B., Ti, C., Zhao, Y., Yan, X., 2016. Estimating Soil Moisture with Landsat Data  and Its Application in Extracting the Spatial Distribution of Winter Flooded Paddies.  Remote Sensing, 8(1), 38. 



**Note:**

> Currently implemented for satellites such as Landsat-4 TM, Landsat-5 TM, Landsat-7 ETM+, Landsat-8 OLI and Sentinel2. The input data must be in top of atmosphere reflectance (toa). Bands required as input must be ordered as: 
>Consider using the following satellite bands: ===============   ================================ Type of Sensor     Name of bands ===============   ================================ Landsat4TM         :blue, green, red, nir, swir1, swir2 Landsat5TM         :blue, green, red, nir, swir1, swir2 Landsat7ETM+       :blue, green, red, nir, swir1, swir2 Landsat8OLI        :blue, green, red, nir, swir1, swir2 Landsat8OLI-Li2016 :coastal, blue, green, red, nir, swir1, swir2 Sentinel2MSI       :coastal, blue, green, red, nir-1, mir-1, mir-2 




---

_This file was automatically generated via [lazydocs](https://github.com/ml-tooling/lazydocs)._
