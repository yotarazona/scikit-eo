<!-- markdownlint-disable -->

<a href="..\scikeo\atmosCorr.py#L0"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

# <kbd>module</kbd> `atmosCorr`






---

<a href="..\scikeo\atmosCorr.py#L8"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>class</kbd> `atmosCorr`
Atmospheric Correction in Optical domain 

<a href="..\scikeo\atmosCorr.py#L12"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `__init__`

```python
__init__(path, nodata=-99999)
```

Parameter: 

 path: String. The folder in which the satellite bands are located. This images could be Landsat  Collection 2 Level-1. For example: path = r'/folder/image/raster'.  

 nodata: The NoData value to replace with -99999. 




---

<a href="..\scikeo\atmosCorr.py#L203"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `DOS`

```python
DOS(sat='LC08', mindn=None)
```

The Dark Object Subtraction Method was proposed by Chavez (1988). This image-based  atmospheric correction method considers absolutely critical and valid the existence  of a dark object in the scene, which is used in the selection of a minimum value in  the haze correction. The most valid dark objects in this kind of correction are areas  totally shaded or otherwise areas representing dark water bodies. 



**Parameters:**
 


 - <b>`sat`</b>:  Type of Satellite. It could be Landsat-5 TM, Landsat-8 OLI or Landsat-9 OLI-2. 


 - <b>`mindn`</b>:  Min of digital number for each band in a list. 

Return:  An array with Surface Reflectance values with 3d, i.e. (rows, cols, bands). 

References: 

Chavez, P.S. (1988). An Improved Dark-Object Subtraction Technique for Atmospheric  Scattering Correction of Multispectral Data. Remote Sensing of Envrironment, 24(3), 459-479. 

---

<a href="..\scikeo\atmosCorr.py#L48"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `RAD`

```python
RAD(sat='LC08')
```

Conversion to TOA Radiance. Landsat Level-1 data can be converted to TOA spectral radiance  using the radiance rescaling factors in the MTL file: 

Lλ = MLQcal + AL  

where: 

Lλ = TOA spectral radiance (Watts/(m2*srad*μm)) ML = Band-specific multiplicative rescaling factor from the metadata (RADIANCE_MULT_BAND_x, where x is the band number) AL = Band-specific additive rescaling factor from the metadata (RADIANCE_ADD_BAND_x, where x is the band number) Qcal =  Quantized and calibrated standard product pixel values (DN)  



**Parameters:**
 
 - <b>`sat`</b>:  Type of Satellite. It could be Landsat-5 TM, Landsat-8 OLI or Landsat-9 OLI-2. 

Return:  An array with radiance values with 3d, i.e. (rows, cols, bands). 

---

<a href="..\scikeo\atmosCorr.py#L126"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `TOA`

```python
TOA(sat='LC08')
```

A reduction in scene-to-scene variability can be achieved by converting the at-sensor  spectral radiance to exoatmospheric TOA reflectance, also known as in-band planetary albedo. 

Equation to obtain TOA reflectance: 

ρλ′ = Mρ*DN + Aρ 

ρλ = ρλ′/sin(theta) 



**Parameters:**
 
 - <b>`sat`</b>:  Type of Satellite. It could be Landsat-5 TM, Landsat-8 OLI or Landsat-9 OLI-2. 

Return:  An array with TOA values with 3d, i.e. (rows, cols, bands). 




---

_This file was automatically generated via [lazydocs](https://github.com/ml-tooling/lazydocs)._
