<!-- #region -->
# scikit-eo: A Python package for Remote Sensing Data Analysis


<!-- #region -->
# Introduction

Nowadays, remotely sensed data has increased dramatically. Microwaves and optical images with different spatial and temporal resolutions are available and are used to monitor a variety of environmental issues such as deforestation, land degradation, land use and land cover change, among others. Although there are efforts (i.e., Python packages, forums, communities, etc.) to make available line-of-code tools for pre-processing, processing and analysis of satellite imagery, there is still a gap that needs to be filled. In other words, too much time is still spent by many users developing Python lines of code. Algorithms for mapping land degradation through a linear trend of vegetation indices, fusion optical and radar images to classify vegetation cover, and calibration of machine learning algorithms, among others, are not available yet.

Therefore, [scikit-eo](https://github.com/yotarazona/scikit-eo) is a Python package that provides tools for remote sensing. This package was developed to fill the gaps in remotely sensed data processing tools. Most of the tools are based on scientific publications, and others are useful algorithms that will allow processing to be done in a few lines of code. With these tools, the user will be able to invest time in analyzing the results of their data and not spend time on elaborating lines of code, which can sometimes be stressful.


## 1. From PyPI

**scikit-eo** is available on [PyPI](https://pypi.org/project/scikeo/), so to install it, run this command in your terminal:

```python
pip install scikeo
```

## 2. Installing from source

It is also possible to install the latest development version directly from the GitHub repository with:

```python
pip install git+https://github.com/yotarazona/scikit-eo
```

## containerizing ```scikit-eo```

**Note**: It is a recommended practice to provide some instructions for isolating/containerizing ```scikit-eo```. It would benefit their use and thus avoid that some dependencies are not compatible with others. For example, conda provides an easy solution.

```python
conda create -n scikiteo python = 3.8
```
Then, activate the environment created

```python
conda activate scikiteo
```
Then finally, ```scikit-eo``` can be install within this new environment using via PyPI or from the GitHub repository.

<!-- #endregion -->

# Example

## 1.0 Applying Machine Learning

Libraries to be used:

```python
import rasterio
import numpy as np
from scikeo.mla import MLA
from scikeo.process import extract
import matplotlib.pyplot as plt
import matplotlib as mpl
import geopandas as gpd
from scikeo.plot import plotRGB
from scikeo.writeRaster import writeRaster
```

