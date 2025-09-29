<!-- #region -->
# scikit-eo: A Python package for Remote Sensing Data Analysis

[![Google Colab](https://colab.research.google.com/assets/colab-badge.svg)]()
[![License: MIT](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)
[![PythonVersion]( https://img.shields.io/badge/python-3.5%20%7C%203.6%20%7C%203.7%20%7C%203.8-green)]()
[![PyPI version](https://badge.fury.io/py/scikeo.svg)](https://badge.fury.io/py/scikeo)
[![Youtube](https://img.shields.io/badge/YouTube-Channel-red)]()
[![Downloads](https://static.pepy.tech/badge/scikeo)](https://pepy.tech/project/scikeo)
[![Downloads](https://raw.githubusercontent.com/yotarazona/scikit-eo/main/docs/images/docs-passing-brightgreen.svg)]()
[![tests](https://github.com/yotarazona/scikit-eo/actions/workflows/tests.yml/badge.svg)](https://github.com/yotarazona/scikit-eo/actions/workflows/tests.yml)
<a href="https://joss.theoj.org/papers/46bccc5be81d7ea886e05807cfe6790c"><img src="https://joss.theoj.org/papers/46bccc5be81d7ea886e05807cfe6790c/status.svg"></a>



<img src="https://raw.githubusercontent.com/yotarazona/scikit-eo/main/docs/images/scikit-eo_logo.jpg" align="right" width="220"/>

<!-- #region -->
# Introduction

Nowadays, remotely sensed data has increased dramatically. Microwaves and optical images with different spatial and temporal resolutions are available and are used to monitor a variety of environmental issues such as deforestation, land degradation, land use and land cover change, among others. Although there are efforts (i.e., Python packages, forums, communities, etc.) to make available line-of-code tools for pre-processing, processing and analysis of satellite imagery, there is still a gap that needs to be filled. In other words, too much time is still spent by many users developing Python lines of code. Algorithms for mapping land degradation through a linear trend of vegetation indices, fusion optical and radar images to classify vegetation cover, and calibration of machine learning algorithms, among others, are not available yet.

Therefore, [scikit-eo](https://github.com/yotarazona/scikit-eo) is a Python package that provides tools for remote sensing. This package was developed to fill the gaps in remotely sensed data processing tools. Most of the tools are based on scientific publications, and others are useful algorithms that will allow processing to be done in a few lines of code. With these tools, the user will be able to invest time in analyzing the results of their data and not spend time on elaborating lines of code, which can sometimes be stressful.

# Tools for Remote Sensing

| Name of functions/classes  | Description|
| -------------------| --------------------------------------------------------------------------|
| **`mla`**          | Machine Learning (Random Forest, Support Vector Machine, Decition Tree, Naive Bayes, Neural Network, etc.)                                                          |
| **`calmla`**       | Calibrating supervised classification in Remote Sensing (e.g., Monte Carlo Cross-Validation, Leave-One-Out Cross-Validation, etc.)                   |
| **`confintervalML`**       | Information of confusion matrix by proportions of area, overall accuracy, user's accuracy with confidence interval and estimated area with confidence interval as well.                                    |
| **`rkmeans`**      | K-means classification                                                    |
| **`calkmeans`**    | This function allows to calibrate the kmeans algorithm. It is possible to obtain the best k value and the best embedded algorithm in kmeans.                               |
| **`pca`**          | Principal Components Analysis                                             |
| **`atmosCorr`**    | Atmospheric Correction of satellite imagery                               |
| **`deepLearning`** | Deep Learning algorithms                                                  |
| **`linearTrend`**  | Linear trend is useful for mapping forest degradation or land degradation |
| **`fusionrs`**     | This algorithm allows to fuse images coming from different spectral sensors (e.g., optical-optical, optical and SAR or SAR-SAR). Among many of the qualities of this function, it is possible to obtain the contribution (%) of each variable in the fused image |
| **`sma`**          | Spectral Mixture Analysis - Classification sup-pixel                      |
| **`tassCap`**      | The Tasseled-Cap Transformation                                           |

You will find more algorithms!.

# Dependencies used by **scikit-eo**

All dependencies used by ```scikit-eo``` are as follows:

```numpy```, ```pandas```, ```matplotlib```, ```rasterio```, ```seaborn```, ```statsmodels```, ```scikit-learn```, ```scipy```, ```pytest```, ```dbfread```, ```fiona``` and ```geopandas```. By installing ```scikit-eo``` all these packages will be installed!.

# Installation

To use ```scikit-eo``` it is necessary to install it. There are two options:

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

## 2.0 Optical image

Landsat-8 OLI (Operational Land Imager) will be used to obtain in order to classify using Random Forest (RF). This image, which is in surface reflectance with bands:
- Blue -> B2
- Green -> B3 
- Red -> B4
- Nir -> B5
- Swir1 -> B6
- Swir2 -> B7

This image will be downloaded using the following codes:

```python
import requests, zipfile
from io import BytesIO

# Defining the zip file URL
url = 'https://github.com/yotarazona/data/raw/main/data/01_machine_learning.zip'

# Split URL to get the file name
filename = url.split('/')[-1]

# Downloading the file by sending the request to the URL
req = requests.get(url)

# extracting the zip file contents
file = zipfile.ZipFile(BytesIO(req.content))
file.extractall()
```

## 3.0 Supervised Classification using Random Forest

Image and endmembers

```python
data_dir = "01_machine_learning/"

# satellite image
path_raster = data_dir + "LC08_232066_20190727_SR.tif"
img = rasterio.open(path_raster)

# endmembers
path_endm = data_dir + "endmembers.shp"
endm = gpd.read_file(path_endm)
```

```python
# endmembers
endm = extract(img, endm)
endm
```
<p align="left">
  <a href="https://github.com/yotarazona/scikit-eo"><img src="https://raw.githubusercontent.com/yotarazona/scikit-eo/main/docs/images/endembers.png" alt ="header" width = 75%>
</a>
</p>


Instance of ```mla()```:

```python
inst = MLA(image = img, endmembers = endm)
```

Applying Random Forest:

```python
svm_class = inst.SVM(training_split = 0.7)
```

## 4.0 Results

Dictionary of results

```python
svm_class.keys()
```

Overall accuracy

```python
svm_class.get('Overall_Accuracy')
```

Kappa index

```python
svm_class.get('Kappa_Index')
```

Confusion matrix or error matrix

```python
svm_class.get('Confusion_Matrix')
```

<p align="left">
  <a href="https://github.com/yotarazona/scikit-eo"><img src="https://raw.githubusercontent.com/yotarazona/scikit-eo/main/docs/images/confusion_matrix.png" alt ="header" width = 80%>
</a>
</p>


Preparing the image before plotting

```python
# Let's define the color palette
palette = mpl.colors.ListedColormap(["#2232F9","#F922AE","#229954","#7CED5E"])
```

Applying the ```plotRGB()``` algorithm is easy:

```python
# Let´s plot
fig, axes = plt.subplots(nrows = 1, ncols = 2, figsize = (15, 9))

# satellite image
plotRGB(img, title = 'Image in Surface Reflectance', ax = axes[0])

# class results
axes[1].imshow(svm_class.get('Classification_Map'), cmap = palette)
axes[1].set_title("Classification map")
axes[1].grid(False)
```

<p align="left">
  <a href="https://github.com/yotarazona/scikit-eo"><img src="https://raw.githubusercontent.com/yotarazona/scikit-eo/main/docs/images/classification.png" alt ="header" width = "750">
</a>
</p>

This result shows us how we can use the ```scikit-eo``` python package in order to obtaind a Land Cover map. Following this tutorial is it possible to use different algorithms such as Support Vector Machine, Decision Tree, Neural Networks, amont others.


-   Free software: Apache Software License 2.0
-   Documentation: 

## Acknowledgment

Special thanks to:
- [David Montero Loaiza](https://github.com/davemlz) for the idea of the package name **scikit-eo**.

- [Qiusheng Wu](https://github.com/giswqs) for the suggestions that helped to improve the package.

## Credits

This package was created with [Cookiecutter](https://github.com/cookiecutter/cookiecutter)

## Logo inspiration

The logo on the package is inspired by fog oasis known as "lomas".
