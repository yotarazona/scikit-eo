# Welcome to

# **scikit-eo: A Python package for Remote Sensing Tools**

[![Google Colab](https://colab.research.google.com/assets/colab-badge.svg)]()
[![License: MIT](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)
[![PythonVersion]( https://img.shields.io/badge/python-3.5%20%7C%203.6%20%7C%203.7%20%7C%203.8-green)]()
[![PyPI version](https://badge.fury.io/py/scikeo.svg)](https://badge.fury.io/py/scikeo)
[![Youtube](https://img.shields.io/badge/YouTube-Channel-red)]()
[![Downloads](https://pepy.tech/badge/scikeo)](https://pepy.tech/project/scikeo)
[![Downloads](https://raw.githubusercontent.com/ytarazona/scikit-eo/main/docs/images/docs-passing-brightgreen.svg)]()
[![tests](https://github.com/ytarazona/scikit-eo/actions/workflows/tests.yml/badge.svg)](https://github.com/ytarazona/scikit-eo/actions/workflows/tests.yml)

<p align="center">
  <a href="https://github.com/ytarazona/scikit-eo"><img src="https://raw.githubusercontent.com/ytarazona/scikit-eo/main/docs/images/scikit-eo_logo.jpg" alt="header" width = '180'></a>
</p>


### Links of interest:

- **GitHub repo**: <https://github.com/ytarazona/scikit-eo>
- **Documentation**: <https://ytarazona.github.io/scikit-eo/>
- **PyPI**: <https://pypi.org/project/scikeo/>
- **Notebooks examples**: <https://github.com/ytarazona/scikit-eo/tree/main/examples>
- **Google Colab examples**: <https://github.com/ytarazona/scikit-eo/tree/main/examples>
- **Free software**: [Apache 2.0](https://opensource.org/license/apache-2-0/)
- **Tutorials: step by step**: <https://ytarazona.github.io/scikit-eo/tutorials/>

# **Introduction**

Nowadays, remotely sensed data has increased dramatically. Microwaves and optical images with different spatial and temporal resolutions are available and are used to monitor a variety of environmental issues such as deforestation, land degradation, land use and land cover change, among others. Although there are efforts (i.e., Python packages, forums, communities, etc.) to make available line-of-code tools for pre-processing, processing and analysis of satellite imagery, there is still a gap that needs to be filled. In other words, too much time is still spent by many users developing Python lines of code. Algorithms for mapping land degradation through a linear trend of vegetation indices, fusion optical and radar images to classify vegetation cover, and calibration of machine learning algorithms, among others, are not available yet.

Therefore, **scikit-eo** is a Python package that provides tools for remote sensing. This package was developed to fill the gaps in remotely sensed data processing tools. Most of the tools are based on scientific publications, and others are useful algorithms that will allow processing to be done in a few lines of code. With these tools, the user will be able to invest time in analyzing the results of their data and not spend time on elaborating lines of code, which can sometimes be stressful.

# **Audience**

**Scikit-eo** is a versatile Python package designed to cover a wide range of users, including students, professionals of remote sensing, researchers of environmental analysis, and organizations looking for satellite image analysis. Its comprehensive features make it well-suited for various applications, such as university teaching, that include technical and practical sessions, and cutting-edge research using the most recent machine learning and deep learning techniques applied to the field of remote sensing. Whether the user are students seeking to get insights from a satellite image analysis or a experienced researcher looking for advanced tools, scikit-eo offers a valuable resource to support the most valuable methods for environmental studies.

# **Tools for Remote Sensing**

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

<!-- #region -->
# **Installation**

To use **scikit-eo** it is necessary to install it. There are two options:

## 1. From PyPI

```python
pip install scikeo
```

## 2. Installing from source

It is also possible to install the latest development version directly from the GitHub repository with:

```python
pip install git+https://github.com/ytarazona/scikit-eo
```
<!-- #endregion -->


