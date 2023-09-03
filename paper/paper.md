---
title: 'scikit-eo: A Python package for Remote Sensing Data Analysis'
tags:
  - Python
  - Remote Sensing
  - Earth Observation
  - Machine Learning
  - Deep Learning
  - Spatial Analysis
authors:
  - name: Yonatan Tarazona
    orcid: 0000-0002-5208-1004
    affiliation: 1
  - name: Jakub Nowosad
    affiliation: 2
    orcid: 0000-0002-1057-3721
  - name: Fernando Benitez-Paez
    affiliation: 3
    orcid: 0000-0002-9884-6471
  - name: Fabian Drenkhan
    orcid: 0000-0002-9443-9596
    affiliation: 4
  - name: Martín E. Timaná
    orcid: 0000-0003-1559-4449
    affiliation: 1
affiliations:
  - name: Applied Geography Research Center, Department of Humanities, Pontificia Universidad Católica del Perú, Lima, Peru
    index: 1
  - name: Adam Mickiewicz University in Poznań
    index: 2
  - name: The school of geography and sustainable development, University of St Andrews
    index: 3
  - name: Geography and the Environment, Department of Humanities, Pontificia Universidad Católica del Perú, Lima, Peru
    index: 4
date: 19 May 2023
bibliography: paper.bib
---

<!-- #region -->
# Summary

In recent years the availability of remote sensing data has increased substantially. A growing body of spaceborne and drone imagery including optical and radar imagery with increasing spatial (i.e. sub-metric) and temporal resolutions is now available. This remotely sensed data has enabled researchers to address and tackle a broader range of challenges effectively, by using novel tools and data at multiple scales that help to shed new light into the underlying factors that contribute to both local and global issues. 

As more accessible the data is, the bigger the need is for open-source tools to read, process and execute analysis that contributes to underpin patterns, changes and trends that are critical for environmental studies. Applications that integrate spatial-temporal data are used to map, analyze and monitor a variety of complex environmental processes and impacts, such as monitoring and assessment land cover changes [@Chaves2020], crop classifications [@POTT2021196], deforestation [@TARAZONA2018367], impact on urbanization level [@Trinder2020], climate change impacts [@Yang2013]  including assessments of glacier retreat [@Hugonnet2021] and related hydrological change [@Huss2018], biodiversity conservation [@Jeannine2022], and natural disaster management [@Kucharczyk2021].

But more availability of data, does not necessary means easier way to process the data. Analysts spend an important amount of time finding the correct libraries that allow them to read and process the remotely sensed data. 

Although there are multiple efforts to make available command line tools for selecting, processing and analyzing satellite imagery, tools for mapping land degradation through linear trend of vegetation indices [@TARAZONA2020100337], data fusion process for optical and radar images to quickly classify the vegetation cover, and the integration of machine learning techniques are still separate in multiple libraries, along with diffuse documentation that limits the analysis where the main concerns of the causes or patterns of the environmental conditions should be the main focus.

Therefore, here we present **scikit-eo**, a new Python package that provides the necessary tools for remote sensing analysis. It is a centralized, scalable and open source toolkit, developed to fill the gaps in remotely sensed data processing tools. This toolkit can be used in multiple venues from a lecturer room as a toolkit for introduction of programming skills using python and remote sensing for environmental studies, or can be used as part of any Python setting in a research project. The majority of the tools included in **scikit-eo** are derived from scientific publications, while others are valuable algorithms that streamline data processing into just a few lines of code. By integrating this set of diverse tools, users can focus their time and energy on analyzing the results of their data, rather than being challenged by the intricacies lines of code. By centralizing, integrating use cases and example of data, **scikit-eo** serves as a way to allow researchers to optimize their resources and dedicate their focus to the meaningful interpretation of their findings with greater efficiency.

# Highlights

**scikit-eo** is an open-source package built entirely in Python, through Object-Oriented Programming and Structured Programming, that provides a useful variety of remote sensing tools, from basic and exploratory functions to more advanced methods to classify, calibrate, or fusing satellite imagery. Depending on users' needs, **scikit-eo** can provide the basic but essential land cover characterization mapping including the confusion matrix and the required metrics such as user's accuracy, producer's accuracy, omissions and commissions errors. The combination of these required metrics can be obtained in a form of a pandas ```DataFrame``` object. Furthermore, a class prediction map as a result of land cover mapping, i.e., a land cover map which represents the output of the classification algorithm or the output of the segmentation algorithm. These two outcomes must include uncertainties with a confidence levels (e.g., at $95$% or $90$%). All required metrics from the confusion matrix can be easily computed and included confidence levels with **scikit-eo** following guidance proposed by @OLOFSSON201442. There are other useful tools for remote sensing analysis that can be found in this package, for more information about the full list of the supported function as well as how to install and execute the package within a python setting, visit the [scikit-eo](https://ytarazona.github.io/scikit-eo/) website.

# Audience

**scikit-eo** is a versatile Python package designed to cover a wide range of users, including students, professionals of remote sensing, researchers of environmental analysis, and organizations looking for satellite image analysis. Its comprehensive features make it well-suited for various applications, such as university teaching, that include technical and practical sessions, and cutting-edge research using the most recent machine learning and deep learning techniques applied to the field of remote sensing. Whether the user are students seeking to get insights from a satellite image analysis or a experienced researcher looking for advanced tools, **scikit-eo** offers a valuable resource to support the most valuable methods for environmental studies.

### **scikit-eo** as Research tool:

In the field of environmental studies, particularly in the study of land cover classification, the availability of scalable but user-friendly software tools is crucial for facilitating research, analysis, and modeling. With the widespread adoption of Python as one of the most popular programming language, particularly in the GIScience and remote sensing fields, the development of specialized packages has greatly enhanced the effectiveness of environmental research. **scikit-eo** is a dedicated piece of software tailored to address unique challenges encountered in the land cover mapping, forest or land degradation as well the fusion of multiple satellite imagery from several formats [@Tarazona2021]. **scikit-eo** provides the assessment and calibration metrics to evaluate the provided outputs. In the current version **scikit-eo** requires that users provide the dataset to process but we expect in the near future provide a wide range of functionalities for acquiring environmental data from diverse sources and file formats, enabling researchers to access satellite imagery.

One of the key strengths of **scikit-eo** lies in its advanced analysis capabilities. It provides a rich suite of algorithms specifically designed for environmental studies. These include statistical analysis, machine learning, deep learning, data fusion and spatial analysis. Researchers can leverage these tools to explore patterns, relationships, and trends within their datasets, to uncover complex land or forest degradation or mapping and classify the land cover, and generate insightful visualizations.

### **scikit-eo** in the lecture room:

 **scikit-eo** can be part of a lecturer room as part of the set of tools for environmental studies where a quantitative approach and computer labs are required. After the appropriate introduction of Python, basics of remote sensing, and the relevance of environmental studies to address climate change challenges or impacts of anthropogenic activity. Lectures can take advance of the simplicity of **scikit-eo** routines to execute supervised classification methods, Principal Components Analysis, Spectral Mixture Analysis, Mapping forest or land degradation and more types of analysis. By reducing the number of required lines of code, students can focus on the analysis and how the methods work rather of dealing with complex, and unnecessary programming tasks. Lecturers can structure their computer labs using open data sources and integrate **scikit-eo** to allow students understand the importance of the calibration and assessment metrics, get insights about the classification mapping suing satellite imagery as well as providing an introduction to more advances methods that include machine learning techniques.

### **scikit-eo** as open source tool:

As open source software keeps transforming the landscape of scientific research [@Community2019], enabling collaboration, reproducibility and transparency, **scikit-eo** was specifically developed as an open source tool. **scikit-eo** integrates most of the popular open source python libraries from the so-called geo-python stack for remote sensing (e.g,. [numpy](https://numpy.org/), [pandas](https://pandas.pydata.org/), [rasterio](https://rasterio.readthedocs.io/en/stable/) and few more) to extent and create a centralise package for advanced spatial analysis of remotely sensed data. The package provides researchers and developers with a free, scalable, and community-driven platform to process, analyze, and visualize satellite imagery more specifically, built upon the most popular python libraries, but centralizes the use of multiple functions for classification and mapping land cover.

# Functionalities


## Main tools

**Scikit-eo** includes several algorithms to process satellite images to assess complex environmental processes and impacts. These include major functions, such as atmospheric correction, machine learning and deep learning techniques, estimating area and uncertainty, linear trend analysis, combination of optical and radar images, classification sub-pixel, to name a few. Some main functions are listed below:

| Name of functions/classes  | Description|
| -------------------| --------------------------------------------------------------------------|
| **`mla`**          | Supervised Classification in Remote Sensing                               |
| **`calmla`**       | Calibrating Supervised Classification in Remote Sensing                   |
| **`confintervalML`**       | Information of Confusion Matrix by proportions of area, overall accuracy, user's accuracy with confidence interval and estimated area with confidence interval as well.                         |
| **`deepLearning`** | Deep Learning algorithms                                                  |
| **`atmosCorr`**    | Radiometric and Atmospheric Correction                              |
| **`rkmeans`**      | K-means classification                                                    |
| **`calkmeans`**    | This function allows to calibrate the kmeans algorithm. It is possible to obtain the best k value and the best embedded algorithm in kmeans.                               |
| **`pca`**          | Principal Components Analysis                                             |
| **`linearTrend`**  | Linear trend is useful for mapping forest degradation or land degradation |
| **`fusionrs`**     | This algorithm allows to fuse images coming from different spectral sensors (e.g., optical-optical, optical and SAR or SAR-SAR). Among many of the qualities of this function, it is possible to obtain the contribution (%) of each variable in the fused image |
| **`sma`**          | Spectral Mixture Analysis - Classification sup-pixel                      |
| **`tassCap`**      | The Tasseled-Cap Transformation                                           |

: Main tools available for **scikit-eo** package. \label{table:1}

Of course, there are others functions will be found in the package. 
<!-- #endregion -->

# Acknowledgments

The authors would like to thank to David Montero Loaiza for the idea of the package name and Qiusheng Wu for the suggestions that helped to improve the package.

# References
