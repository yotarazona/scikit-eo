<!-- #region -->
# **Get Started**

This Get Started is intended as a guide to apply several remote sensing tools in order to analyze and process satellite imagery such as Landsat, Sentinel-2, etc. Various methods including ML/DL, Spectral Mixture Analysis, Calibrations methods, Principal Component Analysis, among others are available in this python package. 

# **Content**

[Example 01: Random Forest (RF) classifier](#example01)

[Example 02: Calibration methods for supervised classification](#example02)

[Example 03: Imagery Fusion - optical/radar](#example03)

[Example 04: Confusion matrix by estimated proportions of area with a confidence interval at 95%](#example04)

[Example 05: Linear trend analysis to map forest degradation](#example05)

# **Brief examples**

## <a name = "example01"></a>**Example 01: Random Forest (RF) classifier**

In this example, in a small region of southern Brazil, optical imagery from Landsat-8 OLI (Operational Land Imager) will be used to classify land cover using the machine learning algorithm Random Forest (RF) [(Breiman, 2001)](https://doi.org/10.1023/A:1010933404324). Four types of land cover will be mapped, i.e., agriculture, forest, bare soil and water. The input data needed is the satellite image and the spectral signatures collected. The output as a dictionary will provide: i) confusion matrix, ii) overall accuracy, iii) kappa index and iv) a classes map.

#### 01. Optical image to be used

Landsat-8 OLI (Operational Land Imager) will be used to obtain in order to classify using Random Forest (RF). This image, which is in surface reflectance with bands:

- Blue -> B2
- Green -> B3 
- Red -> B4
- Nir -> B5
- Swir1 -> B6
- Swir2 -> B7

The image and signatures to be used can be downloaded [here](https://drive.google.com/drive/folders/193RhNpACu9THcOZu8OzMh-btnFCOgHrU?usp=sharing):

#### 02. Libraries to be used in this example


```python
import rasterio
import numpy as np
from scikeo.mla import MLA
from scikeo.plot import plotRGB
import matplotlib.pyplot as plt
from dbfread import DBF
import matplotlib as mpl
import pandas as pd
```
#### 03. Image and endmembers
We will upload a satellite image as a *.tif* and endmembers as a *.dbf*. It is possible to extract endmembers values using the ```extract()``` function. In this case, upload only a shapefile (point feature) of samples without extracted spectral values. Please see [tutorials](https://yotarazona.github.io/scikit-eo/tutorials/) section for more details.

```python
path_raster = r"/home/data/LC08_232066_20190727_SR.tif"
img = rasterio.open(path_raster)

path_endm = r"home/data/endmembers.dbf"
endm = DBF(path_endm)

# endmembers
df = pd.DataFrame(iter(endm))
df.head()
```
<p align="left">
  <a href="https://github.com/yotarazona/scikit-eo"><img src="https://raw.githubusercontent.com/yotarazona/scikit-eo/main/docs/images/endembers.png" alt ="header" width = 75%>
</a>
</p>

#### 04. Classifying with Random Forest
An instance of ```mla()```:

```python
inst = MLA(image = img, endmembers = endm)
```
Applying with 70% of data to train:

```python
rf_class = inst.SVM(training_split = 0.7)
```
#### 5.0 Results

Dictionary of results

```python
rf_class.keys()
```
Overall accuracy

```python
rf_class.get('Overall_Accuracy')
```
Kappa index

```python
rf_class.get('Kappa_Index')
```
Confusion matrix or error matrix

```python
rf_class.get('Confusion_Matrix')
```
<p align="left">
  <a href="https://github.com/yotarazona/scikit-eo"><img src="https://raw.githubusercontent.com/yotarazona/scikit-eo/main/docs/images/confusion_matrix.png" alt ="header" width = 80%>
</a>
</p>

#### 06. Preparing the image before plotting

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

## <a name = "example02"></a>**Example 02: Calibration methods for supervised classification**

Given a large number of machine learning algorithms, it is necessary to select the one with the best performance in the classification, i.e., the algorithm in which the training and testing data used converge the learning iteratively to a solution that appears to be satisfactory [(Tarazona et al., 2021)](https://www.tandfonline.com/doi/full/10.1080/07038992.2021.1941823).
To deal with this, users can apply the calibration methods Leave One Out Cross-Validation (LOOCV), Cross-Validation (CV) and Monte Carlo Cross-Validation (MCCV) in order to calibrate a supervised classification with different algorithms. The input data needed are the spectral signatures collected as a *.dbf* or *.csv*. The output will provide a graph with the errors of each classifier obtained.

#### 01. Endmembers as a .dbf

```python
path_endm = "/home/data/ex_O2/endmembers.dbf"
endm = DBF(path_endm)
```
#### 02. An instance of calmla()

```python
inst = calmla(endmembers = endm)
```
#### 03. Applying the splitData() method

```python
data = inst.splitData()
```
**Calibrating with *Monte Carlo Cross-Validation Calibration* (MCCV)**

**Parameters**:

- ```split_data```: An instance obtained with ```splitData()```.
- ```models```: Support Vector Machine (svm), Decision Tree (dt), Random Forest (rf) and Naive Bayes (nb).
- ```n_iter```: Number of iterations.

#### 04. Running MCCV

```python
error_mccv = inst.MCCV(split_data = data, models = ('svm', 'dt', 'rf', 'nb'), 
                       n_iter = 10)
```

Calibration results:

![Result of the calibration methods using svm, dt, rf and nb.](images/scikit_eo_01.png){ width=90% }

With this result it can be observed that SVM and RF obtained a higher overall accuracy (less error). Therefore, you can use these algorithms to classify a satellite image.
<!-- #endregion -->

## <a name = "example03"></a>**Example 03: Imagery Fusion - optical/radar**

This is an area where **scikit-eo** provides a novel approach to merge different types of satellite imagery. We are in a case where, after combining different variables into a single output, we want to know the contributions of the different original variables in the data fusion. The fusion of radar and optical images, despite of its well-know use, to improve land cover mapping, currently has no tools that help researchers to integrate or combine those resources. In this third example, users can apply imagery fusion with different observation geometries and different ranges of the electromagnetic spectrum [(Tarazona et al., 2021)](https://www.tandfonline.com/doi/full/10.1080/07038992.2021.1941823). The input data needed are the optical satellite image and the radar satellite image, for instance.

In ```scikit-eo``` we developed the ```fusionrs()``` function which provides us with a dictionary with the following image fusion interpretation features:

- *Fused_images*: The fusion of both images into a 3-dimensional array (rows, cols, bands).
- *Variance*: The variance obtained.
- *Proportion_of_variance*: The proportion of the obtained variance.
- *Cumulative_variance*: The cumulative variance.
- *Correlation*: Correlation of the original bands with the principal components.
- *Contributions_in_%*: The contributions of each optical and radar band in the fusion.


#### 01. Loading dataset

Loading a radar and optical imagery with a total of 9 bands. Optical imagery has 6 bands Blue, Green, Red, NIR, SWIR1 and SWIR2, while radar imagery has 3 bands VV, VH and VV/VH.

```python
path_optical = "/home/data/ex_03/LC08_003069_20180906.tif"
optical = rasterio.open(path_optical)

path_radar = "/home/data/ex_03/S1_2018_VV_VH.tif"
radar = rasterio.open(path_radar)
```

#### 02. Applying the fusionrs:

```python
fusion = fusionrs(optical = optical, radar = radar)
```

#### 03. Dictionary of results:

```python
fusion.keys()
```

#### 04. Proportion of variance:

```python
prop_var = fusion.get('Proportion_of_variance')
```

#### 05. Cumulative variance (%):

```python
cum_var = fusion.get('Cumulative_variance')*100
```

#### 06. Showing the proportion of variance and cumulative:

```python
x_labels = ['PC{}'.format(i+1) for i in range(len(prop_var))]

fig, axes = plt.subplots(figsize = (6,5))
ln1 = axes.plot(x_labels, prop_var, marker ='o', markersize = 6,  
                label = 'Proportion of variance')

axes2 = axes.twinx()
ln2 = axes2.plot(x_labels, cum_var, marker = 'o', color = 'r', 
                 label = "Cumulative variance")

ln = ln1 + ln2
labs = [l.get_label() for l in ln]

axes.legend(ln, labs, loc = 'center right')
axes.set_xlabel("Principal Component")
axes.set_ylabel("Proportion of Variance")
axes2.set_ylabel("Cumulative (%)")
axes2.grid(False)
plt.show()
```

![Proportion of Variance and accumulative.](images/scikit_eo_02.png){ width=70% }


#### 07. Contributions of each variable in %:

```python
fusion.get('Contributions_in_%')
```

![Contributions of each variable in %.](images/scikit_eo_03.png){ width=90% }

Here, *var1*, *var2*, ... *var12* refer to *Blue*, *Green*, ... *VV/VH* bands respectively. It can be observed that *var2* (Green) has a higher contribution percentage 16.9% than other variables. With respect to radar polarizations, we can note that *var8* (VH polarization) has a higher contribution 11.8% than other radar bands.


#### 08. Preparing the image:

```python
arr = fusion.get('Fused_images')

## Let´s plot
fig, axes = plt.subplots(figsize = (8, 8))
plotRGB(arr, bands = [1,2,3], title = 'Fusion of optical and radar images')
plt.show()
```

![Fusion of optical and radar images. Principal Component 1 corresponds to red channel, Principal Component 2 corresponds to green channel and Principal Component 3 corresponds to blue channel.](images/scikit_eo_04.png){ width=55% }

<!-- #region -->
## <a name = "example04"></a>**Example 04: Confusion matrix by estimated proportions of area with a confidence interval at 95%**

In this final example, after obtaining the predicted class map, we are in a case where we want to know the uncertainties of each class. The assessing accuracy and area estimate will be obtained following guidance proposed by [(Olofsson et al., 2014)](https://doi.org/10.1016/j.rse.2014.02.015). All that users need are the confusion matrix and a previously obtained predicted class map.

In ```scikit-eo``` we developed the ```confintervalML``` function to estimate area and uncertainty with 95%. ```confintervalML``` requires the following parameters:

- *matrix*: confusion matrix or error matrix in numpy.ndarray.
- *image_pred*: a 2-dimensional array (rows, cols). This array should be the classified image with predicted classes.
- *pixel_size*: Pixel size of the classified image. Set by default as 10 meters. In this example is 30 meters (Landsat).
- *conf*: Confidence interval. By default is 95% (1.96).
- *nodata*: No data must be specified as 0, NaN or any other value. Keep in mind with this parameter.

<!-- #region -->
```python
#### 01. Load raster data
path_raster = r"/home/data/ex_O4/ml/predicted_map.tif"
img = rasterio.open(path_optical).read(1)

#### 02. Load confusion matrix as .csv
path_cm = r"/home/data/ex_O4/ml/confusion_matrix.csv"
values = pd.read_csv(path_radar)

#### 03. Applying the confintervalML:
confintervalML(matrix = values, image_pred = img, pixel_size = 30, conf = 1.96, 
               nodata = -9999)
```

Results:

![Estimating area and uncertainty with 95%.](images/scikit_eo_05.png){ width=80%}
<!-- #endregion -->
<!-- #endregion -->

## <a name = "example05"></a>**Example 05: Linear trend analysis to map forest degradation**

In this tutorial, forest degradation will be mapped using the Maximum Value Compositing (MVC) of Normalized Difference Vegetation Index (NDVI). Therefore, NDVI as a proxy of NPP makes it a useful indicator of resilience of ecosystems that allows its monitoring and evaluation through remote sensing, and it is the indicator we will use for the purposes of this example. 

Detecting forest degradation in terms of long-term loss of forest productivity of ecosystems through a Simple Linear Trend (SLT) is a standard method. Initiatives to monitor forest degradation using remote sensing techniques are few and urgent for the effects of conservation and restoration of degraded ecosystems in Peru. 

The study area is located in Madre de Dios, a region located in southeastern Peru. Madre de Dios is a biodiversity hotspot, with 40% of its area protected by a set of Natural Protected Areas and Native Communities, and it has the largest coal reserves in the world. 

The main drivers of forest degradation are economic activities such as gold mining, logging and agriculture. Gold mining represents the first large-scale economic activity in Madre de Dios.

<!-- #region -->
#### 01. Optical image to be used

NDVI from 2000 to 2015 will be used. These indices were obtained from Landsat-5 and Landsat-8 images. One per year.

The images as stack can be downloaded [here](https://drive.google.com/drive/folders/1mGSSg_sRFS3_SeNrR390w3UEMmMwHwg8?usp=sharing):

#### 02. Libraries to be used in this example


```python
import rasterio
import numpy as np
from scikeo.linearTrend import linearTrend
import matplotlib.pyplot as plt
import earthpy.plot as ep
```

Let´s plot the images (NDVI) from 2000 to 2015:

```python
ep.plot_bands(ts_img.read(), cmap = 'RdYlGn', cols = 4)
```
<!-- #endregion -->

<p align="left">
  <a href="https://github.com/yotarazona/scikit-eo"><img src="https://raw.githubusercontent.com/yotarazona/scikit-eo/main/docs/images/scikit_eo_06.png" alt ="header" width = "600">
</a>
</p>

<!-- #region -->
Instance of ```linearTrend```:

```python
inst = linearTrend(image = ts_img)
```

Applying the ```LN()``` function:

```python
trend = inst.LN()
```

Dictionary of esults:

```python
# dictionary as result
trend.keys()
```

Let´s plot the slope and p-value obtained:

```python
# Let´s plot
fig, axes = plt.subplots(nrows = 2, ncols = 2, figsize = (12, 9))
evi_2001 = axes[0][0].imshow(ts_img.read(1), cmap = 'RdYlGn')
axes[0][0].set_title("Enhanced Vegetation Index - 2000")
axes[0][0].grid(False)
bar1 = fig.colorbar(evi_2001, ax = axes[0][0])

evi_2015 = axes[0][1].imshow(ts_img.read(14), cmap = 'RdYlGn')
axes[0][1].set_title("Enhanced Vegetation Index  - 2015")
axes[0][1].grid(False)
bar2 = fig.colorbar(evi_2015, ax = axes[0][1])

slope = axes[1][0].imshow(trend.get('slope'), cmap = 'Spectral')
axes[1][0].set_title("Slope: Linear trend analysis")
axes[1][0].grid(False)
bar3 = fig.colorbar(slope, ax = axes[1][0])

pvalue = axes[1][1].imshow(trend.get('pvalue'), cmap = 'gist_rainbow')
axes[1][1].set_title("P-value: Linear trend Analysis")
axes[1][1].grid(False)
bar4 = fig.colorbar(pvalue, ax = axes[1][1])
```

<p align="left">
  <a href="https://github.com/yotarazona/scikit-eo"><img src="https://raw.githubusercontent.com/yotarazona/scikit-eo/main/docs/images/scikit_eo_07.png" alt ="header" width = "600">
</a>
</p>
<!-- #endregion -->

It is necessary to mention that forest degradation is expected to result in a statistically significant negative **slope** in the NDVI trend. The statistical robustness of the estimated trend was tested with a test from **ρ** (*P-value*) obtained from the simple regression function. *We defined the identification of degradation, with greater probability, at 95% of reliability*.
